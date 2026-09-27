from __future__ import annotations

import base64
import asyncio
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from mcp.types import EmbeddedResource

from main import (
    RequestError,
    RenderJobManager,
    decode_blend_base64,
    download_render_frames,
    download_render_result,
    parse_nvidia_smi_csv,
    start_tailscale,
    validate_render_request,
    validate_frame_archive,
    validate_frame_list,
    validate_mp4_size,
    submit_frame_render_job,
)


class DecodeBlendTests(unittest.TestCase):
    def test_accepts_blend_header(self) -> None:
        contents = b"BLENDER-v300"
        self.assertEqual(decode_blend_base64(base64.b64encode(contents).decode()), contents)

    def test_accepts_zstandard_compressed_blend_signature(self) -> None:
        contents = b"\x28\xb5\x2f\xfd" + b"compressed blend payload"
        self.assertEqual(decode_blend_base64(base64.b64encode(contents).decode()), contents)

    def test_rejects_invalid_base64_and_non_blend_data(self) -> None:
        with self.assertRaisesRegex(RequestError, "valid base64"):
            decode_blend_base64("not base64!")
        with self.assertRaisesRegex(RequestError, "header"):
            decode_blend_base64(base64.b64encode(b"not a blend").decode())

    def test_enforces_decoded_size_limit(self) -> None:
        encoded = base64.b64encode(b"BLENDER" + b"x" * 8).decode()
        with self.assertRaisesRegex(RequestError, "exceeds"):
            decode_blend_base64(encoded, max_bytes=10)


class GpuAndRenderValidationTests(unittest.TestCase):
    def test_parses_nvidia_inventory(self) -> None:
        output = 'NVIDIA RTX 6000 Ada, GPU-123, 49140, 48000, 575.64\n'
        self.assertEqual(
            parse_nvidia_smi_csv(output),
            [
                {
                    "name": "NVIDIA RTX 6000 Ada",
                    "uuid": "GPU-123",
                    "memory_total_mib": "49140",
                    "memory_free_mib": "48000",
                    "driver_version": "575.64",
                }
            ],
        )

    def test_cycles_cpu_is_allowed_without_gpu(self) -> None:
        validate_render_request("cycles", False, None, None, None, None, None, None)

    def test_gpu_engines_are_rejected_without_gpu(self) -> None:
        for engine in ("eevee", "workbench"):
            with self.subTest(engine=engine), self.assertRaisesRegex(RequestError, "requires a GPU"):
                validate_render_request(engine, False, None, None, None, None, None, None)

    def test_gpu_engines_are_allowed_when_gpu_is_visible(self) -> None:
        for engine in ("eevee", "workbench"):
            with self.subTest(engine=engine):
                validate_render_request(engine, True, None, None, None, None, None, None)

    def test_gpu_engine_capability_check_can_be_deferred_to_job_worker(self) -> None:
        validate_render_request("eevee", None, None, None, None, None, None, None)

    def test_rejects_invalid_ranges_and_resolution(self) -> None:
        with self.assertRaisesRegex(RequestError, "frame_end"):
            validate_render_request("cycles", False, 5, 2, None, None, None, None)
        with self.assertRaisesRegex(RequestError, "resolution_x"):
            validate_render_request("cycles", False, None, None, 0, None, None, None)

    def test_enforces_mp4_size_limit_without_loading_file(self) -> None:
        validate_mp4_size(250 * 1024 * 1024)
        with self.assertRaisesRegex(RequestError, "MP4 output exceeds"):
            validate_mp4_size(250 * 1024 * 1024 + 1)

    def test_validates_frame_lists(self) -> None:
        validate_frame_list([1, 5, -2])
        with self.assertRaisesRegex(RequestError, "non-empty"):
            validate_frame_list([])
        with self.assertRaisesRegex(RequestError, "integer"):
            validate_frame_list([1, True])
        with self.assertRaisesRegex(RequestError, "duplicate"):
            validate_frame_list([3, 3])
        with self.assertRaisesRegex(RequestError, "at most"):
            validate_frame_list(list(range(101)))

    def test_validates_frame_archive_contents_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "frames.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("frame_000001.png", b"\x89PNG\r\n\x1a\nimage-one")
                archive.writestr("frame_000007.png", b"\x89PNG\r\n\x1a\nimage-two")

            self.assertEqual(validate_frame_archive(archive_path, [1, 7]), archive_path.stat().st_size)
            with self.assertRaisesRegex(RequestError, "requested PNG"):
                validate_frame_archive(archive_path, [1, 2])
            with self.assertRaisesRegex(RequestError, "exceeds"):
                validate_frame_archive(archive_path, [1, 7], max_bytes=1)

            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("frame_000001.png", b"not a png file")
            with self.assertRaisesRegex(RequestError, "invalid PNG"):
                validate_frame_archive(archive_path, [1])


class RenderJobManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_returns_id_and_status_reports_frame_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RenderJobManager(Path(temp_dir), max_active_jobs=1)
            release_render = asyncio.Event()

            async def fake_render(job) -> None:
                job.progress_path.write_text(
                    json.dumps({"processed_frames": 2, "total_frames": 5}),
                    encoding="utf-8",
                )
                job.output_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")
                await release_render.wait()

            manager._render_with_blender = fake_render
            job = await manager.submit(
                b"BLENDER-test",
                {"engine": "cycles", "frame_start": 1, "frame_end": 5},
            )

            self.assertTrue(job.job_id)
            self.assertEqual(job.status, "queued")
            await asyncio.sleep(0)
            status = manager.status(job)
            self.assertEqual(status["status"], "running")
            self.assertEqual(status["processed_frames"], 2)
            self.assertEqual(status["total_frames"], 5)
            self.assertEqual(status["progress_percent"], 40.0)
            self.assertGreaterEqual(status["running_time_seconds"], 0)

            release_render.set()
            await job.task
            completed = manager.status(job)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["processed_frames"], 5)
            self.assertTrue(completed["result_available"])

    async def test_active_job_limit_rejects_extra_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RenderJobManager(Path(temp_dir), max_active_jobs=1)
            release_render = asyncio.Event()

            async def fake_render(job) -> None:
                job.output_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")
                await release_render.wait()

            manager._render_with_blender = fake_render
            job = await manager.submit(b"BLENDER-one", {"engine": "cycles"})
            await asyncio.sleep(0)
            with self.assertRaisesRegex(RequestError, "queue is full"):
                await manager.submit(b"BLENDER-two", {"engine": "cycles"})
            release_render.set()
            await job.task

    async def test_gpu_engine_failure_is_reported_in_job_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RenderJobManager(Path(temp_dir), max_active_jobs=1)
            cpu_only_info = {
                "gpu_available": False,
                "render_engines": {"cycles": {"backend": "CPU"}},
            }
            with patch("main.collect_gpu_info", return_value=cpu_only_info):
                job = await manager.submit(b"BLENDER-test", {"engine": "eevee"})
                await job.task

            status = manager.status(job)
            self.assertEqual(status["status"], "failed")
            self.assertIn("eevee rendering requires a GPU", status["error"])

    async def test_completed_job_can_be_downloaded_as_mp4_resource(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RenderJobManager(Path(temp_dir), max_active_jobs=1)

            async def fake_render(job) -> None:
                job.output_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00")

            manager._render_with_blender = fake_render
            job = await manager.submit(b"BLENDER-test", {"engine": "cycles"})
            await job.task

            with patch("main.job_manager", manager):
                result = await download_render_result(job.job_id)

            resource = next(item for item in result.content if isinstance(item, EmbeddedResource))
            self.assertEqual(resource.resource.mime_type, "video/mp4")
            self.assertEqual(
                base64.b64decode(resource.resource.blob),
                b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00",
            )

    async def test_selected_frame_job_reuses_status_and_downloads_zip_resource(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RenderJobManager(Path(temp_dir), max_active_jobs=1)

            async def fake_render(job) -> None:
                with zipfile.ZipFile(job.output_path, "w") as archive:
                    for frame in job.options["frames"]:
                        archive.writestr(
                            f"frame_{frame:06d}.png",
                            b"\x89PNG\r\n\x1a\nframe-payload",
                        )

            manager._render_with_blender = fake_render
            with (
                patch("main.job_manager", manager),
                patch("main.decode_blend_base64", return_value=b"BLENDER-test"),
            ):
                submission = await submit_frame_render_job(
                    "unused", [8, 2], engine="cycles"
                )
                job_info = submission.structured_content
                self.assertIn(job_info["status"], ("queued", "running"))
                self.assertEqual(job_info["total_frames"], 2)
                self.assertEqual(job_info["output_format"], "png_zip")
                self.assertEqual(job_info["download_tool"], "download_render_frames")
                job = manager.get(job_info["job_id"])
                await job.task

                status = manager.status(job)
                self.assertEqual(status["status"], "completed")
                self.assertEqual(status["processed_frames"], 2)
                self.assertTrue(status["result_uri"].endswith("rendered_frames.zip"))

                downloaded = await download_render_frames(job.job_id)

            resource = next(item for item in downloaded.content if isinstance(item, EmbeddedResource))
            self.assertEqual(resource.resource.mime_type, "application/zip")
            archive_bytes = base64.b64decode(resource.resource.blob)
            with tempfile.TemporaryDirectory() as extracted_dir:
                archive_path = Path(extracted_dir) / "frames.zip"
                archive_path.write_bytes(archive_bytes)
                with zipfile.ZipFile(archive_path) as archive:
                    self.assertEqual(
                        archive.namelist(),
                        ["frame_000008.png", "frame_000002.png"],
                    )


class TailscaleTests(unittest.TestCase):
    def test_missing_or_empty_key_disables_tailscale(self) -> None:
        for value in (None, "", "   "):
            environment = {} if value is None else {"TAILSCALE_AUTH_KEY": value}
            with (
                self.subTest(value=value),
                patch.dict(os.environ, environment, clear=True),
                patch("main.subprocess.Popen") as popen,
            ):
                self.assertIsNone(start_tailscale())
                popen.assert_not_called()

    def test_key_starts_userspace_daemon(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir) / ".local/share/tailscale"
            commands = []
            key_paths = []

            def launch_daemon(command):
                state_dir.joinpath("tailscaled.sock").touch()
                return daemon

            def run_cli(command, **kwargs):
                commands.append(command)
                if "up" in command:
                    key_path = Path(next(arg for arg in command if arg.startswith("--auth-key=file:"))[16:])
                    key_paths.append(key_path)
                    self.assertEqual(key_path.read_text(), "test-auth-key")

            with (
                patch.dict(os.environ, {"TAILSCALE_AUTH_KEY": "test-auth-key"}, clear=True),
                patch("main.Path.home", return_value=Path(temp_dir)),
                patch("main.subprocess.Popen", side_effect=launch_daemon) as popen,
                patch("main.subprocess.run", side_effect=run_cli),
            ):
                daemon = unittest.mock.Mock()
                process = start_tailscale()

            self.assertIs(process, daemon)
            self.assertEqual(popen.call_args.args[0][0:2], ["tailscaled", "--tun=userspace-networking"])
            self.assertIn("--hostname=blender-render-mcp", commands[0])
            self.assertIn("--ssh=false", commands[0])
            self.assertEqual(len(commands), 1)
            self.assertNotIn("test-auth-key", str(commands))
            self.assertFalse(key_paths[0].exists())

    def test_ssh_can_be_enabled_for_headless_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_dir = Path(temp_dir) / ".local/share/tailscale"
            commands = []

            def launch_daemon(command):
                state_dir.joinpath("tailscaled.sock").touch()
                return unittest.mock.Mock()

            with (
                patch.dict(
                    os.environ,
                    {"TAILSCALE_AUTH_KEY": "test-auth-key", "TAILSCALE_SSH": "true"},
                    clear=True,
                ),
                patch("main.Path.home", return_value=Path(temp_dir)),
                patch("main.subprocess.Popen", side_effect=launch_daemon),
                patch("main.subprocess.run", side_effect=lambda command, **kwargs: commands.append(command)),
            ):
                start_tailscale()

            self.assertIn("--ssh=true", commands[0])

if __name__ == "__main__":
    unittest.main()
