"""Exercise the running HTTP server; optionally render a supplied .blend file."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
import zipfile
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.types import EmbeddedResource


async def run(
    endpoint: str,
    blend_path: Path | None,
    token: str | None,
    check_cpu_denials: bool,
    check_gpu_engines: bool,
    frame_number: int | None,
    timeout_seconds: int,
    poll_interval: float,
) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    http_client = create_mcp_http_client(headers=headers)
    async with streamable_http_client(endpoint, http_client=http_client) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            assert {
                "get_gpu_info",
                "submit_render_job",
                "submit_frame_render_job",
                "get_render_job_status",
                "download_render_result",
                "download_render_frames",
            } <= tool_names, tool_names
            gpu_result = await session.call_tool("get_gpu_info")
            print("GPU info:", gpu_result.content[0].text)
            gpu_info = json.loads(gpu_result.content[0].text)
            if check_gpu_engines and not gpu_info["gpu_available"]:
                raise RuntimeError("GPU engine checks require a GPU visible to the Docker container")
            if blend_path is None:
                return

            blend_bytes = blend_path.read_bytes()
            encoded = base64.b64encode(blend_bytes).decode("ascii")

            async def submit_and_wait(engine: str | None):
                arguments = {"blend_file_base64": encoded}
                if engine is not None:
                    arguments["engine"] = engine
                submission = await session.call_tool("submit_render_job", arguments)
                if submission.is_error:
                    raise RuntimeError(
                        "render submission failed: "
                        + "\n".join(item.text for item in submission.content if hasattr(item, "text"))
                    )
                job_info = submission.structured_content or json.loads(submission.content[0].text)
                job_id = job_info["job_id"]
                deadline = time.monotonic() + timeout_seconds
                while time.monotonic() < deadline:
                    status_result = await session.call_tool(
                        "get_render_job_status", {"job_id": job_id}
                    )
                    if status_result.is_error:
                        raise RuntimeError(
                            "status query failed: "
                            + "\n".join(item.text for item in status_result.content if hasattr(item, "text"))
                        )
                    status = status_result.structured_content or json.loads(
                        status_result.content[0].text
                    )
                    print(
                        f"{engine or 'saved engine'} job {job_id}: {status['status']}, "
                        f"{status['processed_frames']}/{status['total_frames']} frames, "
                        f"running {status['running_time_seconds']:.1f}s",
                        flush=True,
                    )
                    if status["status"] == "failed":
                        return job_id, status, None
                    if status["status"] == "completed":
                        result = await session.call_tool(
                            "download_render_result", {"job_id": job_id}
                        )
                        if result.is_error:
                            raise RuntimeError(
                                "render download failed: "
                                + "\n".join(item.text for item in result.content if hasattr(item, "text"))
                            )
                        return job_id, status, result
                    await asyncio.sleep(poll_interval)
                raise TimeoutError(f"render job {job_id} did not finish within {timeout_seconds}s")

            if check_cpu_denials:
                if gpu_info["gpu_available"]:
                    raise RuntimeError("CPU denial checks require a CPU-only server")
                for engine in ("eevee", "workbench"):
                    _, denied_status, _ = await submit_and_wait(engine)
                    if denied_status["status"] != "failed" or "requires a GPU" not in (
                        denied_status.get("error") or ""
                    ):
                        raise RuntimeError(
                            f"{engine} did not fail with the expected GPU requirement: {denied_status}"
                        )
                    print(f"{engine} correctly rejected without GPU: {denied_status['error']}")

            async def render(engine: str | None, output_path: Path) -> None:
                job_id, status, result = await submit_and_wait(engine)
                if status["status"] != "completed" or result is None:
                    raise RuntimeError(
                        f"{engine or 'saved engine'} render failed: {status.get('error')}"
                    )
                resource = next(
                    item for item in result.content if isinstance(item, EmbeddedResource)
                )
                if resource.resource.mime_type != "video/mp4":
                    raise RuntimeError(
                        f"unexpected resource MIME type: {resource.resource.mime_type}"
                    )
                mp4 = base64.b64decode(resource.resource.blob, validate=True)
                if len(mp4) < 12 or mp4[4:8] != b"ftyp":
                    raise RuntimeError("server returned an invalid MP4 resource")
                output_path.write_bytes(mp4)
                metadata = result.structured_content or {}
                if engine is not None and metadata.get("engine") != engine:
                    raise RuntimeError(f"render metadata did not report engine {engine!r}")
                if engine in (None, "cycles") and not gpu_info["gpu_available"]:
                    if metadata.get("backend") != "CPU":
                        raise RuntimeError("CPU-only Cycles render did not report the CPU backend")
                print(
                    f"{engine or 'saved engine'} MP4 verified ({len(mp4)} bytes): {output_path} "
                    f"(job {job_id})"
                )

            await render(None, blend_path.with_suffix(".render.mp4"))
            if check_gpu_engines:
                await render("eevee", blend_path.with_suffix(".eevee.mp4"))
                await render("workbench", blend_path.with_suffix(".workbench.mp4"))

            if frame_number is not None:
                submission = await session.call_tool(
                    "submit_frame_render_job",
                    {"blend_file_base64": encoded, "frames": [frame_number]},
                )
                if submission.is_error:
                    raise RuntimeError(
                        "frame render submission failed: "
                        + "\n".join(
                            item.text
                            for item in submission.content
                            if hasattr(item, "text")
                        )
                    )
                job_info = submission.structured_content or json.loads(
                    submission.content[0].text
                )
                job_id = job_info["job_id"]
                deadline = time.monotonic() + timeout_seconds
                while time.monotonic() < deadline:
                    status_result = await session.call_tool(
                        "get_render_job_status", {"job_id": job_id}
                    )
                    if status_result.is_error:
                        raise RuntimeError(
                            "frame status query failed: "
                            + "\n".join(
                                item.text
                                for item in status_result.content
                                if hasattr(item, "text")
                            )
                        )
                    status = status_result.structured_content or json.loads(
                        status_result.content[0].text
                    )
                    print(
                        f"frame {frame_number} job {job_id}: {status['status']}, "
                        f"{status['processed_frames']}/{status['total_frames']} frames, "
                        f"running {status['running_time_seconds']:.1f}s",
                        flush=True,
                    )
                    if status["status"] == "failed":
                        raise RuntimeError(
                            f"frame render failed: {status.get('error')}"
                        )
                    if status["status"] == "completed":
                        result = await session.call_tool(
                            "download_render_frames", {"job_id": job_id}
                        )
                        if result.is_error:
                            raise RuntimeError(
                                "frame archive download failed: "
                                + "\n".join(
                                    item.text
                                    for item in result.content
                                    if hasattr(item, "text")
                                )
                            )
                        resource = next(
                            item
                            for item in result.content
                            if isinstance(item, EmbeddedResource)
                        )
                        if resource.resource.mime_type != "application/zip":
                            raise RuntimeError(
                                "unexpected frame result MIME type: "
                                f"{resource.resource.mime_type}"
                            )
                        archive_bytes = base64.b64decode(
                            resource.resource.blob, validate=True
                        )
                        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                            expected_name = f"frame_{frame_number:06d}.png"
                            if archive.namelist() != [expected_name]:
                                raise RuntimeError(
                                    f"unexpected frame archive contents: {archive.namelist()}"
                                )
                            if archive.read(expected_name)[:8] != b"\x89PNG\r\n\x1a\n":
                                raise RuntimeError("frame archive contains an invalid PNG")
                        print(
                            f"PNG frame archive verified ({len(archive_bytes)} bytes; "
                            f"frame {frame_number}, job {job_id})",
                            flush=True,
                        )
                        break
                    await asyncio.sleep(poll_interval)
                else:
                    raise TimeoutError(
                        f"frame render job {job_id} did not finish within {timeout_seconds}s"
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8080/mcp")
    parser.add_argument("--blend", type=Path)
    parser.add_argument("--token", default=None)
    parser.add_argument("--check-cpu-denials", action="store_true")
    parser.add_argument("--check-gpu-engines", action="store_true")
    parser.add_argument(
        "--frame",
        type=int,
        help="also render this scene frame and verify the downloaded PNG ZIP",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="maximum wait per render job")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()
    asyncio.run(
        run(
            args.endpoint,
            args.blend,
            args.token,
            args.check_cpu_denials,
            args.check_gpu_engines,
            args.frame,
            args.timeout,
            args.poll_interval,
        )
    )


if __name__ == "__main__":
    main()
