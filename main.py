from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import hmac
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import BlobResourceContents, CallToolResult, EmbeddedResource, TextContent

MIB = 1024 * 1024
MAX_BLEND_BYTES = 100 * MIB
MAX_MP4_BYTES = 250 * MIB
MAX_FRAME_ARCHIVE_BYTES = 250 * MIB
MAX_FRAME_COUNT = 100
MAX_REQUEST_BODY_BYTES = 150 * MIB
MAX_RENDER_SECONDS = int(os.getenv("MAX_RENDER_SECONDS", "1800"))
MAX_ACTIVE_RENDER_JOBS = int(os.getenv("MAX_ACTIVE_RENDER_JOBS", "4"))
JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "86400"))
JOB_STORAGE_DIR = Path(
    os.getenv("JOB_STORAGE_DIR", str(Path(tempfile.gettempdir()) / "blender-mcp-jobs"))
)
MCP_PORT = int(os.getenv("MCP_PORT", "8080"))
BLENDER_EXECUTABLE = os.getenv(
    "BLENDER_EXECUTABLE", "/home/headless/blender/blender"
)
WORKER_SCRIPT = Path(__file__).with_name("blender_worker.py")
GPU_PROBE_SCRIPT = Path(__file__).with_name("blender_gpu_probe.py")
ENGINE_NAMES = ("cycles", "eevee", "workbench")

server = MCPServer(
    name="blender-rendering",
    version="1.0.0",
    instructions=(
        "Reports visible NVIDIA GPU capabilities and renders uploaded Blender projects "
        "as asynchronous jobs. Use submit_render_job for H.264 animation or "
        "submit_frame_render_job for selected PNG frames in a ZIP. Poll either with "
        "get_render_job_status, then use its matching download tool. Cycles uses OptiX "
        "when detected and otherwise uses CPU. Eevee and Workbench require a GPU "
        "visible to the container."
    ),
)


class RequestError(ValueError):
    """A user-correctable MCP tool request error."""


def decode_blend_base64(encoded: str, max_bytes: int = MAX_BLEND_BYTES) -> bytes:
    if not isinstance(encoded, str) or not encoded:
        raise RequestError("blend_file_base64 must be a non-empty base64 string")
    if not encoded.isascii():
        raise RequestError("blend_file_base64 must contain ASCII base64 data")

    maximum_encoded_size = 4 * ((max_bytes + 2) // 3)
    if len(encoded) > maximum_encoded_size:
        raise RequestError(f".blend input exceeds the {max_bytes}-byte limit")

    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestError("blend_file_base64 is not valid base64") from exc

    if len(decoded) > max_bytes:
        raise RequestError(f".blend input exceeds the {max_bytes}-byte limit")
    # Blender 5.2 saves compressed projects as a Zstandard frame containing
    # the usual BLENDER header, so accept either on-disk signature here.
    if not decoded.startswith((b"BLENDER", b"\x28\xb5\x2f\xfd")):
        raise RequestError("decoded input does not have a Blender .blend file header")
    return decoded


def parse_nvidia_smi_csv(output: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) < 5:
            continue
        name, uuid_value, memory_total, memory_free, driver = [part.strip() for part in row[:5]]
        if not name or name.upper() == "N/A":
            continue
        devices.append(
            {
                "name": name,
                "uuid": uuid_value,
                "memory_total_mib": memory_total,
                "memory_free_mib": memory_free,
                "driver_version": driver,
            }
        )
    return devices


def detect_nvidia_gpus() -> tuple[list[dict[str, str]], str | None]:
    if not shutil.which("nvidia-smi"):
        return [], "nvidia-smi is not available in this container"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"nvidia-smi failed: {exc}"
    return parse_nvidia_smi_csv(result.stdout), None


def probe_blender() -> dict[str, Any]:
    if not Path(BLENDER_EXECUTABLE).is_file() and not shutil.which(BLENDER_EXECUTABLE):
        return {"available": False, "error": "Blender executable was not found"}
    try:
        result = subprocess.run(
            [
                BLENDER_EXECUTABLE,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(GPU_PROBE_SCRIPT),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "Blender GPU probe timed out"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"Blender GPU probe failed: {exc}"}

    marker = "BLENDER_MCP_GPU_PROBE="
    for line in result.stdout.splitlines():
        if line.startswith(marker):
            try:
                payload = json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                break
            payload["available"] = True
            return payload
    return {
        "available": False,
        "error": "Blender GPU probe did not return a result",
        "log_tail": result.stderr[-2000:],
    }


def collect_gpu_info() -> dict[str, Any]:
    devices, nvidia_smi_error = detect_nvidia_gpus()
    blender = probe_blender()
    optix_available = bool(blender.get("optix_available"))
    # GPU inventory alone is not sufficient for Eevee/Workbench: Blender must
    # also be able to create a hardware graphics context (software GL is rejected).
    nvidia_gpu_visible = bool(devices) or optix_available
    gpu_available = nvidia_gpu_visible and bool(blender.get("graphics_available"))
    cycles_backend = "OPTIX" if optix_available else "CPU"
    return {
        "gpu_available": gpu_available,
        "gpu_devices": devices,
        "nvidia_smi_error": nvidia_smi_error,
        "blender": blender,
        "graphics_available": gpu_available,
        "render_engines": {
            "cycles": {"available": True, "backend": cycles_backend},
            "eevee": {
                "available": gpu_available,
                "backend": "GPU" if gpu_available else None,
            },
            "workbench": {
                "available": gpu_available,
                "backend": "GPU" if gpu_available else None,
            },
        },
    }


def validate_render_request(
    engine: str | None,
    gpu_available: bool | None,
    frame_start: int | None,
    frame_end: int | None,
    resolution_x: int | None,
    resolution_y: int | None,
    resolution_percentage: int | None,
    samples: int | None,
) -> None:
    if engine is not None and engine not in ENGINE_NAMES:
        raise RequestError(f"engine must be one of: {', '.join(ENGINE_NAMES)}")
    if engine in ("eevee", "workbench") and gpu_available is False:
        raise RequestError(f"{engine} rendering requires a GPU visible to the container")
    if frame_start is not None and frame_end is not None and frame_end < frame_start:
        raise RequestError("frame_end must be greater than or equal to frame_start")
    for name, value in (("resolution_x", resolution_x), ("resolution_y", resolution_y)):
        if value is not None and not 1 <= value <= 16384:
            raise RequestError(f"{name} must be between 1 and 16384")
    if resolution_percentage is not None and not 1 <= resolution_percentage <= 100:
        raise RequestError("resolution_percentage must be between 1 and 100")
    if samples is not None and not 1 <= samples <= 65536:
        raise RequestError("samples must be between 1 and 65536")


def validate_mp4_size(size_bytes: int, max_bytes: int = MAX_MP4_BYTES) -> None:
    if size_bytes > max_bytes:
        raise RequestError(f"MP4 output exceeds the {max_bytes}-byte limit")


def validate_frame_list(frames: list[int]) -> None:
    if not isinstance(frames, list) or not frames:
        raise RequestError("frames must be a non-empty list of integer frame numbers")
    if len(frames) > MAX_FRAME_COUNT:
        raise RequestError(f"frames may contain at most {MAX_FRAME_COUNT} frame numbers")
    if any(type(frame) is not int for frame in frames):
        raise RequestError("frames must contain only integer frame numbers")
    if len(set(frames)) != len(frames):
        raise RequestError("frames must not contain duplicate frame numbers")


def validate_frame_archive(
    path: Path,
    frames: list[int],
    max_bytes: int = MAX_FRAME_ARCHIVE_BYTES,
) -> int:
    try:
        archive_size = path.stat().st_size
    except OSError as exc:
        raise RequestError(f"could not read frame archive: {exc}") from exc
    if archive_size > max_bytes:
        raise RequestError(f"frame archive exceeds the {max_bytes}-byte limit")
    if not zipfile.is_zipfile(path):
        raise RequestError("render output is not a valid ZIP archive")

    expected_names = [f"frame_{frame:06d}.png" for frame in frames]
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if [entry.filename for entry in entries] != expected_names:
                raise RequestError("frame archive does not contain the requested PNG files")
            for entry in entries:
                if entry.is_dir() or entry.file_size < 8:
                    raise RequestError(f"frame archive contains an invalid PNG: {entry.filename}")
                with archive.open(entry) as png_file:
                    if png_file.read(8) != b"\x89PNG\r\n\x1a\n":
                        raise RequestError(f"frame archive contains an invalid PNG: {entry.filename}")
    except zipfile.BadZipFile as exc:
        raise RequestError("render output is not a valid ZIP archive") from exc
    return archive_size


def _tool_error(message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


@server.tool(structured_output=False)
async def get_gpu_info() -> CallToolResult:
    """Return NVIDIA GPU inventory and Blender rendering backend availability."""
    info = await asyncio.to_thread(collect_gpu_info)
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(info, indent=2))],
        structuredContent=info,
    )


@dataclass
class RenderJob:
    job_id: str
    directory: Path
    options: dict[str, Any]
    submitted_at: float = field(default_factory=time.monotonic)
    status: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    processed_frames: int = 0
    total_frames: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    task: asyncio.Task[None] | None = None

    @property
    def blend_path(self) -> Path:
        return self.directory / "input.blend"

    @property
    def config_path(self) -> Path:
        return self.directory / "render.json"

    @property
    def progress_path(self) -> Path:
        return self.directory / "progress.json"

    @property
    def output_format(self) -> str:
        return self.options.get("output_format", "mp4")

    @property
    def output_filename(self) -> str:
        return "rendered_frames.zip" if self.output_format == "png_zip" else "render.mp4"

    @property
    def output_path(self) -> Path:
        return self.directory / self.output_filename

    @property
    def log_path(self) -> Path:
        return self.directory / "blender.log"


class RenderJobManager:
    """Keeps render jobs and their files for status polling and later download."""

    def __init__(
        self,
        storage_root: Path = JOB_STORAGE_DIR,
        max_active_jobs: int = MAX_ACTIVE_RENDER_JOBS,
        retention_seconds: int = JOB_RETENTION_SECONDS,
    ) -> None:
        self.storage_root = storage_root
        self.max_active_jobs = max_active_jobs
        self.retention_seconds = retention_seconds
        self.jobs: dict[str, RenderJob] = {}
        self._submission_lock = asyncio.Lock()
        self._render_lock = asyncio.Semaphore(1)

    async def submit(self, blend_bytes: bytes, options: dict[str, Any]) -> RenderJob:
        async with self._submission_lock:
            self._prune_expired()
            active_jobs = sum(job.status in ("queued", "running") for job in self.jobs.values())
            if active_jobs >= self.max_active_jobs:
                raise RequestError(
                    f"render queue is full ({self.max_active_jobs} active jobs); try again later"
                )

            await asyncio.to_thread(self.storage_root.mkdir, parents=True, exist_ok=True)
            job_id = uuid.uuid4().hex
            directory = self.storage_root / job_id
            await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=False)
            job = RenderJob(job_id=job_id, directory=directory, options=options)
            frames = options.get("frames")
            if frames is not None:
                job.total_frames = len(frames)
            else:
                frame_start = options.get("frame_start")
                frame_end = options.get("frame_end")
                if frame_start is not None and frame_end is not None:
                    job.total_frames = frame_end - frame_start + 1

            try:
                await asyncio.to_thread(job.blend_path.write_bytes, blend_bytes)
            except Exception:
                await asyncio.to_thread(shutil.rmtree, directory, True)
                raise

            self.jobs[job_id] = job
            job.task = asyncio.create_task(self._run_job(job))
            return job

    def get(self, job_id: str) -> RenderJob:
        self._prune_expired()
        job = self.jobs.get(job_id)
        if job is None:
            raise RequestError("render job was not found or has expired")
        return job

    def status(self, job: RenderJob) -> dict[str, Any]:
        self._read_progress(job)
        if job.status == "completed" and job.total_frames is not None:
            job.processed_frames = job.total_frames
        now = time.monotonic()
        end_time = job.finished_at if job.finished_at is not None else now
        elapsed = max(0.0, end_time - job.submitted_at)
        running = (
            max(0.0, end_time - job.started_at)
            if job.started_at is not None
            else 0.0
        )
        queue_time = (
            max(0.0, job.started_at - job.submitted_at)
            if job.started_at is not None
            else elapsed
        )
        progress = (
            min(100.0, 100.0 * job.processed_frames / job.total_frames)
            if job.total_frames
            else None
        )
        return {
            "job_id": job.job_id,
            "status": job.status,
            "requested_engine": job.options.get("engine"),
            "requested_frames": job.options.get("frames"),
            "output_format": job.output_format,
            "running_time_seconds": round(running, 3),
            "queue_time_seconds": round(queue_time, 3),
            "elapsed_time_seconds": round(elapsed, 3),
            "processed_frames": job.processed_frames,
            "total_frames": job.total_frames,
            "progress_percent": round(progress, 2) if progress is not None else None,
            "result_available": job.status == "completed",
            "error": job.error,
            "render": job.metadata or None,
            "result_uri": (
                f"render://jobs/{job.job_id}/{job.output_filename}"
                if job.status == "completed"
                else None
            ),
        }

    def _read_progress(self, job: RenderJob) -> None:
        try:
            progress = json.loads(job.progress_path.read_text(encoding="utf-8"))
            processed = int(progress.get("processed_frames", job.processed_frames))
            total = progress.get("total_frames")
            job.processed_frames = max(0, processed)
            if total is not None:
                job.total_frames = max(0, int(total))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    def _prune_expired(self) -> None:
        now = time.monotonic()
        expired = [
            job_id
            for job_id, job in self.jobs.items()
            if job.finished_at is not None
            and now - job.finished_at >= self.retention_seconds
        ]
        for job_id in expired:
            job = self.jobs.pop(job_id)
            shutil.rmtree(job.directory, ignore_errors=True)

    async def _run_job(self, job: RenderJob) -> None:
        try:
            async with self._render_lock:
                job.status = "running"
                job.started_at = time.monotonic()
                await self._render_with_blender(job)
                self._read_progress(job)
                if not job.output_path.is_file():
                    raise RequestError(
                        f"Blender completed without producing {job.output_filename}"
                    )
                if job.output_format == "mp4":
                    output_size = job.output_path.stat().st_size
                    validate_mp4_size(output_size)
                    with job.output_path.open("rb") as output_file:
                        header = output_file.read(8)
                    if output_size < 12 or header[4:8] != b"ftyp":
                        raise RequestError("Blender output is not a valid MP4 file")
                elif job.output_format == "png_zip":
                    output_size = validate_frame_archive(
                        job.output_path,
                        job.options["frames"],
                    )
                else:
                    raise RequestError(f"unsupported output format {job.output_format!r}")
                if job.total_frames is not None:
                    job.processed_frames = job.total_frames
                job.metadata["size_bytes"] = output_size
                job.status = "completed"
        except asyncio.CancelledError:
            job.error = "render job was interrupted because the server is shutting down"
            job.status = "failed"
            raise
        except Exception as exc:
            job.error = str(exc)[:4000] or type(exc).__name__
            job.status = "failed"
            print(f"Render job {job.job_id} failed: {job.error}", flush=True)
        finally:
            if job.status in ("completed", "failed"):
                job.finished_at = time.monotonic()

    async def _render_with_blender(self, job: RenderJob) -> None:
        gpu_info = await asyncio.to_thread(collect_gpu_info)
        options = job.options
        validate_render_request(
            options.get("engine"),
            bool(gpu_info["gpu_available"]),
            options.get("frame_start"),
            options.get("frame_end"),
            options.get("resolution_x"),
            options.get("resolution_y"),
            options.get("resolution_percentage"),
            options.get("samples"),
        )
        job.config_path.write_text(
            json.dumps(
                {
                    **options,
                    "output_format": job.output_format,
                    "gpu_available": bool(gpu_info["gpu_available"]),
                    "optix_available": bool(
                        gpu_info["render_engines"]["cycles"]["backend"] == "OPTIX"
                    ),
                    "output_file": str(job.output_path),
                    "max_archive_bytes": MAX_FRAME_ARCHIVE_BYTES,
                    "progress_file": str(job.progress_path),
                }
            ),
            encoding="utf-8",
        )
        command = [
            BLENDER_EXECUTABLE,
            "--background",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            str(job.blend_path),
            "--python",
            str(WORKER_SCRIPT),
            "--",
            str(job.config_path),
        ]
        process: asyncio.subprocess.Process | None = None
        try:
            with job.log_path.open("wb") as log_file:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=log_file,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await asyncio.wait_for(process.wait(), timeout=MAX_RENDER_SECONDS)
        except asyncio.TimeoutError as exc:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            raise RequestError(
                f"render exceeded the {MAX_RENDER_SECONDS}-second limit"
            ) from exc
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except OSError as exc:
            raise RequestError(f"could not start Blender: {exc}") from exc

        assert process is not None
        self._read_progress(job)
        if process.returncode != 0:
            log_tail = job.log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RequestError(
                "Blender render failed"
                + (f":\n{log_tail}" if log_tail else f" (exit code {process.returncode})")
            )
        marker = "BLENDER_MCP_RENDER="
        for line in job.log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(marker):
                try:
                    job.metadata.update(json.loads(line[len(marker) :]))
                except json.JSONDecodeError:
                    pass
                break


job_manager = RenderJobManager()


def _job_status_result(job: RenderJob) -> CallToolResult:
    status = job_manager.status(job)
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(status, indent=2))],
        structuredContent=status,
    )


@server.tool(structured_output=False)
async def submit_render_job(
    blend_file_base64: str,
    engine: Literal["cycles", "eevee", "workbench"] | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    resolution_x: int | None = None,
    resolution_y: int | None = None,
    resolution_percentage: int | None = None,
    samples: int | None = None,
) -> CallToolResult:
    """Submit a .blend animation for rendering and immediately return its job ID."""
    try:
        validate_render_request(
            engine,
            None,
            frame_start,
            frame_end,
            resolution_x,
            resolution_y,
            resolution_percentage,
            samples,
        )
        blend_bytes = await asyncio.to_thread(decode_blend_base64, blend_file_base64)
        job = await job_manager.submit(
            blend_bytes,
            {
                "engine": engine,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "resolution_x": resolution_x,
                "resolution_y": resolution_y,
                "resolution_percentage": resolution_percentage,
                "samples": samples,
            },
        )
        result = job_manager.status(job)
        result["status_tool"] = "get_render_job_status"
        result["download_tool"] = "download_render_result"
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            structuredContent=result,
        )
    except (RequestError, OSError, json.JSONDecodeError) as exc:
        return _tool_error(str(exc))
    except Exception as exc:
        return _tool_error(f"could not submit render job: {exc}")


@server.tool(structured_output=False)
async def submit_frame_render_job(
    blend_file_base64: str,
    frames: list[int],
    engine: Literal["cycles", "eevee", "workbench"] | None = None,
    resolution_x: int | None = None,
    resolution_y: int | None = None,
    resolution_percentage: int | None = None,
    samples: int | None = None,
) -> CallToolResult:
    """Render 1–100 unique frames in the saved scene range to PNGs in an async ZIP job."""
    try:
        validate_frame_list(frames)
        validate_render_request(
            engine,
            None,
            None,
            None,
            resolution_x,
            resolution_y,
            resolution_percentage,
            samples,
        )
        blend_bytes = await asyncio.to_thread(decode_blend_base64, blend_file_base64)
        job = await job_manager.submit(
            blend_bytes,
            {
                "engine": engine,
                "frames": list(frames),
                "output_format": "png_zip",
                "resolution_x": resolution_x,
                "resolution_y": resolution_y,
                "resolution_percentage": resolution_percentage,
                "samples": samples,
            },
        )
        result = job_manager.status(job)
        result["status_tool"] = "get_render_job_status"
        result["download_tool"] = "download_render_frames"
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))],
            structuredContent=result,
        )
    except (RequestError, OSError, json.JSONDecodeError) as exc:
        return _tool_error(str(exc))
    except Exception as exc:
        return _tool_error(f"could not submit frame render job: {exc}")


@server.tool(structured_output=False)
async def get_render_job_status(job_id: str) -> CallToolResult:
    """Return render state, elapsed time, and processed/total frame counts."""
    try:
        return _job_status_result(job_manager.get(job_id))
    except RequestError as exc:
        return _tool_error(str(exc))


@server.tool(structured_output=False)
async def download_render_result(job_id: str) -> CallToolResult:
    """Download the embedded H.264 MP4 for a completed render job."""
    try:
        job = job_manager.get(job_id)
        if job.output_format != "mp4":
            raise RequestError("this job contains PNG frames; use download_render_frames")
        if job.status != "completed":
            raise RequestError(
                f"render job is {job.status}; check get_render_job_status and retry after completion"
            )
        mp4_bytes = await asyncio.to_thread(job.output_path.read_bytes)
        validate_mp4_size(len(mp4_bytes))
        if len(mp4_bytes) < 12 or mp4_bytes[4:8] != b"ftyp":
            raise RequestError("stored render result is not a valid MP4 file")
        result_info = {
            "job_id": job.job_id,
            "filename": "render.mp4",
            "mime_type": "video/mp4",
            "size_bytes": len(mp4_bytes),
            **job.metadata,
        }
        return CallToolResult(
            content=[
                TextContent(type="text", text=f"Rendered MP4 is {len(mp4_bytes)} bytes."),
                EmbeddedResource(
                    resource=BlobResourceContents(
                        uri=f"render://jobs/{job.job_id}/render.mp4",
                        mimeType="video/mp4",
                        blob=base64.b64encode(mp4_bytes).decode("ascii"),
                    )
                ),
            ],
            structuredContent=result_info,
        )
    except (RequestError, OSError) as exc:
        return _tool_error(str(exc))
    except Exception as exc:
        return _tool_error(f"could not download render result: {exc}")


@server.tool(structured_output=False)
async def download_render_frames(job_id: str) -> CallToolResult:
    """Download the completed selected-frame PNG archive as an application/zip resource."""
    try:
        job = job_manager.get(job_id)
        if job.output_format != "png_zip":
            raise RequestError("this job contains an MP4; use download_render_result")
        if job.status != "completed":
            raise RequestError(
                f"render job is {job.status}; check get_render_job_status and retry after completion"
            )
        zip_bytes = await asyncio.to_thread(job.output_path.read_bytes)
        validate_frame_archive(job.output_path, job.options["frames"])
        result_info = {
            **job.metadata,
            "job_id": job.job_id,
            "filename": job.output_filename,
            "mime_type": "application/zip",
            "size_bytes": len(zip_bytes),
            "frames": job.options["frames"],
            "frame_count": len(job.options["frames"]),
        }
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Rendered {len(job.options['frames'])} PNG frames in a {len(zip_bytes)}-byte ZIP archive.",
                ),
                EmbeddedResource(
                    resource=BlobResourceContents(
                        uri=f"render://jobs/{job.job_id}/{job.output_filename}",
                        mimeType="application/zip",
                        blob=base64.b64encode(zip_bytes).decode("ascii"),
                    )
                ),
            ],
            structuredContent=result_info,
        )
    except (RequestError, OSError) as exc:
        return _tool_error(str(exc))
    except Exception as exc:
        return _tool_error(f"could not download rendered frames: {exc}")


class BearerAuthApp:
    """ASGI wrapper for optional static bearer-token authentication."""

    def __init__(self, app: Any, token: str | None) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode("utf-8") if token is not None else None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if self.expected is not None and scope.get("type") == "http":
            authorization = next(
                (
                    value
                    for name, value in scope.get("headers", [])
                    if name.lower() == b"authorization"
                ),
                b"",
            )
            if not hmac.compare_digest(authorization, self.expected):
                body = b'{"error":"unauthorized"}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode("ascii")),
                            (b"www-authenticate", b"Bearer"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def create_app(token: str | None = None) -> BearerAuthApp:
    if token is None:
        token = os.getenv("MCP_BEARER_TOKEN")
    if token is not None and not token.strip():
        raise ValueError("MCP_BEARER_TOKEN is set but empty")

    mcp_app = server.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        max_request_body_size=MAX_REQUEST_BODY_BYTES,
        host="0.0.0.0",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    return BearerAuthApp(mcp_app, token)


def cloudflared_command(token: str | None = None) -> list[str] | None:
    if token is None:
        token = os.getenv("CLOUDFLARE_TUNNEL_TOKEN")
    if token is None or not token.strip():
        return None
    return ["cloudflared", "tunnel", "run", "--token", token]


def start_cloudflare_tunnel() -> subprocess.Popen[bytes] | None:
    command = cloudflared_command()
    if command is None:
        print(
            "CLOUDFLARE_TUNNEL_TOKEN is unset; starting without a Cloudflare Tunnel.",
            flush=True,
        )
        return None
    print("Starting Cloudflare Tunnel connector.", flush=True)
    return subprocess.Popen(command)


def stop_cloudflare_tunnel(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> None:
    import uvicorn

    token = os.getenv("MCP_BEARER_TOKEN")
    app = create_app(token)
    tunnel_process = start_cloudflare_tunnel()
    try:
        uvicorn.run(app, host="0.0.0.0", port=MCP_PORT, timeout_keep_alive=120)
    finally:
        stop_cloudflare_tunnel(tunnel_process)


if __name__ == "__main__":
    main()
