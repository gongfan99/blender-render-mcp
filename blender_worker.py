"""Render one uploaded Blender project to an H.264 MP4 file."""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import bpy


ENGINE_IDS = {
    "cycles": "CYCLES",
    "eevee": "BLENDER_EEVEE",
    "workbench": "BLENDER_WORKBENCH",
}
ENGINE_NAMES = {
    "CYCLES": "cycles",
    "BLENDER_EEVEE": "eevee",
    "BLENDER_EEVEE_NEXT": "eevee",
    "BLENDER_WORKBENCH": "workbench",
}


def read_config() -> dict[str, Any]:
    try:
        separator = sys.argv.index("--")
        config_path = Path(sys.argv[separator + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("render config path missing after --") from exc
    return json.loads(config_path.read_text(encoding="utf-8"))


def configure_cycles(scene: bpy.types.Scene, config: dict[str, Any]) -> str:
    optix_enabled = False
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        if config.get("optix_available"):
            preferences.compute_device_type = "OPTIX"
            preferences.get_devices()
            devices = list(preferences.devices)
            optix_devices = [device for device in devices if device.type == "OPTIX"]
            if optix_devices:
                for device in devices:
                    device.use = device.type == "OPTIX"
                scene.cycles.device = "GPU"
                optix_enabled = True
    except Exception:
        optix_enabled = False

    if not optix_enabled:
        scene.cycles.device = "CPU"
    return "OPTIX" if optix_enabled else "CPU"


def apply_settings(scene: bpy.types.Scene, config: dict[str, Any]) -> tuple[str, str]:
    requested_engine = config.get("engine")
    if requested_engine is None:
        engine_name = ENGINE_NAMES.get(scene.render.engine)
        if engine_name is None:
            raise RuntimeError(
                f"The project uses unsupported render engine {scene.render.engine!r}"
            )
    else:
        engine_name = requested_engine

    if engine_name not in ENGINE_IDS:
        raise RuntimeError(f"unsupported render engine {engine_name!r}")
    if engine_name in ("eevee", "workbench") and not config.get("gpu_available"):
        raise RuntimeError(f"{engine_name} rendering requires a GPU visible to the container")

    scene.render.engine = ENGINE_IDS[engine_name]
    backend = "GPU"
    if engine_name == "cycles":
        backend = configure_cycles(scene, config)

    if config.get("frame_start") is not None:
        scene.frame_start = int(config["frame_start"])
    if config.get("frame_end") is not None:
        scene.frame_end = int(config["frame_end"])
    if scene.frame_end < scene.frame_start:
        raise RuntimeError("frame_end must be greater than or equal to frame_start")

    if config.get("resolution_x") is not None:
        scene.render.resolution_x = int(config["resolution_x"])
    if config.get("resolution_y") is not None:
        scene.render.resolution_y = int(config["resolution_y"])
    if config.get("resolution_percentage") is not None:
        scene.render.resolution_percentage = int(config["resolution_percentage"])

    samples = config.get("samples")
    if samples is not None:
        if engine_name == "cycles":
            scene.cycles.samples = int(samples)
        elif engine_name == "eevee":
            scene.eevee.taa_render_samples = int(samples)
        else:
            property_def = scene.display.bl_rna.properties.get("render_aa")
            valid = [item.identifier for item in property_def.enum_items] if property_def else []
            numeric = [int(item) for item in valid if item.isdigit()]
            if not numeric:
                raise RuntimeError("Workbench sample override is unavailable in this Blender build")
            selected = min(numeric, key=lambda value: abs(value - int(samples)))
            scene.display.render_aa = str(selected)

    output_format = config.get("output_format", "mp4")
    if output_format == "mp4":
        scene.render.image_settings.media_type = "VIDEO"
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.filepath = str(Path(config["output_file"]).with_suffix(""))
    elif output_format == "png_zip":
        scene.render.image_settings.media_type = "IMAGE"
        scene.render.image_settings.file_format = "PNG"
    else:
        raise RuntimeError(f"unsupported output format {output_format!r}")
    scene.render.use_file_extension = True
    return engine_name, backend


def write_progress(
    progress_path: Path | None,
    processed_frames: int,
    total_frames: int,
    current_frame: int | None = None,
) -> None:
    if progress_path is None:
        return
    payload = {
        "processed_frames": processed_frames,
        "total_frames": total_frames,
        "current_frame": current_frame,
    }
    temporary_path = progress_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload), encoding="utf-8")
    temporary_path.replace(progress_path)


def render_png_frames(
    scene: bpy.types.Scene,
    frames: list[int],
    output_path: Path,
    max_archive_bytes: int,
) -> None:
    if not frames:
        raise RuntimeError("at least one frame is required for a PNG archive")
    if any(type(frame) is not int for frame in frames):
        raise RuntimeError("requested frames must be integers")
    if len(set(frames)) != len(frames):
        raise RuntimeError("requested frames must be unique")
    out_of_range = [
        frame for frame in frames if frame < scene.frame_start or frame > scene.frame_end
    ]
    if out_of_range:
        raise RuntimeError(
            f"requested frames must be within the saved scene range "
            f"{scene.frame_start}-{scene.frame_end}: {out_of_range}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_png_bytes = 0
    with tempfile.TemporaryDirectory(
        prefix="rendered-png-frames-", dir=str(output_path.parent)
    ) as temporary_directory:
        frame_directory = Path(temporary_directory)
        rendered_files: list[Path] = []
        for frame in frames:
            scene.frame_set(frame)
            frame_path = frame_directory / f"frame_{frame:06d}.png"
            scene.render.filepath = str(frame_path.with_suffix(""))
            bpy.ops.render.render(write_still=True)
            if not frame_path.is_file():
                raise RuntimeError(f"Blender did not produce PNG for frame {frame}")
            with frame_path.open("rb") as png_file:
                if png_file.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise RuntimeError(f"Blender produced an invalid PNG for frame {frame}")
            total_png_bytes += frame_path.stat().st_size
            if total_png_bytes > max_archive_bytes:
                raise RuntimeError(
                    f"PNG frame data exceeds the {max_archive_bytes}-byte archive limit"
                )
            rendered_files.append(frame_path)

        with zipfile.ZipFile(
            output_path, mode="w", compression=zipfile.ZIP_STORED
        ) as archive:
            for frame_path in rendered_files:
                archive.write(frame_path, arcname=frame_path.name)
        if output_path.stat().st_size > max_archive_bytes:
            raise RuntimeError(
                f"frame archive exceeds the {max_archive_bytes}-byte archive limit"
            )


def main() -> None:
    config = read_config()
    output_format = config.get("output_format", "mp4")
    if output_format not in ("mp4", "png_zip"):
        raise RuntimeError(f"unsupported output format {output_format!r}")
    if output_format == "mp4" and not getattr(bpy.app.build_options, "codec_ffmpeg", False):
        raise RuntimeError("this Blender build does not include FFmpeg support")
    scene = bpy.context.scene
    if scene is None:
        raise RuntimeError("the .blend file does not contain an active scene")

    engine, backend = apply_settings(scene, config)
    frames = config.get("frames") if output_format == "png_zip" else None
    if output_format == "png_zip" and not isinstance(frames, list):
        raise RuntimeError("frames must be a list for PNG ZIP output")
    progress_path = (
        Path(config["progress_file"]) if config.get("progress_file") else None
    )
    total_frames = (
        len(frames)
        if frames is not None
        else scene.frame_end - scene.frame_start + 1
    )
    processed_frames = 0
    write_progress(progress_path, processed_frames, total_frames)

    def record_rendered_frame(rendered_scene: bpy.types.Scene) -> None:
        nonlocal processed_frames
        processed_frames = min(processed_frames + 1, total_frames)
        write_progress(
            progress_path,
            processed_frames,
            total_frames,
            rendered_scene.frame_current,
        )

    bpy.app.handlers.render_post.append(record_rendered_frame)
    try:
        if frames is None:
            bpy.ops.render.render(animation=True)
        else:
            render_png_frames(
                scene,
                frames,
                Path(config["output_file"]),
                int(config.get("max_archive_bytes", 250 * 1024 * 1024)),
            )
    finally:
        if record_rendered_frame in bpy.app.handlers.render_post:
            bpy.app.handlers.render_post.remove(record_rendered_frame)
        write_progress(
            progress_path,
            processed_frames,
            total_frames,
            scene.frame_current,
        )
    output_path = Path(config["output_file"])
    if output_format == "mp4" and not output_path.is_file():
        # Blender appends the animation frame range to movie filenames. The
        # render directory is job-private, so normalize its single MP4 output.
        candidates = list(output_path.parent.glob(f"{output_path.stem}*.mp4"))
        if len(candidates) == 1:
            candidates[0].replace(output_path)
    if not output_path.is_file():
        raise RuntimeError(f"render did not create expected output file: {output_path}")
    print(
        "BLENDER_MCP_RENDER="
        + json.dumps(
            {
                "engine": engine,
                "backend": backend,
                "output_format": output_format,
                "frame_start": scene.frame_start,
                "frame_end": scene.frame_end,
                "rendered_frames": frames,
                "resolution_x": scene.render.resolution_x,
                "resolution_y": scene.render.resolution_y,
                "resolution_percentage": scene.render.resolution_percentage,
                "fps": scene.render.fps,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"BLENDER_MCP_ERROR={exc}", file=sys.stderr, flush=True)
        raise
