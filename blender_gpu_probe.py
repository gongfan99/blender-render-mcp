"""Small Blender-side probe used by the system-Python MCP server."""

import json
import sys

import bpy


result = {
    "available": True,
    "blender_version": bpy.app.version_string,
    "ffmpeg_available": bool(getattr(bpy.app.build_options, "codec_ffmpeg", False)),
    "optix_available": False,
    "optix_devices": [],
    "graphics_available": False,
}

try:
    import gpu

    gpu.init()
    graphics = {
        "vendor": gpu.platform.vendor_get(),
        "renderer": gpu.platform.renderer_get(),
        "version": gpu.platform.version_get(),
    }
    renderer_name = (graphics["renderer"] or "").lower()
    result["graphics_device"] = graphics
    result["graphics_available"] = bool(renderer_name) and not any(
        marker in renderer_name
        for marker in ("llvmpipe", "softpipe", "swiftshader", "software rasterizer")
    )
except Exception as exc:
    result["graphics_error"] = str(exc)

try:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    # Blender 5.2 can expose an empty dynamic enum_items list even though
    # assigning OPTIX and enumerating devices works on a compatible GPU.
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    result["optix_devices"] = [
        {"name": device.name, "id": device.id}
        for device in preferences.devices
        if device.type == "OPTIX"
    ]
    result["optix_available"] = bool(result["optix_devices"])
except Exception as exc:  # The probe should still report Blender and FFmpeg status.
    result["optix_error"] = str(exc)

print("BLENDER_MCP_GPU_PROBE=" + json.dumps(result), flush=True)
