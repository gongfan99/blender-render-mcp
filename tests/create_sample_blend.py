"""Create a tiny animated Cycles scene used for the Docker smoke test."""

import sys

import bpy


try:
    output = sys.argv[sys.argv.index("--") + 1]
except (ValueError, IndexError) as exc:
    raise SystemExit("usage: blender --background --python create_sample_blend.py -- output.blend") from exc

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.render.resolution_x = 48
scene.render.resolution_y = 48
scene.render.resolution_percentage = 100
scene.frame_start = 1
scene.frame_end = 1
scene.render.fps = 24
bpy.ops.wm.save_as_mainfile(filepath=output)
