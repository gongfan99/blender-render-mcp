# Project structure

- `main.py` runs the MCP Streamable HTTP server with system Python. It defines `get_gpu_info` plus the asynchronous `submit_render_job`, `get_render_job_status`, and `download_render_result` tools; manages queued jobs, progress, retention, optional bearer authentication, and Blender subprocesses.
- `blender_gpu_probe.py` is executed by Blender to report its version, OptiX devices, FFmpeg support, and hardware OpenGL renderer.
- `blender_worker.py` is executed by a separate Blender background process. It loads the uploaded `.blend`, applies requested overrides, selects the render backend, writes frame progress to the job directory, and produces an H.264 MP4.
- `Dockerfile` builds the runtime image. `requirements.txt` lists the system-Python MCP dependencies.
- `main.py` starts an optional `cloudflared` tunnel in the MCP container when `CLOUDFLARE_TUNNEL_TOKEN` is set; `compose.yaml` builds one combined image and `compose.gpu.yaml` adds optional NVIDIA GPU access.
- `tests/test_main.py` contains unit tests for validation, authentication, and asynchronous job state. `tests/create_sample_blend.py` creates a small render fixture, and `tests/smoke_mcp.py` exercises submit, polling, and result download against a running MCP server.
- `README.md` documents setup, the asynchronous tool lifecycle, configuration, and smoke-test commands.
