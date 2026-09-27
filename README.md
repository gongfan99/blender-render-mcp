# Blender MCP renderer

This image runs a Streamable HTTP MCP server from a Python virtual environment and launches a separate Blender 5.2 background process for each render. It exposes `get_gpu_info` and an asynchronous render-job API at `http://localhost:8080/mcp`.

The image also includes Tailscale. Set `TAILSCALE_AUTH_KEY` at runtime to join your tailnet as `blender-render-mcp`. Tailscale runs in userspace mode and forwards incoming tailnet connections to the same port on `127.0.0.1`, so clients on the tailnet can use `http://blender-render-mcp:8080/mcp` (or the assigned name, such as `blender-render-mcp-1`). The MCP server keeps running if Tailscale cannot connect; check the container logs and restart it after fixing the connection. Userspace mode needs no TUN device or extra container privileges. Compose persists its login state in the `tailscale-state` volume; mount `/home/headless/.local/share/tailscale` on other platforms if you want the node identity to survive container replacement. Keep the auth key in a runtime environment variable, not a Docker build argument. See Tailscale's [Linux install](https://tailscale.com/docs/install/linux) and [userspace networking behavior](https://tailscale.com/docs/reference/faq/other-vpns).

To access the container shell over your tailnet, set `TAILSCALE_SSH=true` and allow SSH in your [tailnet access policy](https://tailscale.com/docs/features/tailscale-ssh). Connect as the container's local user, for example `ssh headless@blender-render-mcp`; a RunPod SSH routing name is not a Linux username in the container. Tailscale SSH does not require a separate `sshd` process. If Tailscale assigns a suffix such as `blender-render-mcp-1`, use that name until you remove the old offline node from your tailnet.

## Start

CPU-only:

```sh
docker compose up --build -d
```

On an NVIDIA Docker host with the NVIDIA Container Toolkit configured:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

The GPU override is optional. Without GPU passthrough, Cycles uses CPU. Eevee and Workbench are reported as unavailable and render requests for them return an error. On a GPU host, the service detects GPUs visible inside the container; Cycles selects OptiX when Blender reports an OptiX device and otherwise falls back to CPU.

To build and publish the RunPod image as `gongfan99/blender-render-mcp:latest`, first authenticate with Docker Hub using `docker login`, then run `deploy.bat` from Windows.

The local Compose port mapping binds only to `127.0.0.1`. For RunPod, provide `TAILSCALE_AUTH_KEY` as a container environment variable to make the MCP endpoint available to permitted devices on your tailnet. The MCP server has no application-level authentication, so do not configure RunPod to expose port 8080 publicly.

## Tools

`get_gpu_info` reports NVIDIA devices visible through `nvidia-smi`, Blender's version and OptiX detection, its OpenGL renderer, FFmpeg support, and engine availability. Eevee and Workbench require both a visible NVIDIA GPU and a hardware OpenGL context; software renderers such as llvmpipe do not count.

Rendering uses asynchronous jobs. `submit_render_job` queues an H.264 animation, while `submit_frame_render_job` queues selected still frames as PNG files in a ZIP archive. Both return a job ID without waiting for Blender. Use `get_render_job_status` to poll either job; it returns `queued`, `running`, `completed`, or `failed`, elapsed and running time, processed and total frame counts, and any error. Download completed animation jobs with `download_render_result` and frame jobs with `download_render_frames`.

`submit_render_job` accepts:

| Argument | Type | Behavior |
| --- | --- | --- |
| `blend_file_base64` | string | Base64-encoded `.blend` file, maximum 100 MiB decoded. |
| `engine` | `cycles`, `eevee`, `workbench` or omitted | Omitted preserves the saved scene engine. |
| `frame_start`, `frame_end` | integer or omitted | Omitted preserves saved frame range. |
| `resolution_x`, `resolution_y` | integer or omitted | Omitted preserves saved resolution. |
| `resolution_percentage` | integer or omitted | Omitted preserves saved percentage. |
| `samples` | integer or omitted | Omitted preserves saved sampling settings. |

The submit response includes `job_id`, `status_tool`, and `download_tool`. Check `get_render_job_status` with that ID until the job finishes. `processed_frames` and `total_frames` are available after Blender opens the project and reports its frame range; `total_frames` can initially be `null` when the saved range is used. The server runs one Blender render at a time; additional jobs remain queued. Up to four jobs may be queued or running by default.

`submit_frame_render_job` accepts the same engine, resolution, and sample overrides as `submit_render_job`, plus a required `frames` list. The list must contain 1–100 unique integer frame numbers within the scene's saved frame range. The worker renders each requested frame once, preserves the scene's output resolution unless overridden, and returns numbered PNGs (for example, `frame_000024.png`) in `rendered_frames.zip`. The ZIP archive is capped at 250 MiB. Use the returned `download_tool` after status becomes `completed` to retrieve it as an embedded `application/zip` resource.

MCP call sequence:

```text
submit_render_job({"blend_file_base64": "...", "engine": "cycles"})
get_render_job_status({"job_id": "<returned id>"})
download_render_result({"job_id": "<returned id>"})
```

For selected still frames:

```text
submit_frame_render_job({"blend_file_base64": "...", "frames": [1, 24, 48], "engine": "eevee"})
get_render_job_status({"job_id": "<returned id>"})
download_render_frames({"job_id": "<returned id>"})
```

Animation MP4s use H.264 and are capped at 250 MiB; selected-frame PNG archives are also capped at 250 MiB. A job may render the saved animation frame range, so keep the range and output dimensions reasonable. Pack external textures and other assets into the `.blend` file before uploading; only the `.blend` bytes are sent. Blender's automatic Python execution is disabled for uploaded projects.

Jobs and results are held on the server's local filesystem for 24 hours by default. They are process-local and are lost if the container restarts; use one server instance for submit, status polling, and download.

Blender keeps the scene's saved engine, frame range, resolution, and samples unless the request overrides them. The worker selects the Cycles device and output path and format. Animation jobs use MPEG-4/H.264; selected-frame jobs save PNG images and package them in a ZIP archive. For Workbench, a sample override selects the closest supported value.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `TAILSCALE_AUTH_KEY` | unset | Auth key for joining the tailnet as `blender-render-mcp`; unset or empty skips Tailscale startup. |
| `TAILSCALE_SSH` | unset | Set to `true` to enable Tailscale SSH for the container's `headless` user; requires a tailnet SSH access rule. |
| `MCP_PORT` | `8080` | HTTP listen port inside the container. |
| `BLENDER_EXECUTABLE` | `/home/headless/blender/blender` | Blender executable path. |
| `MAX_RENDER_SECONDS` | `1800` | Maximum duration of one render process. |
| `MAX_ACTIVE_RENDER_JOBS` | `4` | Maximum queued and running jobs. Completed jobs do not count toward this limit. |
| `JOB_RETENTION_SECONDS` | `86400` | Time completed and failed jobs remain available before cleanup. |
| `JOB_STORAGE_DIR` | system temp directory + `blender-mcp-jobs` | Local directory for uploaded projects, progress, logs, and render results. |

The decoded `.blend` input limit is fixed at 100 MiB; MP4 and PNG ZIP outputs are each limited to 250 MiB, and frame-list jobs accept at most 100 frames. Only one Blender render runs at a time in each container.

## Smoke tests

Run the unit tests with Python 3.10 or later after installing `requirements.txt`:

```sh
python -m unittest discover -s tests -v
```

For a live CPU-only test, create a tiny sample file with Blender in the image, then run the MCP client smoke test. It checks that Eevee and Workbench jobs fail clearly without a GPU and that Cycles returns a valid embedded MP4:

```sh
docker run --rm --entrypoint /home/headless/blender/blender -v "${PWD}:/workspace" -w /workspace blender-render-mcp:local --background --python tests/create_sample_blend.py -- /workspace/tests/cpu_sample.blend
python tests/smoke_mcp.py --blend tests/cpu_sample.blend --check-cpu-denials --frame 1
```

On an NVIDIA host, start Compose with `compose.gpu.yaml` and run `python tests/smoke_mcp.py --blend tests/cpu_sample.blend --check-gpu-engines` to render through Cycles, Eevee, and Workbench. Add `--frame 1` to also verify a selected-frame PNG ZIP render; the requested frame must be within the blend's saved frame range.
