FROM blenderkit/headless-blender:blender-5.2

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl \
    && curl -fsSL https://tailscale.com/install.sh -o /tmp/install-tailscale.sh \
    && sh /tmp/install-tailscale.sh \
    && rm -f /tmp/install-tailscale.sh \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -c "import sys; assert sys.version_info >= (3, 10)"

# Tailscale SSH needs to set the headless user's supplementary group when
# the container runtime starts the process without supplementary groups.
RUN setcap cap_setgid+ep /usr/sbin/tailscaled

RUN python3 -m venv /opt/mcp-venv
ENV PATH="/opt/mcp-venv/bin:${PATH}" \
    BLENDER_EXECUTABLE="/home/headless/blender/blender" \
    MCP_PORT="8080" \
    PYTHONUNBUFFERED="1"

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY main.py blender_worker.py blender_gpu_probe.py ./
RUN mkdir -p /home/headless/.config/blender /home/headless/.cache /home/headless/.vnc /home/headless/.local/share/tailscale \
    && chown -R headless:headless /home/headless/.config /home/headless/.cache /home/headless/.vnc /home/headless/.local \
    && : > /dockerstartup/.initial_sudo_password

EXPOSE 8080
USER headless
CMD ["python3", "main.py"]
