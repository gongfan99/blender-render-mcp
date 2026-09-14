FROM blenderkit/headless-blender:blender-5.2

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl \
    && curl --location --fail --silent --show-error \
        --output /tmp/cloudflared.deb \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$(dpkg --print-architecture).deb" \
    && apt-get install -y --no-install-recommends /tmp/cloudflared.deb \
    && rm -f /tmp/cloudflared.deb \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -c "import sys; assert sys.version_info >= (3, 10)"

RUN python3 -m venv /opt/mcp-venv
ENV PATH="/opt/mcp-venv/bin:${PATH}" \
    BLENDER_EXECUTABLE="/home/headless/blender/blender" \
    MCP_PORT="8080" \
    PYTHONUNBUFFERED="1"

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY main.py blender_worker.py blender_gpu_probe.py ./
RUN mkdir -p /home/headless/.config/blender /home/headless/.cache /home/headless/.vnc \
    && chown -R headless:headless /home/headless/.config /home/headless/.cache /home/headless/.vnc \
    && : > /dockerstartup/.initial_sudo_password

EXPOSE 8080
USER headless
CMD ["python3", "main.py"]
