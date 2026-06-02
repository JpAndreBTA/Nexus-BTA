ARG NEXUS_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel
FROM ${NEXUS_BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEXUS_BACKEND_HOST=0.0.0.0 \
    NEXUS_BACKEND_PORT=7861 \
    NEXUS_COMFY_PORT=8189 \
    NEXUS_MODELS_DIR=/workspace/NexusBTA/models \
    NEXUS_COMFY_ROOT=/workspace/NexusBTA/runtime/ComfyUI \
    NEXUS_CUSTOM_NODES_DIR=/workspace/NexusBTA/custom_nodes \
    NEXUS_ALLOW_MODEL_DOWNLOADS=0 \
    NEXUS_AUTO_TUNNEL=0

WORKDIR /workspace/NexusBTA

RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      bash \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      git-lfs \
      libglib2.0-0 \
      libgl1 \
      python3-dev \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/nexus-requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    grep -Ev '^(--extra-index-url|torch==|torchvision==|torchaudio==|xformers @|sageattention)' /tmp/nexus-requirements.txt > /tmp/nexus-requirements-runpod.txt && \
    python -m pip install -r /tmp/nexus-requirements-runpod.txt && \
    python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu130 xformers==0.0.35 && \
    (python -m pip install "sageattention>=1.0.6" || true)

COPY . /workspace/NexusBTA

RUN if [ ! -f runtime/ComfyUI/main.py ]; then \
      git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git runtime/ComfyUI; \
    fi && \
    python -m pip install -r runtime/ComfyUI/requirements.txt && \
    chmod +x scripts/start_online_runpod.sh

EXPOSE 7861 8189

CMD ["bash", "scripts/start_online_runpod.sh"]
