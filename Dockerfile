# Use public Nvidia images (rather than Beaker), for reproducibility
FROM --platform=linux/amd64 nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 
#nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

RUN apt update && apt install -y openjdk-8-jre-headless

ARG DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

# Install base tools.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    jq \
    language-pack-en \
    make \
    sudo \
    unzip \
    vim \
    wget \
    parallel \
    iputils-ping \
    tmux

# This ensures the dynamic linker (or NVIDIA's container runtime, I'm not sure)
# puts the right NVIDIA things in the right place (that THOR requires).
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Install conda. We give anyone in the users group the ability to run
# conda commands and install packages in the base (default) environment.
# Things installed into the default environment won't persist, but we prefer
# convenience in this case and try to make sure the user is aware of this
# with a message that's printed when the session starts.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh \
    && echo "32d73e1bc33fda089d7cd9ef4c1be542616bd8e437d1f77afeeaf7afdb019787 Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" \
        | sha256sum --check \
    && bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

ENV PATH=/opt/miniconda3/bin:/opt/miniconda3/condabin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Ensure users can modify their container environment.
RUN echo '%users ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Install uv for fast, reliable package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency manifests first (better layer caching)
COPY pyproject.toml uv.lock ./

# Pin torch<=2.8 FIRST so we can use prebuilt flash-attn wheels
# (vllm extra not installed here - use Dockerfile.vllm for that)
RUN uv pip install --system "torch>=2.1,<=2.8" torchvision torchaudio

# Install dependencies (api + v1 extras, NO vllm)
RUN uv sync --frozen --no-install-project --extra api --extra v1

# flash-attn prebuilt wheel (fast!) + jinja2 BEFORE source copy
# This layer is cached unless deps change
RUN uv pip install --system https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/flash_attn-2.7.4+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN uv pip install --system jinja2

# Now copy source code (invalidates only later layers, but flash-attn cached)
COPY rewardbench rewardbench
COPY scripts scripts
COPY Makefile Makefile
COPY README.md README.md

# Install the project (non-editable for deployment)
RUN uv sync --frozen --no-editable --extra api --extra v1
RUN chmod +x scripts/*

# for interactive session
RUN chmod -R 777 /stage/
