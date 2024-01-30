# This dockerfile is forked from ai2/cuda11.8-cudnn8-dev-ubuntu20.04
# TODO: Need to update to the latest ID
FROM gcr.io/ai2-beaker-core/public/cl5erg1ebj67821o3200:latest

RUN apt update && apt install -y openjdk-8-jre-headless

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# RUN pip install packaging
RUN pip install flash-attn --no-build-isolation
RUN pip install -r requirements.txt

COPY herm herm
# COPY eval eval
# COPY ds_configs ds_configs
COPY scripts scripts
RUN chmod +x scripts/*

# for interactive session
RUN chmod -R 777 /stage/
