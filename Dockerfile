# This dockerfile is forked from ai2/cuda11.8-cudnn8-dev-ubuntu20.04
# To get the latest id, run `beaker image pull ai2/cuda11.8-cudnn8-dev-ubuntu20.04` and then `docker image list`
FROM gcr.io/ai2-beaker-core/public/cmunp4nu6epv94rv5si0:latest

RUN apt update && apt install -y openjdk-8-jre-headless

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TODO: at least 3 things to install too:
# TRL (pip)
# upgrade transformers (check latest open instruct version)
# fastchat (pip3 install "fschat[model_worker,webui]")
# pip install -e .
# huggingface login (add as beaker token)

# RUN pip install packaging
RUN pip install flash-attn --no-build-isolation
RUN pip install -r requirements.txt
RUN pip install "fschat[model_worker,webui]"
RUN pip install -e .

COPY herm herm
# COPY eval eval
COPY ds_configs ds_configs
COPY scripts scripts
RUN chmod +x scripts/*

# for interactive session
RUN chmod -R 777 /stage/
