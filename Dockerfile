# TODO: Update this when releasing RewardBench publicly
# This dockerfile is forked from ai2/cuda11.8-cudnn8-dev-ubuntu20.04
# To get the latest id, run `beaker image pull ai2/cuda11.8-cudnn8-dev-ubuntu20.04` 
# and then `docker image list`, to verify docker image is pulled
# e.g. `Image is up to date for gcr.io/ai2-beaker-core/public/cncl3kcetc4q9nvqumrg:latest`
FROM gcr.io/ai2-beaker-core/public/cnh71s73h9jm088farmg:latest

RUN apt update && apt install -y openjdk-8-jre-headless

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/

RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TODO: Install flash attention when training code is complete, and consider using built in flash attn
# RUN pip install flash-attn==2.2.2 --no-build-isolation

# TODO: enable these when training code is complete
COPY rewardbench rewardbench
COPY scripts scripts
COPY setup.py setup.py
COPY Makefile Makefile
COPY README.md README.md
RUN pip install -e .
RUN chmod +x scripts/*
RUN pip install flash-attn==2.5.0 --no-build-isolation

# for interactive session
RUN chmod -R 777 /stage/
