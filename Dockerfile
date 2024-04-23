# TODO: Update this when releasing RewardBench publicly
# This dockerfile is forked from ai2/cuda11.8-cudnn8-dev-ubuntu20.04
# To get the latest id, run `beaker image pull ai2/cuda11.8-cudnn8-dev-ubuntu20.04` 
# and then `docker image list`, to verify docker image is pulled
# e.g. `Image is up to date for gcr.io/ai2-beaker-core/public/cncl3kcetc4q9nvqumrg:latest`
FROM gcr.io/ai2-beaker-core/public/cojd4q5l9jpqudh7p570:latest

RUN apt update && apt install -y openjdk-8-jre-headless

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get -y install git-lfs

WORKDIR /stage/

RUN pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY rewardbench rewardbench
COPY scripts scripts
COPY setup.py setup.py
COPY Makefile Makefile
COPY README.md README.md
RUN pip install -e .
RUN chmod +x scripts/*
RUN pip install flash-attn==2.5.0 --no-build-isolation
RUN pip install ai2-olmo 
# TODO remove above when olmo supported in Transformers verion
RUN pip install jinja2 
# for better-pairRM
# generative installs
RUN pip install anthropic
RUN pip install openai
RUN pip install git+https://github.com/vllm-project/vllm.git@d87f39e9a9dd149f5dd7a58b4d98b21f713827b6

# for interactive session
RUN chmod -R 777 /stage/
