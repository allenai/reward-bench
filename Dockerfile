# This dockerfile is forked from ai2/cuda11.8-cudnn8-dev-ubuntu20.04
# To get the latest id, run `beaker image pull ai2/cuda11.8-cudnn8-dev-ubuntu20.04` 
# and then `docker image list`, to verify docker image is pulled
# e.g. `Image is up to date for gcr.io/ai2-beaker-core/public/cncl3kcetc4q9nvqumrg:latest`
FROM gcr.io/ai2-beaker-core/public/cq29hmn3sck728v1o7d0:latest

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

# this is just very slow
RUN pip install flash-attn==2.5.0 --no-build-isolation

# for olmo-instruct v1, weird install requirements
RUN pip install ai2-olmo 

# for better-pairRM
RUN pip install jinja2 

# generative installs
RUN pip install anthropic
RUN pip install openai
RUN pip install together
RUN pip install google-generativeai

# updated for Gemma 2
RUN pip install vllm==0.5.1
# from git+https://github.com/vllm-project/vllm.git@d87f39e9a9dd149f5dd7a58b4d98b21f713827b6

# for interactive session
RUN chmod -R 777 /stage/
