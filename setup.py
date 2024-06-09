# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

# instructions for releasing new version: update the version number, then follow
# from 6 https://github.com/huggingface/diffusers/blob/49b959b5408b97274e2ee423059d9239445aea26/setup.py#L36C43-L38C1
# this has not yet been pushed to pypyi-test
setup(
    name="rewardbench",
    version="0.1.3",  # do not import from init, or we get a weird build error
    author="Nathan Lambert",
    author_email="nathanl@allenai.org",
    description="Tools for evaluating reward models",
    entry_points={
        "console_scripts": [
            "rewardbench=rewardbench.rewardbench:main",
            "rewardbench-gen=scripts.run_generative:main",
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/allenai/rewardbench",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "accelerate",
        "bitsandbytes",
        "black",
        "datasets",
        "deepspeed",
        "einops",
        "flake8>=6.0",
        "fschat",
        "huggingface_hub",
        "isort>=5.12.0",
        "pandas",
        "peft",
        "pytest",
        "scipy",
        "sentencepiece",
        "tabulate",  # dependency for markdown rendering in pandas
        "tokenizers",
        "torch",
        "tiktoken==0.6.0",  # added for llama 3
        "transformers==4.40.0",  # pinned at llama 3
        "trl>=0.8.2",  # fixed transformers import error
    ],
    extras_require={
        "generative": [
            "vllm>=0.4.1",
            "openai",
            "anthropic",
            "together",
        ],
        "training": [
            "wandb",
        ],
    },
)
