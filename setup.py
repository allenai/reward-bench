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

setup(
    name="herm",
    version="0.1.0.dev",
    author="Nathan Lambert",
    author_email="nathanl@allenai.org",
    description="Tools for evaluating reward models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/allenai/herm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "accelerate",
        "bitsandbytes",
        "black==23.1.0",
        "datasets",
        "flake8>=6.0",
        "fschat[model_worker,webui]",
        "huggingface_hub",
        "isort>=5.12.0",
        "pandas",
        "pytest",
        "spacy-alignments",  # dependency for aligning tokenizers
        "scipy",
        "tabulate",  # dependency for markdown rendering in pandas
        "tokenizers",
        "transformers",
        "trl>=0.7.7",
    ],
)
