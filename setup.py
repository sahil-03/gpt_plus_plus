import logging
import os
import re

import setuptools

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(__file__)


with open(os.path.join(ROOT_DIR, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gpt_plus_plus",
    version="1.0",
    packages=setuptools.find_packages(),
    description="GPT++",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "pycuda",
        "tqdm==4.58.0",
        "requests==2.25.1",
        "importlib-metadata==3.7.0",
        "filelock==3.0.12",
        "sklearn==0.0",
        "tokenizers==0.20",
        "explainaboard_client==0.0.7",
        "einops==0.8.0",
        "transformers==4.46.3",
        "sacrebleu==2.5.1",
        "triton",
    ],
    python_requires=">=3.8",
    author="Sahil Adhawade",
    author_email="sahil.adhawade@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
)