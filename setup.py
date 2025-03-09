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
        # List your main dependencies here
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