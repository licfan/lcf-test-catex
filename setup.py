#!/usr/bin/env Python3

"""CATEX setup script."""

import os
from setuptools import find_packages
from setuptools import setup

requirements = {
    "install": [
        "setuptools>=38.5.1",
        "configargparse>=1.2.1",
        "typeguard>=2.7.0",
        "humanfriendly",
        "scipy>=1.4.1",
        "filelock",
        "librosa>=0.8.0",
        "jamo==0.4.1",
        "PyYAML>=5.1.2",
        "soundfile>=0.10.2",
        "h5py>=2.10.0",
        "kaldiio>=2.17.0",
        "torch>=1.3.0",
        "torchaudio",
        "torch_optimizer",
        "transformers",
        "torch_complex",
    ],
    "setup": ["numpy", "pytest-runner"],
    # train: The modules invoked when training only.
    "train": [
        "pillow>=6.1.0",
        "tensorboard>=1.14",
    ],
    "test": [
        "pytest>=3.3.0",
        "pytest-timeouts>=1.2.1",
        "pytest-pythonpath>=0.7.3",
        "pytest-cov>=2.7.1",
        "hacking>=2.0.0",
        "mock>=2.0.0",
        "pycodestyle",
        "jsondiff>=1.2.0",
        "flake8>=3.7.8",
        "flake8-docstrings>1.3.1",
        "black",
    ],
    "doc": [
        "Sphinx==2.1.2",
        "sphinx-rtd-theme>=0.2.4",
        "sphinx-argparse>=0.2.5",
        "commonmark==0.8.1",
        "recommonmark>=0.4.0",
        "nbsphinx>=0.4.2",
        "sphinx-markdown-tables>=0.0.12",
    ]
}

requirements["test"].extend(requirements["train"])

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {
    k: v for k, v in requirements.items() if k not in ["install", "setup"]
}

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "catex", "version.txt")
with open(version_file, "r") as f:
    version = f.read().strip()

setup(
    name="catex",
    version=version,
    author="Qiangqiang Wang, Zhijian Ou",
    author_email="ozj@tsinghua.edu.cn",
    description = "CATEX: Crf based Asr Toolkit with EXtensions",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(), 
    long_description_content_type="test/markdown",
    license="Apaceh Software License",
    packages=find_packages(include=["catex*"]),
    package_data={"catex": ["version.txt"]},
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    python_requires=">=3.7.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: Android",
        "Operating System :: POSIX :: Linux",
        "Operating System :: iOS",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

