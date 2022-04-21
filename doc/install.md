# Installation

[中文Chinese](install_ch.md)

This is a **step-by-step** tutorial of installation of CATEX

ctc-crf only support gcc above 5.0 , in centos, you can get higher versions of gcc that way:

```
yum install centos-release-scl -y

yum install devtoolset-7 -y

scl enable devtoolset-7 bash

```

We recommend to install CATEX and its dependencies in `conda` environment, otherwise you may require root permission for installing some libraries.

```
conda create -n catex python=3.8
conda activate catex

cd catex/tools
make TH_VERSION=1.8.1

```
You can use A higher version of pytorch , On my machine, I used version  1.8.1