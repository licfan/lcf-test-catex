#!/usr/bin/env bash
set -euo pipefail
log(){
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $# -ne 3 ]; then
    log "Usage: $0 <use_conda| true or false> <torch_version> <cuda_version>"
    exit 1;
elif [ $# -eq 3 ]; then
    use_conda="$1"
    if [ "${use_conda}" != false ] && ["${use_conda}" != true ]; then
        log "[ERROR] <use_conda> must be true or false, but ${use_conda} is given."
        log "Usage: $0 <use_conda| true or false> <torch_version> <cuda_version>"
        exit 1
    fi
    torch_version="$2"
    cuda_version="$3"
fi

if [ "${cuda_version}" = cpu ] || [ "${cuda_version}" = CPU ]; then
    cuda_version=
fi

python_version=$(python3 -c "import sys; print(sys.version.split()[0])")
cuda_version_without_dot="${cuda_version/\./}"

python_plus(){
    python3 <<EOF
from distutils.version import LooseVersion as L
if L('$python_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

pytorch_plus(){
    python3 <<EOF
from distutils.version import LooseVersion as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

#TODO: support install pytorch in  MAC OS and windows
install_torch(){
# Usage: install_torch <torchaudio-version> <default-cuda-version-for-pip-install-torch>
    if "${use_conda}"; then
        if [ -z "${cuda_version}" ]; then
            log conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
            conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
        else
            #https://anaconda.org/conda-forge/cudatoolkit/files
            cuda_toolkit_channel=conda-forge
            #TODO: Test cuda 11.1, 11.2 with pytorch and conda
            log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch -c ${cuda_toolkit_channel}
            conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch -c ${cuda_toolkit_channel}
        fi
    else
        if $(pytorch_plus 1.10.0); then
            if [-z "${cuda_version}" ]; then
                log python3 -m pip intall "torch==${torch_version}+cpu" "torchaudio==$1+cpu" -f https://download.pytorch.org/whl/torch_stable.html
                python3 -m pip intall "torch==${torch_version}+cpu" "torchaudio==$1+cpu" -f https://download.pytorch.org/whl/torch_stable.html
            else
                log python3 -m pip install "torch==${torch_version}+cu${cuda_version_without_dot}" "torchaudio==$1+cu${cuda_version_without_dot}" -f "https://download.pytorch.org/whl/torch_stable.html"
                python3 -m pip install "torch==${torch_version}+cu${cuda_version_without_dot}" "torchaudio==$1+cu${cuda_version_without_dot}" -f "https://download.pytorch.org/whl/torch_stable.html"
            fi
        else
            if [ -z "${cuda_version}" ]; then    
                log python3 -m pip install "torch==${torch_version}+cpu" "torchaudio==$1" -f https://download.pytorch.org/whl/torch_stable.html
                python3 -m pip install "torch==${torch_version}+cpu" "torchaudio==$1" -f https://download.pytorch.org/whl/torch_stable.html
            else
                log python3 -m pip install "torch==${torch_version}+cu${cuda_version_without_dot}" "torchaudio==$1" -f "https://download.pytorch.org/whl/torch_stable.html"
                python3 -m pip install "torch==${torch_version}+cu${cuda_version_without_dot}" "torchaudio==$1" -f "https://download.pytorch.org/whl/torch_stable.html"
            fi
        fi
    fi    
}

check_python_version(){
    if $(python_plus $1) || ! $(python_plus 3.6); then
        log "[ERROR] pytorch=${torch_version} doesn't provide binary build for python>=$1,<3.6, but your python is ${python_version}"
        exit 1
    fi
}
# TODO: Provides specific prompts when CUDA version do not match
check_cuda_version(){
    supported=false
    for v in "" $@; do
        [ "${cuda_version}" = "${v}" ] && supported=true
    done
    if ! "${supported}"; then
        # See https://anaconda.org/pytorch/pytorch/files to provided version
        log "[ERROR] Pytorch=${torch_version} binary for CUDA=${cuda_version} is not provided. $@ are supported."
        exit 1
    fi
}

log "[INFO] python_version=${python_version}"
log "[INFO] torch_version=${torch_version}"
log "[INFO] cuda_version=${cuda_version}"

if $(pytorch_plus 1.10.2); then
    log "[ERROR] This script doesn't support pytorch=${torch_version}."
    exit 1
elif $(pytorch_plus 1.10.1); then
    check_python_version 3.10 # Error if python>=<number>
    check_cuda_version 11.3 11.1 10.2 # Error if cuda_version doesn't match with any given numbers
    install_torch 0.10.1 10.2 # install 
elif $(pytorch_plus 1.10.0); then
    check_python_version 3.10 #Error if python>=<number>
    check_cuda_version 11.3 11.1 10.2 # Error if cuda_version does'nt math with any given numbers
    install_torch 0.10.0 10.2 # install_torch <torch-audio-ver> 
elif $(pytorch_plus 1.9.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1
elif $(pytorch_plus 1.9.1); then
    check_python_version 3.10
    check_cuda_version 11.1 10.2
    install_torch 0.9.1 10.2
elif $(pytorch_plus 1.9.0); then
    check_python_version 3.10
    check_cuda_version 11.1 10.2
    install_torch 0.9.0 10.2
elif $(pytorch_plus 1.8.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1
elif $(pytorch_plus 1.8.1); then
    check_python_version 3.10
    check_cuda_version 11.1 10.2 10.1
    install_torch 0.8.1 10.2
elif $(pytorch_plus 1.8.0); then
    check_python_version 3.10
    check_cuda_version 11.1 10.2 10.1
    install_torch 0.8.0 10.2
elif $(pytorch_plus 1.7.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1
elif $(pytorch_plus 1.7.1); then
    check_python_version 3.10
    check_cuda_version 11.0 10.2 10.1 9.2
    install_torch 0.7.0 10.2
elif $(pytorch_plus 1.6.1); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1
elif $(pytorch_plus 1.6.0); then
    check_python_version 3.9
    check_cuda_version 10.2 10.1 9.2
    install_torch 0.7.0 10.2
elif $(pytorch_plus 1.5.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1
elif $(pytorch_plus 1.5.1); then
    check_python_version 3.9
    check_cuda_version 10.2 10.1 9.2
    install_torch 0.5.1 10.2
elif $(pytorch_plus 1.5.0); then
    check_python_version 3.9
    check_cuda_version 10.2 10.1 9.2
    install_torch 0.5.0 10.2
elif $(pytorch_plus 1.4.1); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1
elif $(pytorch_plus 1.4.0); then
    check_python_version 3.9
    check_cuda_version 10.1 10.0 9.2
    install_torch 0.4.0 10.1
elif $(pytorch_plus 1.3.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1
elif $(pytorch_plus 1.3.1); then
    check_python_version 3.8
    check_cuda_version 10.1 10.0 9.2
    install_torch 0.3.2 10.1
elif $(pytorch_plus 1.3.0); then
    check_python_version 3.8
    check_cuda_version 10.1 10.0 9.2
    install_torch 0.3.1 10.1
elif $(pytorch_plus 1.2.1); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1
elif $(pytorch_plus 1.2.0); then
    check_python_version 3.8
    check_cuda_version 10.0 9.2
    install_torch 0.3.0 10.0
else
    log "[ERROR] pytorch=${torch_version} doesn't support pytorch=${torch_version}"
    exit 1
fi