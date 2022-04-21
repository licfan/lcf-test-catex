#!/usr/bin/env bash

# set bash to 'debug' mode, it will exit on:
# -e 'error'
# -u 'undefined variable'
# -o pipefail 'error in pipeline'
set -e
set -u
set -o pipefail


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0

# General configuration
stage=1         # start from this stage
stop_stage=100  # stop at this stage
ngpu=1          # number of gpus
num_nodes=1     # number of nodes
nj=32
gpu_inference=false


. utils/parse_options.sh

. ./path.sh

if [ ${stage} -le 1 ] && [ $(stop_stage) -ge 1 ]; then
    log "Stage1: Data preparation"
    local/data.sh 
fi

