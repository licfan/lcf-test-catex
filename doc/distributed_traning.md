# Distributed training

CATEX only supports DDP for data-parallel distributed training.

## Examples

### Single node with 4 GPUS 
```bash
% export CUDA_VISIBLE_DEVICES="0,1,2,3"
% python -m catex.bin.asr_train  --dist_url tcp://localhost:12357
```

### 2Host and 2GPU  for each host 

```bash
(host1)% export CUDA_VISIBLE_DEVICES="0,1"
(host1)% python -m catex.bin.asr_train --rank 0 --dist_url tcp://{host1_ip_address}:{master_port} --data data_path

(host2)% export CUDA_VISIBLE_DEVICES="0,1"
(host2)% python -m catex.bin.asr_train --rank 1 --dist_url tcp://{host1_ip_address}:{master_port} --data data_path
```

Tips:
1. Data_path must A path that can be accessed by both hosts
2. In current version, each host must have the same GPU available