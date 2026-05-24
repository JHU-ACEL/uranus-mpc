#!/bin/bash

# Detect GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi -L | grep -q "GPU"
then
  gpu_flag="--gpus all"
else
  gpu_flag=""
fi

if [[ $# -lt 1 ]] ; then
  echo 'Arguments: tag_name ssh_port'
  exit 1
elif [[ $# -eq 1 ]] ; then
  tagname=$1
  
  docker run \
    --rm -ti \
    --mount type=bind,source="$(pwd)",target=/workspace \
    --workdir /workspace \
    $gpu_flag \
    ${tagname} 
elif [[ $# -eq 2 ]] ; then
  tagname=$1
  ssh_port=$2
  
  docker run \
    --rm -ti \
    -p ${ssh_port}:${ssh_port} \
    -e PORT=${ssh_port} \
    --mount type=bind,source="$(pwd)",target=/workspace \
    --workdir /workspace \
    $gpu_flag \
    ${tagname} 
fi
