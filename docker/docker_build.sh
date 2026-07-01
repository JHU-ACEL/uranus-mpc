#!/bin/bash

if [[ $# -lt 2 ]] ; then
  echo 'Arguments: dockerfile tag_name'
  exit 1
fi

dockerfile=$1
tagname=$2

docker build --rm \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg UNAME=$(whoami) \
  -f ${dockerfile} \
  -t ${tagname} $(dirname ${dockerfile})

# Print success message
echo "Successfully built Docker image with tag: ${tagname}"