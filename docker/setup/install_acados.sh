#!/bin/bash

cd /
git clone https://github.com/acados/acados.git
cd acados

git submodule update --recursive --init --depth=1

sudo apt-get update && sudo apt-get install -y cmake

mkdir -p build
cd build

cmake -DACADOS_WITH_QPOASES=ON .. && make install

cd /

pip install /acados/interfaces/acados_template