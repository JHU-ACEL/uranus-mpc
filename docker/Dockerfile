FROM ubuntu:22.04
# FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG UNAME
ARG UID
ARG GID

ARG DEBIAN_FRONTEND=noninteractive

COPY ./setup .

# 2. Install system dependencies & PPA
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg2 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update

# 3. Install Python 3.12 and build tools
# Note: python3.12-distutils is often needed for pip on deadsnakes
RUN apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    coinor-libipopt-dev \
    gcc g++ gfortran git patch wget pkg-config \
    liblapack-dev libmetis-dev cmake vim \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Python 3.12 as the default 'python' and 'python3'
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# 5. Install and upgrade pip for 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

#RUN apt-get update

#RUN apt install software-properties-common

#RUN add-apt-repository ppa:deadsnakes/ppa

#RUN apt-get install -y gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev cmake curl apt-utils python3.12 python3.12-pip gnupg2 vim git libsm6 libxext6 libxrender-dev nano screen tmux htop openssh-client unzip

#RUN ln -s /usr/bin/python3 /usr/bin/python

#RUN apt-get update

#RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && ln -fs /usr/share/zoneinfo/US/Pacific /etc/localtime && apt-get install -y tzdata && dpkg-reconfigure --frontend noninteractive tzdata

RUN python3.12 -m pip install -U pip && python3.12 -m pip install -U opencv-python numpy Pillow matplotlib scipy natsort h5py seaborn ipython jupyter jupyterlab matplotlib scipy marimo jax cvxpy equinox optax diffrax wandb planetMagFields

RUN python3.12 -m pip install --upgrade "jax[cuda12]"


# Install moreau
ENV MOREAU_LICENSE_KEY=
RUN python3.12 -m pip install moreau[cuda12] --extra-index-url https://kkMeu-DhcjKJQMlfKzSqkxXqjEQndqjgE:@pypi.fury.io/optimalintellect/

# Install casadi
RUN python3.12 -m pip install casadi

# Install ipopt
#RUN apt-get install -y coinor-libipopt-dev -y

# Install acados
RUN /install_acados.sh

ENV ACADOS_SOURCE_DIR="/acados"
ENV ACADOS_PATH="/acados"
ENV LD_LIBRARY_PATH="/acados/lib:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]
