# Adaptive MPC for Satellites Using Learning-Based Methods
## Magnetorquer-only attitude control of satellite in unknown magnetic field.
<img width="878" height="503" alt="hero_white_background" src="https://github.com/user-attachments/assets/e5bd45fb-a483-4015-80d6-821da89fa2e2" />

## Prerequisites
1. [Docker](https://docker-curriculum.com/)
2. GPU Access

## Startup
1. In workstation's bash shell, in desired directory, clone this repo
```bash
git clone git@github.com:JHU-ACEL/uranus-mpc.git
```
2. In the directory you just cloned, build image by running:
```bash
./docker/docker_build.sh Dockerfile IMAGE_NAME:TAG # ex: ./docker/docker_build.sh Dockerfile uranus-mpc:v2
```
3. To verify that the image built properly, run
```bash
docker images
```
4. To run a container, run
```bash
./docker/docker_run.sh <image_repository:tag> <PORT> # example: "./docker/docker_run.sh uranus-mpc:v1 8888"
```
5. To start marimo, run 
```bash
marimo edit --headless --host 0.0.0.0 --port=$PORT
```
6. If you are accessing a workstation with a GPU virtually, you will need to port forward to your local machine. In your local command line, run
```bash
ssh -L localhost:${PORT}:localhost:${PORT} -N -f ${UID}@{remote} # ex: "ssh -L localhost:8888:localhost:8888 -N -f pschwa24@enceladus.wse.jhu.edu"
```
7. On your browser on your local machine you can go to URL given in marimo (e.g. https://0.0.0.0:8888) and copy and paste the access token to access marimo.

## Working with the library
The main functionality is set up and demonstrated in the `Spacecraft_Control.py` notebook. Here you can generate a batch of orbital trajectories and their associated magnetic fields, set up the CEM and TVLQR controller and roll out a single example, and then generate a batch of trajectories comparing the results from when the controller has access to the true B-field, learned full-order b-field, learned low-order b-field, and learned low-order b-field with online adaptation.

The library is set-up to be fully compatible with JAX, and thus everything is parallelized and JIT-compatible. It makes use of [Equinox](https://docs.kidger.site/equinox/all-of-equinox/) to make this easy and set-up with Object-Oriented Programming.

The offline training is done in the `B_Field_Learning.py` notebook. The training is done in JAX using the equinox and optax libraries.
