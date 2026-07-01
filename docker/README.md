# Docker 

### Modify Dockerfile
You must modify the docker file to include your Moreau License Key.

### Build the Docker Container
Use the `docker_build.sh` script to build the Docker image:
```bash
./docker/docker_build.sh docker/Dockerfile uranus-mpc:v1
```

**Arguments:**
- `dockerfile`: Path to the Dockerfile (e.g., `docker/Dockerfile`)
- `tag_name`: Docker image tag (e.g., `uranus-mpc:v1`)

The build script automatically:
- Sets the user ID and group ID to match your current user
- Sets the username to your current username
- Builds the image with the specified tag

### Run the Docker Container
Use the `docker_run.sh` script to run the Docker container:

```bash
./docker/docker_run.sh uranus-mpc:v1
```

**Arguments:**
- `tag_name`: The Docker image tag to run (e.g., `uranus-mpc:v1`)
- `ssh_port` (optional): The port forwarding between host machine and Docker container (e.g., `8888`) 

### Run Jupyter Lab (from inside the container):
```bash
jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser --allow-root
```

### Run marimo (from inside the container):
```bash
marimo edit --headless --host 0.0.0.0 --port=$PORT
```

### Use SSH port forwarding to run a notebook remotely
```bash
ssh -L localhost:${PORT}:localhost:${PORT} -N -f ${UID}@{remote}
```
where `${PORT}` is the port designated when you `docker run` and `${UID}@{remote}` is the remote instance (e.g., `cauligi@enceladus.wse.jhu`).
