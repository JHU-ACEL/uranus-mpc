# Learning-Based Adaptive MPC for Magnetorquer-Only Satellite Attitude Control
<img width="867" height="492" alt="hero_white_background" src="https://github.com/user-attachments/assets/659210b3-bb81-446d-8291-b291ccb81eb3" />

## Prerequisites
1. [Docker](https://docker-curriculum.com/)
2. GPU Access
3. [Moreau](https://www.moreau.so/blog/announcing-moreau) (requires license)

## Startup
1. In workstation's bash shell, in desired directory, clone this repo
```bash
git clone git@github.com:JHU-ACEL/uranus-mpc.git
```
2. In the directory you just cloned, follow the steps in `docker/README.md` to modify the `Dockerfile` to include your Moreau License Key and build the docker image.
4. To verify that the image built properly, run
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

## Examples
- `Spacecraft_Control.py` demonstrates the full functionality of the library and controller with online adaptation.
- `B_field_Learning.py` shows the training pipeline for training the magnetic field models offline.

## Working with the library
The library is set-up to be fully compatible with JAX, and thus everything is parallelized and JIT-compatible. It makes use of [Equinox](https://docs.kidger.site/equinox/all-of-equinox/) to make this easy and set-up with Object-Oriented Programming.

### Base functions
First, set the `dynamics` class. The spacecraft attitude dynamics with magnetorquer-attitude control is initialized as follows:
```bash
spacecraft_dynamics = SpacecraftDynamics(mag_model=model_s,planet=planet)
```

Most of the primary functionality is built into the `TrajectoryGenerator` class, which is initialized by providing it with dynamics and a time-step.
```bash
system = TrajectoryGenerator(dynamics=spacecraft_dynamics, dt=dt)
```

After initializing the class, you can generate a trajectory. If the only thing provided is the initial state, key, and number of steps, the initial states are just propagated according to the dynamics. 
```bash
trajectory, controls = system.generate_trajectory(initial_state=initial_state, key=key, num_steps=num_steps)
```

Alternatively, you could provide a fixed control sequence that it will propagate.
```bash
trajectory, controls = system.generate_trajectory(initial_state=initial_state, key=key, num_steps=num_steps, control_sequence=control_sequence)
```

If your dynamics require and external dynamics parameter (e.g. magnetic field values), then those must also be provided to the `external_dynamics_params` field. Additionally, a random disturbance force/torque is applied at each step if a standard deviation for it is provided.
```bash
trajectory, controls = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key1, external_dynamics_params=b_field_values, control_sequence=control_sequence, num_steps=num_steps, noise_std=noise_std_dyn)
```

You can also propagate controlled trajectories. First, initialize the controllers
```bash
qp_controller = QPController(system, Q_qp, Qf_qp, R_qp, state_limits, control_limits, horizon_qp) # high_level_controller
fb_controller = FeedbackController(system,Q_fb,Qf_fb,R_fb,control_limits) # low_level_controller
```
and then provide them to the `generate_trajectory` function along with a target state and how frequently the `high_level_controller` should replan. The `high_level_controller` is called every `replan_freq` timesteps, and the `low_level_controller` is called every timestep and tracks the nominal trajectory produced by the `high_level_controller`. If no `low_level_controller` is provided, the nominal control inputs produced by the `high_level_controller` are just used at each timestep. If no `high_level_controller` is provided, then the `low_level_controlller` just works by itself trying to reach the target state provided.
```bash
trajectory, controls = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b_field_values, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps, noise_std=noise_std_dyn)
```

A full batch of trajectories can just as easily be generated. If an arrays of `batch_size` are provided, then it will use all of them, but if you only want to change some things for each trajectory (e.g. have the same initial state for all of them, but different target states), that works too. The following example just provides one initial state, but `batch_size` different target states and b field values. The noise always changes for each trajectory. 
```bash
traj_batch, cntrl_batch = system.generate_trajectory_batch(initial_states=init_state, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=None, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, chunk_size=None, num_steps=num_steps_batch, noise_std=noise_std_dyn)
```

### Using learned b-field models and online-adaptation
In order to use the learned b-field model in the controller instead of the true values, provide the cartesian coordinates in the PCI frame of the orbit to `external_dynamics_params_est`
Online adaptation is performed if the frequency that adaptation is performed `adapt_freq` and the size of the history buffer `history` are provided. 
```bash
traj_adapt, cntrl_adapt = system_coarse.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=orbit_xyz, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps, noise_std=noise_std_dyn, adapt_freq=adapt_freq_qp, history=history_qp)
```

### Plotting
The trajectories for all states can be plotted using
```bash
system.plot_traj(trajectory, controls)
```

Multiple trajectories can be plotted for comparison, and you can provide additional arguments to plot the target state as a dotted line and include a legend.
```bash
system.plot_traj(jnp.array([traj_true, traj_full, traj_low, traj_adapt]), jnp.array([cntrl_true, cntrl_full, cntrl_low, cntrl_adapt]), target_state=target_state, legend=["Full-order", "Low-order", "Low-order w/ Adaptation"])
```

You can also plot the costs over time using 
```bash
system.plot_costs(jnp.array([traj_true, traj_full, traj_low, traj_adapt]), target_state,legend=["True","Full", "Dipole", "Adapt"])
```

If you want to visualize statistics for a batch of trajectories, you can use 
```bash
system.plot_costs(traj_batch, target_states, plot_stats=True)
```
and 
```bash
system.plot_violin_and_bar(trajs_batch, target_states, angle_threshold=angle_threshold, omega_threshold=omega_threshold, angle_stability_tol=angle_tol, omega_stability_tol=omega_tol, tail_length=tail_length, verbose=True)
```
