import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    # Prevent memory pre-allocation for flexible memory management
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9' # Use 70% of GPU memory default is '.75'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    return


@app.cell
def _():
    import numpy as np

    import time

    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import jax.extend.backend as jeb

    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    import pdb
    import gc

    from utils.propagate import TrajectoryGenerator, sample_initial_states
    from utils.coord_transforms import coord

    from dynamics.spacecraft_dynamics import SpacecraftDynamics
    from dynamics.orbit_dynamics import OrbitDynamics
    from dynamics.planetary_params import Earth, Uranus
    from dynamics.magnetic_field import MagneticFieldModel

    from utils.learning import Trainer, load_model, save_model

    from controllers.cem_controller import CemController
    from controllers.tvlqr_controller import TVLQRController

    return (
        CemController,
        OrbitDynamics,
        SpacecraftDynamics,
        TVLQRController,
        TrajectoryGenerator,
        Uranus,
        coord,
        jax,
        jnp,
        jrandom,
        load_model,
        sample_initial_states,
        sns,
        time,
    )


@app.cell
def _(jax):
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", "highest") # required for vmap to work properly, weird bug
    jax.devices()
    return


@app.cell
def _(sns):
    sns.set_context("talk", font_scale=.8)
    sns.set_style("whitegrid") 
    return


@app.cell
def _(Uranus, load_model):
    # Define parameters that must be consistent for orbit and spacecraft attitude dynamics
    dt = 0.1 
    planet = Uranus # Earth, Uranus

    # Load learned magnetic field models (Learned in B_Field_Learning.py notebook)
    # Earth
    # model_s = load_model(filename='models/earth_b_4d.eqx') 
    # model_coarse = load_model(filename='models/earth_b_4d_coarse.eqx')

    # Uranus
    model_s, _ = load_model(filename='models/uranus_b_4d.eqx') 
    model_coarse, _ = load_model(filename='models/uranus_b_4d_coarse.eqx')
    return dt, model_coarse, model_s, planet


@app.cell
def _(
    OrbitDynamics,
    TrajectoryGenerator,
    coord,
    dt,
    jnp,
    jrandom,
    planet,
    sample_initial_states,
    time,
):
    # ==================== Orbit Dynamics =================== #
    N_orbit = 55000
    orbit_dynamics = OrbitDynamics(Planet=planet)
    orbit_system = TrajectoryGenerator(dynamics=orbit_dynamics, dt=dt, num_steps=N_orbit, noise_std=0.0) # set noise_std=0 for deterministic trajectory
    key_orbit = jrandom.key(1234)
    r_planet = orbit_dynamics.planet.radius

    # Generate random initial states
    orbit_init_state_specs = [
                {'dist': 'uniform', 'min': r_planet+200, 'max': r_planet+400}, # a (semimajor axis)
                {'dist': 'uniform', 'min': 0.0, 'max': 0.0}, # eccentricity
    #            {'dist': 'uniform', 'min': 70*jnp.pi/180, 'max': jnp.pi/2}, # inclination (near polar)
                {'dist': 'uniform', 'min': 0, 'max': jnp.pi/2}, # inclination
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # right ascension of the ascending node
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # argument of periapsis
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # true anomaly
    ]

    orbit_batch_size = 200
    orbit_states = sample_initial_states(batch_size=orbit_batch_size, key=key_orbit, state_specs=orbit_init_state_specs)
    orbit_init_states = coord.orbital_elements_to_pci(orbit_states,planet=planet)

    # Generate multiple trajectories
    start_orbit_batch = time.time()
    orbit_trajs, _ = orbit_system.generate_trajectory_batch(initial_states=orbit_init_states, target_states=None, key=key_orbit, batch_size=orbit_batch_size)
    orbit_trajs.block_until_ready()
    print(f"Time: {time.time() - start_orbit_batch:.2f}s")
    t = jnp.linspace(0, dt*N_orbit, N_orbit+1)
    return N_orbit, key_orbit, orbit_dynamics, orbit_system, orbit_trajs


@app.cell
def _(orbit_system, orbit_trajs):
    orbit_system.plot_3D(orbit_trajs[80:90])
    return


@app.cell
def _(
    N_orbit,
    dt,
    jnp,
    key_orbit,
    orbit_dynamics,
    orbit_system,
    orbit_trajs,
    time,
):
    chunk_size = 100 # need to chunk otherwise it exhausts memory
    noise_std_mag = 0#1e2 #[nT]
    bias = jnp.array([0, 0, 0])

    start_b_batch = time.time()
    b_trajs = orbit_dynamics.mag_model.generate_magnetometer_data(orbit_trajs, dt, N_orbit, key_orbit, noise_std_mag, bias, chunk_size)
    b_trajs.block_until_ready()
    print(f"Time: {time.time() - start_b_batch:.2f}s")

    print(b_trajs.shape)
    orbit_system.plot_traj(b_trajs[:10],labels_states=["Bx [nT]","By [nT]","Bz [nT]"])
    return (b_trajs,)


@app.cell
def _(
    SpacecraftDynamics,
    TrajectoryGenerator,
    dt,
    jrandom,
    model_coarse,
    model_s,
    planet,
):
    # ==================== Spacecraft Dynamics =================== #
    N_spacecraft = 3000
    spacecraft_dynamics = SpacecraftDynamics(mag_model=model_s,planet=planet)
    spacecraft_dynamics_coarse = SpacecraftDynamics(mag_model=model_coarse,planet=planet)

    system = TrajectoryGenerator(dynamics=spacecraft_dynamics, dt=dt, num_steps=N_spacecraft, noise_std=0.0) # set noise_std=0 for deterministic trajectory
    system_coarse = TrajectoryGenerator(dynamics=spacecraft_dynamics_coarse, dt=dt, num_steps=N_spacecraft, noise_std=0.0)

    key = jrandom.key(4567)
    return N_spacecraft, key, spacecraft_dynamics, system, system_coarse


@app.cell
def _(
    N_spacecraft,
    b_trajs,
    dt,
    jnp,
    jrandom,
    key,
    orbit_trajs,
    spacecraft_dynamics,
):
    # General solver parameters
    horizon = 80 
    replan_freq = 60
    adapt_freq = 100

    control_limits = 0.8*jnp.array([[-1, 1]] * spacecraft_dynamics.num_controls)

    # Weight matrices for 1- |q'*q| cost (first component quat weight, next 3 are angular velocity)
    Q =  jnp.diag(jnp.array([1e5,1e4,1e4,1e4]))
    Qf = jnp.diag(jnp.array([1e5,1e2,1e2,1e2]))
    R = 1e-3*jnp.eye(spacecraft_dynamics.num_controls)

    # Dynamics params
    noise_std_dyn = 0.0
    #noise_std_dyn = jnp.array([0.001]*4 + [0.005]*3)

    # Set init and target states
    init_state = jnp.array([0, 0, 0, 1, 0.0, 0.0, 0.0])
    target_state = jnp.array([0, 1, 0, 0, 0.0, 0.0, 0.0])

    # Choose b-field
    start_b_idx = 0
    orbit_i = 5
    b = b_trajs[orbit_i, start_b_idx:start_b_idx + N_spacecraft, :]
    b_slice = b[:horizon]

    # Select orbit that corresponds with b field select to pass to learned model in dyn_params_est
    orbit_xyz = orbit_trajs[orbit_i, start_b_idx:start_b_idx + N_spacecraft, :3]
    xyz_slice = orbit_xyz[:horizon]

    # Generate nominal traj to initialize solvers (ONLY for open loop, done automatically within system.generate_trajectory)
    quat_idx_start = spacecraft_dynamics.params["quat_start"] # index for determining which states are quaternions
    quat_goal = target_state[quat_idx_start:quat_idx_start+4]
    t_horizon = jnp.linspace(0, dt*horizon, horizon + 1).reshape(-1, 1)  # Shape: (horizon+1, 1)
    init_traj = init_state + t_horizon * (target_state - init_state)  # Shape: (horizon+1, 7)
    _, init_cntrl_key = jrandom.split(key)
    init_cntrl = 0.01*jrandom.normal(init_cntrl_key, shape=((horizon, spacecraft_dynamics.num_controls)))
    return (
        Q,
        Qf,
        R,
        b,
        b_slice,
        control_limits,
        horizon,
        init_cntrl,
        init_state,
        init_traj,
        noise_std_dyn,
        orbit_xyz,
        replan_freq,
        target_state,
    )


@app.cell
def _(TVLQRController, control_limits, jnp, spacecraft_dynamics, system):
    # TVLQR controller init
    Q_lqr = jnp.diag(jnp.array([1e5]*3 + [1e8]*3))
    Qf_lqr = jnp.diag(jnp.array([1e1]*3 + [1e8]*3)) 
    R_lqr = 1e-1*jnp.eye(spacecraft_dynamics.num_controls)
    lqr_controller = TVLQRController(system,Q_lqr,Qf_lqr,R_lqr,control_limits)
    return (lqr_controller,)


@app.cell
def _(CemController, Q, Qf, R, control_limits, horizon, system):
    # Define params for CEM
    num_samples = 3000 
    num_iter = 200
    elite_percent = 0.1
    init_std = 0.1

    # Init CEM controller
    cem_controller = CemController(system,Q,Qf,R,control_limits,num_samples,num_iter,horizon,elite_percent,init_std)
    return (cem_controller,)


@app.cell
def _(
    b_slice,
    cem_controller,
    init_cntrl,
    init_state,
    init_traj,
    jnp,
    key,
    system,
    target_state,
):
    ####### Open loop (one CEM output)
    nom_traj, nom_cntrl = cem_controller(init_state, target_state, key, b_slice, init_traj, init_cntrl)
    system.plot_traj(nom_traj, nom_cntrl,target_state=jnp.tile(target_state,(nom_traj.shape[0],1)))
    system.plot_costs(nom_traj,target_state)
    return


@app.cell
def _(
    b,
    cem_controller,
    init_state,
    jnp,
    key,
    lqr_controller,
    noise_std_dyn,
    replan_freq,
    system,
    target_state,
    time,
):
    # Closed loop CEM - True params
    start_cem = time.time()
    traj_true, cntrl_true, nominal_traj, nominal_cntrl = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=None, high_level_controller=cem_controller, low_level_controller=lqr_controller, replan_freq=replan_freq, num_steps=1200, noise_std=noise_std_dyn)
    traj_true.block_until_ready()
    print(f"Time: {time.time() - start_cem:.2f}s")
    target_state_traj = jnp.tile(target_state,(traj_true.shape[0],1))
    system.plot_traj(traj_true,cntrl_true,target_state=target_state_traj)
    system.plot_costs(traj_true,target_state)
    return cntrl_true, target_state_traj, traj_true


@app.cell
def _(
    b,
    cem_controller,
    init_state,
    key,
    lqr_controller,
    noise_std_dyn,
    orbit_xyz,
    replan_freq,
    system,
    target_state,
):
    # Closed-loop CEM - full-order learned model
    traj_full, cntrl_full, _, _ = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=orbit_xyz, high_level_controller=cem_controller, low_level_controller=lqr_controller, replan_freq=replan_freq, num_steps=1200, noise_std=noise_std_dyn)
    system.plot_costs(traj_full, target_state)
    return


@app.cell
def _(
    b,
    cem_controller,
    init_state,
    key,
    lqr_controller,
    noise_std_dyn,
    orbit_xyz,
    replan_freq,
    system_coarse,
    target_state,
):
    # Closed-loop CEM - low-order model
    traj_low, cntrl_low, _,_ = system_coarse.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=orbit_xyz, high_level_controller=cem_controller, low_level_controller=lqr_controller, replan_freq=replan_freq, num_steps=1200, noise_std=noise_std_dyn)
    return cntrl_low, traj_low


@app.cell
def _(
    b,
    cem_controller,
    init_state,
    key,
    lqr_controller,
    noise_std_dyn,
    orbit_xyz,
    replan_freq,
    system_coarse,
    target_state,
):
    # Closed-loop CEM - coarse model with online-adaptation
    traj_adapt, cntrl_adapt, _,_ = system_coarse.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=orbit_xyz, high_level_controller=cem_controller, low_level_controller=lqr_controller, replan_freq=replan_freq, num_steps=1200, noise_std=noise_std_dyn, adapt_freq=100, history=200)
    return cntrl_adapt, traj_adapt


@app.cell
def _(
    cntrl_adapt,
    cntrl_low,
    cntrl_true,
    jnp,
    system,
    target_state,
    target_state_traj,
    traj_adapt,
    traj_low,
    traj_true,
):
    # Compare all three trajs
    system.plot_traj(jnp.array([traj_true, traj_low, traj_adapt]), jnp.array([cntrl_true, cntrl_low, cntrl_adapt]),target_state=target_state_traj, legend=["Full-order", "Low-order", "Low-order w/ Adaptation"])
    system.plot_costs(jnp.array([traj_true, traj_low, traj_adapt]), target_state,legend=["Full-order", "Low-order", "Low-order w/ Adaptation"])
    return


@app.cell
def _(
    N_spacecraft,
    b_trajs,
    jnp,
    jrandom,
    key,
    orbit_trajs,
    sample_initial_states,
):
    # Generate random initial states and set batch properties
    init_state_specs = [
                {'shape': (4,), 'dist': 'quaternion'}, # quaternion
                {'shape': (3,), 'dist': 'uniform', 'min': 0.0, 'max': 0.0} # angular velocity
    ]

    target_state_specs = [
                {'shape': (4,), 'dist': 'quaternion'}, # quaternion
                {'shape': (3,), 'dist': 'uniform', 'min': 0.0, 'max': 0.0} # angular velocity
    ]

    batch_size = 200

    new_key, init_state_key, targ_state_key = jrandom.split(key,3)
    init_states = sample_initial_states(batch_size=batch_size, key=init_state_key, state_specs=init_state_specs)
    target_states = sample_initial_states(batch_size=batch_size, key=targ_state_key, state_specs=target_state_specs)

    # Get random b-field, always start at beginning of orbit (orbit start points already randomized)
    b_batch_idx = jnp.arange(b_trajs.shape[0])
    new_key, b_key  = jrandom.split(new_key)
    shuffled_indices = jrandom.permutation(b_key, b_batch_idx)
    batch_indices = shuffled_indices[:batch_size]
    b_batch = b_trajs[batch_indices, :N_spacecraft, :]
    orbit_xyz_batch = orbit_trajs[batch_indices, :N_spacecraft, :3]
    return b_batch, batch_size, init_states, orbit_xyz_batch, target_states


@app.cell
def _(
    b_batch,
    batch_size,
    cem_controller,
    init_states,
    key,
    lqr_controller,
    noise_std_dyn,
    replan_freq,
    system,
    target_states,
    time,
):
    # Generate multiple trajectories - true params
    start_batch_true = time.time()
    trajs_cem_true, cntrls_cem_true, _, _ = system.generate_trajectory_batch(initial_states=init_states, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=None, high_level_controller=cem_controller, low_level_controller=lqr_controller, replan_freq=replan_freq, chunk_size=25, num_steps=1000, noise_std=noise_std_dyn)
    trajs_cem_true.block_until_ready()
    print(f"Time: {time.time() - start_batch_true:.2f}s")
    return (trajs_cem_true,)


@app.cell
def _(system, target_states, trajs_cem_true):
    system.plot_costs(trajs_cem_true, target_states)
    return


@app.cell
def _(
    b_batch,
    batch_size,
    cem_controller,
    init_states,
    key,
    lqr_controller,
    noise_std_dyn,
    orbit_xyz_batch,
    replan_freq,
    system,
    target_states,
):
    # Generate multiple trajectories - full order learned model
    trajs_cem_full, cntrls_cem_full, _, _ = system.generate_trajectory_batch(initial_states=init_states, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=orbit_xyz_batch, high_level_controller=cem_controller, low_level_controller=lqr_controller, replan_freq=replan_freq, chunk_size=25, num_steps=1000, noise_std=noise_std_dyn)
    return


@app.cell
def _(
    b_batch,
    batch_size,
    cem_controller,
    init_states,
    key,
    lqr_controller,
    noise_std_dyn,
    orbit_xyz_batch,
    replan_freq,
    system_coarse,
    target_states,
):
    # Generate batch of trajectories - low-order model
    trajs_cem_coarse, cntrls_cem_coarse, _, _ = system_coarse.generate_trajectory_batch(initial_states=init_states, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=orbit_xyz_batch, high_level_controller=cem_controller, low_level_controller=lqr_controller, replan_freq=replan_freq, chunk_size=25, num_steps=1000, noise_std=noise_std_dyn)
    return (trajs_cem_coarse,)


@app.cell
def _(system, target_states, trajs_cem_coarse):
    system.plot_costs(trajs_cem_coarse, target_states)
    return


@app.cell
def _(
    b_batch,
    batch_size,
    cem_controller,
    init_states,
    key,
    lqr_controller,
    noise_std_dyn,
    orbit_xyz_batch,
    replan_freq,
    system_coarse,
    target_states,
):
    # Generate multiple trajectories with adaptation
    trajs_cem_adapt, cntrls_cem_adapt, _, _ = system_coarse.generate_trajectory_batch(initial_states=init_states, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=orbit_xyz_batch, high_level_controller=cem_controller, low_level_controller=lqr_controller, replan_freq=replan_freq, chunk_size=25, num_steps=1000, noise_std=noise_std_dyn, adapt_freq=100)
    return (trajs_cem_adapt,)


@app.cell
def _(system, target_states, trajs_cem_adapt):
    system.plot_costs(trajs_cem_adapt, target_states)
    return


@app.cell
def _(
    jnp,
    system,
    target_states,
    trajs_cem_adapt,
    trajs_cem_coarse,
    trajs_cem_true,
):
    quat_threshold = 0.1 
    omega_threshold = 5 # (deg/s)
    bins = 150
    time_hist_max = 120
    system.plot_hist(jnp.array([trajs_cem_true, trajs_cem_coarse, trajs_cem_adapt]),target_states, quat_threshold, omega_threshold,bins,time_hist_max=time_hist_max, legend=['Full-Order', 'Low-Order', 'Low-Order w/ Adaptation'])
    return


if __name__ == "__main__":
    app.run()
