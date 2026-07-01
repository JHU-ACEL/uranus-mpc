import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # JAX pre-allocates memory when it is imported, this stops that so memory is only used as needed
    import os
    # Prevent memory pre-allocation for flexible memory management
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9' # Use 70% of GPU memory default is '.75'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


    # Import packages
    import numpy as np

    import time

    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import jax.random as jrandom
    import jax.extend.backend as jeb

    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    import pdb
    import gc
    import pickle
    import pandas as pd

    import optax

    from utils.propagate import TrajectoryGenerator, sample_initial_states
    from utils.coord_transforms import coord
    from utils.plotting import save_data_to_pd_df

    from dynamics.spacecraft_dynamics import SpacecraftDynamics
    from dynamics.orbit_dynamics import OrbitDynamics
    from dynamics.planetary_params import Earth, Uranus
    from dynamics.magnetic_field import MagneticFieldModel

    from utils.learning import Trainer, load_model, save_model

    from controllers.feedback_controller import FeedbackController
    from controllers.qp_controller import QPController

    # Jax configuration
    jax.config.update("jax_default_matmul_precision", "highest") # required for vmap to work properly, weird bug
    jax.config.update("jax_enable_x64", True) # enables 64-bit precision

    # Seaborn plotting style
    sns.set_theme(context='paper', style='whitegrid', font='serif', font_scale=1.5)

    jax.devices() # verify GPU is available
    return (
        Earth,
        FeedbackController,
        OrbitDynamics,
        QPController,
        SpacecraftDynamics,
        TrajectoryGenerator,
        Uranus,
        coord,
        jnp,
        jrandom,
        load_model,
        mo,
        pd,
        sample_initial_states,
        save_data_to_pd_df,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialize simulation environment, dynamics, and controllers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set base parameters, and select learned magnetic field models to use (generated in ```B_field_Learning.py```)
    """)
    return


@app.cell
def _(Earth, Uranus, load_model):
    # Define parameters that must be consistent for orbit and spacecraft attitude dynamics
    dt = 0.1
    planet = Earth # Earth, Uranus

    # Load learned magnetic field models
    if planet is Earth:
        model_s, _ = load_model(filename='models/earth_b_4d.eqx') 
        model_coarse, _ = load_model(filename='models/earth_b_4d_coarse.eqx')
    elif planet is Uranus:
        model_s, _ = load_model(filename='models/uranus_b_4d.eqx')
        model_coarse, _ = load_model(filename='models/uranus_b_4d_coarse.eqx')
    return dt, model_coarse, model_s, planet


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate orbital trajectories and magnetic field data
    """)
    return


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
    N_orbit = 5000
    orbit_dynamics = OrbitDynamics(Planet=planet)
    orbit_system = TrajectoryGenerator(dynamics=orbit_dynamics, dt=dt) # set noise_std=0 for deterministic trajectory
    key_orbit = jrandom.key(1234)
    r_planet = orbit_dynamics.planet.radius

    # Generate random initial states
    orbit_init_state_specs = [
                {'dist': 'uniform', 'min': r_planet+200, 'max': r_planet+400}, # a (semimajor axis)
                {'dist': 'uniform', 'min': 0.0, 'max': 0.0}, # eccentricity
                {'dist': 'uniform', 'min': 0, 'max': jnp.pi/2}, # inclination
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # right ascension of the ascending node
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # argument of periapsis
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # true anomaly
    ]

    orbit_batch_size = 200
    orbit_states = sample_initial_states(batch_size=orbit_batch_size, key=key_orbit, state_specs=orbit_init_state_specs)
    orbit_init_states = coord.orbital_elements_to_pci(orbit_states,planet=planet)

    # Generate orbits
    start_orbit_batch = time.time()
    orbit_trajs, _ = orbit_system.generate_trajectory_batch(initial_states=orbit_init_states, num_steps=N_orbit, key=key_orbit, batch_size=orbit_batch_size)
    orbit_trajs.block_until_ready()
    print(f"Time: {time.time() - start_orbit_batch:.2f}s")
    #t = jnp.linspace(0, dt*N_orbit, N_orbit+1)
    return N_orbit, key_orbit, orbit_dynamics, orbit_system, orbit_trajs


@app.cell
def _(orbit_system, orbit_trajs):
    # Visualize a few orbits to verify
    orbit_system.plot_3D(orbit_trajs[80:90])
    return


@app.cell
def _(N_orbit, dt, jnp, key_orbit, orbit_dynamics, orbit_trajs):
    # Generate true magnetic field data corresponding to orbits
    chunk_size = 100 # need to chunk otherwise it exhausts memory

    # Set noise and bias to zero because we are using these as our "True" values
    noise_std_mag = 0 #[nT]
    bias = jnp.array([0, 0, 0])

    # In PCI coords
    b_trajs = orbit_dynamics.mag_model.generate_magnetometer_data(orbit_trajs, dt, N_orbit, key_orbit, noise_std_mag, bias, chunk_size)
    return (b_trajs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set dynamics and trajectory generator classes, and conditions for control simulations
    """)
    return


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
    spacecraft_dynamics = SpacecraftDynamics(mag_model=model_s,planet=planet)
    spacecraft_dynamics_coarse = SpacecraftDynamics(mag_model=model_coarse,planet=planet)

    system = TrajectoryGenerator(dynamics=spacecraft_dynamics, dt=dt)
    system_coarse = TrajectoryGenerator(dynamics=spacecraft_dynamics_coarse, dt=dt)

    key = jrandom.key(4567)
    return key, spacecraft_dynamics, system, system_coarse


@app.cell
def _(
    b_batch,
    jnp,
    jrandom,
    key,
    orbit_xyz_batch,
    spacecraft_dynamics,
    target_states,
):
    # Set general conditions for single runs
    key1, key2, key3, init_cntrl_key = jrandom.split(key, 4) # split to try out a couple different noises before running batch

    # Define standard deviation of random disturbance torque in simulations
    noise_std_dyn = 1e-6 # [Nm]

    batch_i = 86 # Select orbit/magnetic field/target state from one of the randomly sampled ones from batch

    # Choose b-field
    b = b_batch[batch_i]

    # Select orbit that corresponds with b field select to pass to learned model in dyn_params_est
    orbit_xyz = orbit_xyz_batch[batch_i]

    state_limits = jnp.array([[-180, 180]]*3 + [[-2,2]]*3)
    control_limits = 0.8*jnp.array([[-1, 1]] * spacecraft_dynamics.num_controls) # limits for dipole cntrl

    init_state = jnp.array([1, 0, 0, 0, 0.0, 0.0, 0.0])
    target_state = target_states[batch_i]

    quat_start = spacecraft_dynamics.params["quat_start"]
    quat_goal = target_state[quat_start:quat_start+4]
    return (
        b,
        control_limits,
        init_cntrl_key,
        init_state,
        key1,
        noise_std_dyn,
        orbit_xyz,
        quat_start,
        state_limits,
        target_state,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define controller parameters and initialize controllers
    """)
    return


@app.cell
def _(
    FeedbackController,
    QPController,
    control_limits,
    jnp,
    spacecraft_dynamics,
    state_limits,
    system,
):
    # Params for QP MPC
    horizon_qp = 200
    replan_freq_qp = 5
    adapt_freq_qp = 100
    history_qp = 200

    num_steps = 1500 # how long to run sim for

    # QP Solver gains
    Q_qp =  jnp.diag(jnp.array([1e7]*3 + [1e-1]*3))
    Qf_qp = Q_qp
    R_qp = 1e1*jnp.eye(spacecraft_dynamics.num_controls)

    # Feedback controller gains
    Q_fb = jnp.diag(jnp.array([1e2]*3 + [1e1]*3))
    Qf_fb = Q_fb
    R_fb = 1e1*jnp.eye(spacecraft_dynamics.num_controls)

    # Initialize controllers
    qp_controller = QPController(system, Q_qp, Qf_qp, R_qp, state_limits, control_limits, horizon_qp)
    fb_controller = FeedbackController(system,Q_fb,Qf_fb,R_fb,control_limits)
    return (
        Q_qp,
        Qf_qp,
        R_qp,
        adapt_freq_qp,
        fb_controller,
        history_qp,
        horizon_qp,
        num_steps,
        qp_controller,
        replan_freq_qp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run open-loop solve to check performance of QP solver, check difference between linearized and nonlinear dynamics
    """)
    return


@app.cell
def _(
    Q_qp,
    Qf_qp,
    R_qp,
    b,
    horizon_qp,
    init_cntrl_key,
    init_state,
    jnp,
    jrandom,
    key,
    key1,
    qp_controller,
    spacecraft_dynamics,
    system,
    target_state,
):
    # Open loop PDIP w dipole input
    # Initial traj and cntrl for open loop runs
    init_traj_true = jnp.tile(init_state,(horizon_qp+1,1))
    init_cntrl_true = 0.001*jrandom.normal(init_cntrl_key, shape=((horizon_qp, spacecraft_dynamics.num_controls)))

    nom_qp_traj, nom_qp_cntrl = qp_controller(init_state, target_state, key, b[:horizon_qp], init_traj_true, init_cntrl_true, Q_qp, Qf_qp, R_qp)
    traj_nl, _ = system.generate_trajectory(initial_state=init_state, target_state=None, key=key1, external_dynamics_params=b[:horizon_qp], control_sequence=nom_qp_cntrl, num_steps=horizon_qp)
    system.plot_traj(jnp.array([nom_qp_traj, traj_nl]), nom_qp_cntrl, target_state)
    _ = system.plot_costs(jnp.array([nom_qp_traj, traj_nl]), target_state,legend=["Linear", "True"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate single closed-loop trajectories
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate single closed-loop trajectories for all cases of magnetic field model fidelity that controller has access to:
    1. True magnetic field model (True)
    2. Learned magnetic field model trained on data from full-fidelity magnetic field model (Full)
    3. Learned magnetic field model trained on data from dipole magnetic field model (Dipole)
    4. Dipole-learned magnetic field model with online adaptations (Adapt)
    """)
    return


@app.cell
def _(
    b,
    fb_controller,
    init_state,
    jnp,
    key1,
    noise_std_dyn,
    num_steps,
    qp_controller,
    replan_freq_qp,
    system,
    target_state,
):
    # Closed loop QP solver - True params
    traj_true, cntrl_true = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key1, external_dynamics_params=b, external_dynamics_params_est=None, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps, noise_std=noise_std_dyn)
    system.plot_traj(traj_true, cntrl_true, target_state)
    traj_true_df  = system.plot_costs(traj_true,target_state)
    print(jnp.array(traj_true_df["Angle Errors"])[-1])
    return cntrl_true, traj_true


@app.cell
def _(
    b,
    fb_controller,
    init_state,
    key,
    noise_std_dyn,
    num_steps,
    orbit_xyz,
    qp_controller,
    replan_freq_qp,
    system,
    target_state,
):
    # Closed loop QP solver - Learned full-order model
    traj_full, cntrl_full = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=orbit_xyz, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps, noise_std=noise_std_dyn)
    return cntrl_full, traj_full


@app.cell
def _(
    b,
    fb_controller,
    init_state,
    key,
    noise_std_dyn,
    num_steps,
    orbit_xyz,
    qp_controller,
    replan_freq_qp,
    system_coarse,
    target_state,
):
    # Closed loop QP solver - Learned dipole model
    traj_low, cntrl_low = system_coarse.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=orbit_xyz, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps, noise_std=noise_std_dyn)
    return cntrl_low, traj_low


@app.cell
def _(
    adapt_freq_qp,
    b,
    fb_controller,
    history_qp,
    init_state,
    key,
    noise_std_dyn,
    num_steps,
    orbit_xyz,
    qp_controller,
    replan_freq_qp,
    system_coarse,
    target_state,
):
    # Closed loop QP solver -Learned dipole model w/ adaptation
    traj_adapt, cntrl_adapt = system_coarse.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, external_dynamics_params=b, external_dynamics_params_est=orbit_xyz, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, num_steps=num_steps, noise_std=noise_std_dyn, adapt_freq=adapt_freq_qp, history=history_qp)
    return cntrl_adapt, traj_adapt


@app.cell
def _(
    cntrl_adapt,
    cntrl_full,
    cntrl_low,
    cntrl_true,
    jnp,
    system,
    target_state,
    traj_adapt,
    traj_full,
    traj_low,
    traj_true,
):
    # Compare all four cases
    system.plot_traj(jnp.array([traj_true, traj_full, traj_low, traj_adapt]), jnp.array([cntrl_true, cntrl_full, cntrl_low, cntrl_adapt]),target_state=target_state, legend=["True","Full", "Dipole", "Adapt"])
    _ = system.plot_costs(jnp.array([traj_true, traj_full, traj_low, traj_adapt]), target_state,legend=["True","Full", "Dipole", "Adapt"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Monte Carlo Simulations
     Verify controller by generating batch of closed-loop trajectories with randomized orbits and target states
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set batch parameters
    """)
    return


@app.cell
def _(b_trajs, jnp, jrandom, key, orbit_trajs, sample_initial_states):
    # Generate random initial states and set batch properties
    init_state_specs = [
                {'shape': (4,), 'dist': 'quaternion'}, # quaternion
                {'shape': (3,), 'dist': 'uniform', 'min': 0.0, 'max': 0.0} # angular velocity
    ]

    target_state_specs = [
                {'shape': (4,), 'dist': 'quaternion'}, # quaternion
                {'shape': (3,), 'dist': 'uniform', 'min': 0.0, 'max': 0.0} # angular velocity
    ]

    batch_size = 100

    new_key, init_state_key, targ_state_key = jrandom.split(key,3)
    init_states = sample_initial_states(batch_size=batch_size, key=init_state_key, state_specs=init_state_specs)
    target_states = sample_initial_states(batch_size=batch_size, key=targ_state_key, state_specs=target_state_specs)

    # Get random b-field, always start at beginning of orbit (orbit start points already randomized)
    b_batch_idx = jnp.arange(b_trajs.shape[0])
    new_key, b_key  = jrandom.split(new_key)
    shuffled_indices = jrandom.permutation(b_key, b_batch_idx)
    batch_indices = shuffled_indices[:batch_size]
    b_batch = b_trajs[batch_indices, :, :]
    orbit_xyz_batch = orbit_trajs[batch_indices, :, :3]

    num_steps_batch = 1500
    return b_batch, batch_size, num_steps_batch, orbit_xyz_batch, target_states


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate batch of trajectories for all (4) cases
    """)
    return


@app.cell
def _(
    b_batch,
    batch_size,
    fb_controller,
    init_state,
    key,
    noise_std_dyn,
    num_steps_batch,
    qp_controller,
    replan_freq_qp,
    system,
    target_states,
    time,
):
    # QP Solver Batch - true params
    start_batch_qp = time.time()
    trajs_true, cntrls_true= system.generate_trajectory_batch(initial_states=init_state, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=None, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, chunk_size=None, num_steps=num_steps_batch, noise_std=noise_std_dyn)
    trajs_true.block_until_ready()
    print(f"Time: {time.time() - start_batch_qp:.2f}s")
    return (trajs_true,)


@app.cell
def _(
    b_batch,
    batch_size,
    fb_controller,
    init_state,
    key,
    noise_std_dyn,
    num_steps_batch,
    orbit_xyz_batch,
    qp_controller,
    replan_freq_qp,
    system,
    target_states,
):
    # QP Solver Batch - full order learned model
    trajs_full, cntrls_full = system.generate_trajectory_batch(initial_states=init_state, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=orbit_xyz_batch, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, chunk_size=50, num_steps=num_steps_batch, noise_std=noise_std_dyn)
    return (trajs_full,)


@app.cell
def _(
    b_batch,
    batch_size,
    fb_controller,
    init_state,
    key,
    noise_std_dyn,
    num_steps_batch,
    orbit_xyz_batch,
    qp_controller,
    replan_freq_qp,
    system_coarse,
    target_states,
):
    # QP Solver Batch - low-order model
    trajs_qp_coarse, cntrls_qp_coarse = system_coarse.generate_trajectory_batch(initial_states=init_state, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=orbit_xyz_batch, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, chunk_size=100, num_steps=num_steps_batch, noise_std=noise_std_dyn)
    return (trajs_qp_coarse,)


@app.cell
def _(
    adapt_freq_qp,
    b_batch,
    batch_size,
    fb_controller,
    init_state,
    key,
    noise_std_dyn,
    num_steps_batch,
    orbit_xyz_batch,
    qp_controller,
    replan_freq_qp,
    system_coarse,
    target_states,
):
    # QP Solver Batch - adaptation
    trajs_qp_adapt, cntrls_qp_adapt = system_coarse.generate_trajectory_batch(initial_states=init_state, target_states=target_states, key=key, batch_size=batch_size, external_dynamics_params=b_batch, external_dynamics_params_est=orbit_xyz_batch, high_level_controller=qp_controller, low_level_controller=fb_controller, replan_freq=replan_freq_qp, chunk_size=50, num_steps=num_steps_batch, noise_std=noise_std_dyn, adapt_freq=adapt_freq_qp)
    return (trajs_qp_adapt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save data to ```pandas``` dataframe using ```pickle``` (make ```data``` folder first)
    """)
    return


@app.cell
def _(
    Earth,
    Uranus,
    dt,
    horizon_qp,
    jnp,
    planet,
    quat_start,
    save_data_to_pd_df,
    target_states,
    trajs_full,
    trajs_qp_adapt,
    trajs_qp_coarse,
    trajs_true,
):
    # Save data to pandas dataframe
    if planet is Earth:
        if horizon_qp == 50:
            filename = 'earth_50.pkl'
        elif horizon_qp == 100:
            filename = 'earth_100.pkl'
        elif horizon_qp == 150:
            filename = 'earth_150.pkl'
        elif horizon_qp == 200:
            filename = 'earth_200.pkl'
        elif horizon_qp == 250:
            filename = 'earth_250.pkl'

    elif planet is Uranus:
        if horizon_qp == 50:
            filename = 'uranus_50.pkl'
        elif horizon_qp == 100:
            filename = 'uranus_100.pkl'
        elif horizon_qp == 150:
            filename = 'uranus_150.pkl'
        elif horizon_qp == 200:
            filename = 'uranus_200.pkl'
        elif horizon_qp == 250:
            filename = 'uranus_250.pkl'

    _ = save_data_to_pd_df(jnp.array([trajs_true, trajs_full, trajs_qp_coarse, trajs_qp_adapt]), target_states=target_states, labels=["True", "Full", "Dipole", "Adapt"],dt=dt,quat_start=quat_start,filename=filename)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting
    If you make a ```figures``` folder and provide a filename to the plotting functions, the plots are saved
    """)
    return


@app.cell
def _():
    # Thresholds for success
    angle_threshold = 15 # phi = 2*arccos(q.T @ q_g)*180/pi
    omega_threshold = 5 # (deg/s)

    # Tolerances for stability
    angle_tol = 10
    omega_tol = 5
    tail_length = 100
    # time_hist_max = 250
    angle_hist_max = 30
    omega_hist_max = 15
    return (
        angle_hist_max,
        angle_threshold,
        angle_tol,
        omega_hist_max,
        omega_threshold,
        omega_tol,
        tail_length,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can plot using the generated trajectories directly as shown below
    """)
    return


@app.cell
def _(system, target_states, trajs_true):
    _ = system.plot_costs(trajs_true, target_states, plot_stats=True)
    return


@app.cell
def _(
    angle_threshold,
    angle_tol,
    omega_threshold,
    omega_tol,
    system,
    tail_length,
    target_states,
    trajs_true,
):
    system.plot_violin_and_bar(trajs_true, target_states, angle_threshold=angle_threshold, omega_threshold=omega_threshold,angle_stability_tol=angle_tol, omega_stability_tol=omega_tol, tail_length=tail_length, verbose=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or you can plot from the saved dataframe.
    """)
    return


@app.cell
def _(
    angle_threshold,
    batch_size,
    num_steps_batch,
    omega_threshold,
    pd,
    system,
):
    # Load dataframes
    earth_df_200 = pd.read_pickle('data/earth_200.pkl')
    uranus_df_200 = pd.read_pickle('data/uranus_200.pkl')

    system.plot_costs(df=earth_df_200,batch_size=batch_size, N=num_steps_batch+1, angle_threshold=angle_threshold, omega_threshold=omega_threshold, plot_stats=True)#,filename='earth_mrp')
    return earth_df_200, uranus_df_200


@app.cell
def _(
    angle_hist_max,
    angle_threshold,
    angle_tol,
    batch_size,
    earth_df_200,
    num_steps_batch,
    omega_hist_max,
    omega_threshold,
    omega_tol,
    system,
    tail_length,
    uranus_df_200,
):
    df_list = [earth_df_200, uranus_df_200]
    labels = ["Earth", "Uranus"]
    filename_fig='planets_mrp.png'

    system.plot_violin_and_bar(df_list=df_list,batch_size=batch_size,N=num_steps_batch+1,angle_threshold=angle_threshold,omega_threshold=omega_threshold,angle_stability_tol=angle_tol, omega_stability_tol=omega_tol, tail_length=tail_length, angle_hist_max=angle_hist_max, omega_hist_max=omega_hist_max, verbose=True, dataset_labels=labels)#,filename=filename_fig)
    return


if __name__ == "__main__":
    app.run()
