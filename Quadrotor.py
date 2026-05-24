import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np

    import time

    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    import matplotlib.pyplot as plt
    import marimo as mo
    import pdb

    from utils.learning import Trainer

    from utils.propagate import TrajectoryGenerator
    from dynamics.double_integrator_dynamics import DoubleIntegratorDynamics
    from dynamics.spacecraft_dynamics import SpacecraftDynamics
    from dynamics.quadrotor_dynamics import QuadrotorDynamics
    from dynamics.planetary_params import Earth, Uranus, Neptune

    from controllers.mppi_controller import MppiController
    from controllers.cem_controller import CemController

    from diffmpc.problems.optimal_control_problem import OptimalControlProblem
    from diffmpc.solvers.sqp import SQPSolver
    from diffmpc.utils.load_params import (
        load_problem_params,
        load_solver_params,
    )
    return (
        CemController,
        MppiController,
        QuadrotorDynamics,
        Trainer,
        TrajectoryGenerator,
        jax,
        jnp,
        jrandom,
        time,
    )


@app.cell
def _(jax):
    jax.devices()
    return


@app.cell
def _(QuadrotorDynamics, TrajectoryGenerator, jnp, jrandom):
    # ================ Quadrotor dynamics ================== #
    N = 100
    dt = 0.1
    dynamics = QuadrotorDynamics()
    system = TrajectoryGenerator(dynamics=dynamics, dt=dt, num_steps=N, noise_std=0.0) # set noise_std=0 for deterministic trajectory

    key = jrandom.key(1234)

    init_state = jnp.array([0.0, 0.0, 0.0,  # pos (m)
                            0.0, 0.0, 0.0, # vel (m/s)
                            1.0, 0.0, 0.0, 0.0, # quaternion
                            0.0, 0.0, 0.0]) # angular velocity


    hover_thrust = dynamics.dynamics_params["mass"]*9.81
    const_cntrl =  jnp.array([hover_thrust, 0.0, 0.0, 0.0])
    cntrl_sequence = jnp.broadcast_to(const_cntrl, (N, 4))

    key, thrust_key, tau_key = jrandom.split(key, 3)
    thrusts = jrandom.uniform(thrust_key, shape=(N,1), minval=hover_thrust, maxval=hover_thrust+0.2)
    torques = jrandom.uniform(tau_key, shape=(N,3), minval=-0.1, maxval=0.1)
    cntrl_sequence = jnp.hstack([thrusts, torques])

    traj, cntrl = system.generate_trajectory(initial_state=init_state, target_state=None, key=key, control_sequence=cntrl_sequence)

    system.plot_traj(traj, cntrl)

    init_state_specs = [
        {'shape': (3,), 'dist': 'uniform', 'min': -10, 'max': 10},
        {'shape': (3,), 'dist': 'uniform', 'min': -5, 'max': 5},
        {'shape': (4,), 'dist': 'quaternion'},
        {'shape': (3,), 'dist': 'uniform', 'min': -2, 'max': 2}
    ]
    init_states = system.sample_initial_states(batch_size=10, key=key, state_specs=init_state_specs)
    return (
        N,
        cntrl,
        cntrl_sequence,
        dt,
        dynamics,
        hover_thrust,
        init_state,
        init_states,
        key,
        system,
        traj,
    )


@app.cell
def _():
    # =================== diffmpc setup ==================== #
    # problem_params = load_problem_params("quadrotor.yaml")
    # diffmpc_horizon = N - 1  # diffmpc horizon is N+1
    # problem_params["horizon"] = diffmpc_horizon
    # problem_params["mass"] = dynamics.dynamics_params["mass"]
    # problem_params["inertia"] = dynamics.dynamics_params["inertia"]
    # problem_params["initial_state"] = init_state
    # reference_state = problem_params["reference_state_trajectory"][0]
    # reference_control = problem_params["reference_control_trajectory"][0]
    # problem_params["reference_state_trajectory"] = jnp.repeat(
    #     problem_params["reference_state_trajectory"][0:1], diffmpc_horizon + 1, axis=0
    # )
    # problem_params["reference_control_trajectory"] = jnp.repeat(
    #     problem_params["reference_control_trajectory"][0:1], diffmpc_horizon + 1, axis=0
    # )
    # problem_params["final_state"] = reference_state

    # # Solver parameters
    # solver_params = load_solver_params("sqp.yaml")
    # solver_params["num_scp_iteration_max"] = 15
    # solver_params["pcg"]["tol_epsilon"] = 1.0e-12
    # solver_params["linesearch"] = True

    # problem = OptimalControlProblem(dynamics=dynamics, params=problem_params)
    # solver = SQPSolver(program=problem, params=solver_params)
    return


@app.cell
def _(cntrl_sequence, init_states, key, system):
    trajs, cntrls = system.generate_trajectory_batch(initial_states=init_states, target_states=None, key=key, batch_size=10, control_sequence=cntrl_sequence)
    system.plot_3D(trajs)
    return cntrls, trajs


@app.cell
def _(dynamics, hover_thrust, jnp):
    # Define params for MPPI/CEM
    Q = jnp.eye(dynamics.num_states-1)
    Qf = 1e6*jnp.eye(dynamics.num_states-1)
    R = 0.0001*jnp.eye(dynamics.num_controls)
    control_limits = jnp.array([
        [hover_thrust, hover_thrust + 5],
        *([[-0.0, 0.0]] * 3)
    ])

    num_samples = 12000
    num_iter = 15
    horizon = 100
    plan_update_freq = 100 # should be less than horizon

    # Target state:
    target_state = jnp.array([0.0, 0.0, 10.0,  # pos (m)
                            0.0, 0.0, 0.0, # vel (m/s)
                            1.0, 0.0, 0.0, 0.0, # quaternion
                            0.0, 0.0, 0.0]) # angular velocity
    return (
        Q,
        Qf,
        R,
        control_limits,
        horizon,
        num_iter,
        num_samples,
        plan_update_freq,
        target_state,
    )


@app.cell
def _(
    MppiController,
    Q,
    Qf,
    R,
    control_limits,
    horizon,
    init_state,
    jnp,
    key,
    num_iter,
    num_samples,
    plan_update_freq,
    system,
    target_state,
    time,
):
    # MPPI specific param
    kappa = 0.5

    # Init controller
    mppi_controller = MppiController(system,Q,Qf,R,control_limits,num_samples,num_iter,horizon,kappa)

    # Generate controlled trajectory - MPPI
    start_mppi = time.time()
    traj_mppi, cntrl_mppi = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, high_level_controller=mppi_controller,plan_update_freq=plan_update_freq)
    traj_mppi.block_until_ready()
    print(f"Time: {time.time() - start_mppi:.2f}s")

    # Generate MPPI traj where controller has lower order mag model
    start_mppi = time.time()
    traj_mppi_low, cntrl_mppi_low = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, high_level_controller=mppi_controller,plan_update_freq=plan_update_freq)
    traj_mppi_low.block_until_ready()
    print(f"Time: {time.time() - start_mppi:.2f}s")

    system.plot_traj(jnp.array([traj_mppi, traj_mppi_low]),jnp.array([cntrl_mppi,cntrl_mppi_low]))
    return


@app.cell
def _(
    CemController,
    Q,
    Qf,
    R,
    control_limits,
    horizon,
    init_state,
    key,
    num_iter,
    num_samples,
    plan_update_freq,
    system,
    target_state,
    time,
):
    cem_controller = CemController(system,Q,Qf,R,control_limits,num_samples,num_iter,horizon)

    # Generate controlled trajectory - CEM
    start_cem = time.time()
    traj_cem, cntrl_cem = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, high_level_controller=cem_controller,plan_update_freq=plan_update_freq)
    traj_cem.block_until_ready()
    print(f"Time: {time.time() - start_cem:.2f}s")

    system.plot_traj(traj_cem, cntrl_cem)
    return


@app.cell
def _(Trainer, cntrls, dynamics, jrandom, system, trajs):
    nn_inputs, nn_outputs = system.gen_train_data(trajs,cntrls)
    peak_lr = 3e-4
    num_epochs = 100
    trainer = Trainer(dynamics.num_states, dynamics.num_controls, nn_inputs, nn_outputs, key=jrandom.key(1234), n_epochs=num_epochs, lr=peak_lr)

    trainer.setup_network(depth=6, neurons=32)
    return (trainer,)


@app.cell
def _(trainer):
    # Define training params
    trainer.training_params['CHECKPOINT_AFTER'] = 10
    trainer.training_params['SAVEPOINT_AFTER'] = 100
    trainer.training_params['FILENAME'] = 'model.eqx'
    trainer.training_params['RUN_NAME'] = "FFNet-run-quadrotor"

    # Run training loop
    train_losses, val_losses = trainer.train()
    return


@app.cell
def _(
    N,
    TrajectoryGenerator,
    cntrl,
    cntrl_sequence,
    dt,
    init_state,
    jnp,
    key,
    pd_controller,
    system,
    target_state,
    trainer,
    traj,
):
    trainer.load_model(filename='model.eqx') # load model from file

    nn_dynamics = trainer.get_nn_dynamics(dt)
    nn_system = TrajectoryGenerator(dynamics=nn_dynamics, controller=pd_controller, dt=dt, num_steps=N, noise_std=0.0)

    nn_traj, _ = nn_system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, control_sequence=cntrl_sequence, use_control_seq=True)

    trajectories = jnp.array([traj, nn_traj])
    system.plot_traj(trajectory=trajectories,controls=cntrl)
    #system.compare_traj(traj, cntrl_sequence, nn_traj, cntrl_sequence, state_labels=dynamics.names_states, cntrl_labels=dynamics.names_controls)
    rmse = jnp.sqrt(jnp.mean((traj - nn_traj) ** 2, axis=0))
    print(rmse)
    return


if __name__ == "__main__":
    app.run()
