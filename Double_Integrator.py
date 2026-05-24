import marimo

__generated_with = "0.23.0"
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

    from controllers.mppi_controller import MppiController
    from controllers.cem_controller import CemController
    from controllers.tvlqr_controller import TVLQRController

    return (
        CemController,
        DoubleIntegratorDynamics,
        MppiController,
        TVLQRController,
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
def _(DoubleIntegratorDynamics, TrajectoryGenerator, jnp, jrandom):
    # ============= Double Integrator ============= #
    N = 200
    dt = 0.1
    dynamics = DoubleIntegratorDynamics()
    system = TrajectoryGenerator(dynamics=dynamics, dt=dt, num_steps=N, noise_std=0.0) # set noise_std=0 for deterministic trajectory

    init_state = jnp.array([10.0, 10.0, 2.0, 3.0])
    target_state = jnp.array([0.0, 0.0, 0.0, 0.0])

    key = jrandom.key(4567)

    key, thrust_key = jrandom.split(key, 2)
    cntrl_sequence = jrandom.uniform(thrust_key, shape=(N,2), minval=-1, maxval=1)

    traj, cntrl = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, control_sequence=cntrl_sequence)

    system.plot_traj(traj, cntrl)
    return (
        N,
        cntrl,
        cntrl_sequence,
        dt,
        dynamics,
        init_state,
        key,
        system,
        target_state,
        traj,
    )


@app.cell
def _(
    CemController,
    MppiController,
    dynamics,
    init_state,
    jnp,
    key,
    system,
    target_state,
    time,
):
    # Define params for MPPI/CEM
    Q = jnp.eye(dynamics.num_states)
    Qf = jnp.eye(dynamics.num_states)
    R = 0.1*jnp.eye(dynamics.num_controls)
    control_limits = jnp.array([[-1., 1.]]*dynamics.num_controls)
    num_samples = 1000
    num_iter = 6
    horizon = 30
    kappa = 0.2
    plan_update_freq = 10

    # Init controller
    mppi_controller = MppiController(system,Q,Qf,R,control_limits,num_samples,num_iter,horizon,kappa)

    cem_controller = CemController(system,Q,Qf,R,control_limits,num_samples,num_iter,horizon)

    # Generate controlled trajectory - MPPI
    start_mppi = time.time()
    traj_mppi, cntrl_mppi, nominal_traj_mppi, nominal_cntrl_mppi = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, high_level_controller=mppi_controller, plan_update_freq=plan_update_freq)
    traj_mppi.block_until_ready()
    print(f"Time: {time.time() - start_mppi:.2f}s")

    # Generate controlled trajectory - CEM
    traj_cem, cntrl_cem, nominal_traj, nominal_cntrl = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, high_level_controller=cem_controller,plan_update_freq=plan_update_freq)

    system.plot_traj(jnp.array([traj_mppi, traj_cem]), jnp.array([cntrl_mppi, cntrl_cem]))
    return Q, Qf, R, control_limits, plan_update_freq


@app.cell
def _(
    Q,
    Qf,
    R,
    TVLQRController,
    control_limits,
    init_state,
    key,
    plan_update_freq,
    system,
    target_state,
):
    tvlqr_controller = TVLQRController(system,Q,Qf,R,control_limits)
    traj_lqr, cntrl_lqr = system.generate_trajectory(initial_state=init_state, target_state=target_state, key=key, low_level_controller=tvlqr_controller,plan_update_freq=plan_update_freq)
    system.plot_traj(traj_lqr, cntrl_lqr)
    return (tvlqr_controller,)


@app.cell
def _(key, system, target_state, time, tvlqr_controller):
    # Generate multiple init states
    batch_size = 4000
    init_state_specs = [
        {'shape': (2,), 'dist': 'uniform', 'min': -10, 'max': 10},
        {'shape': (2,), 'dist': 'uniform', 'min': -5, 'max': 5}
    ]
    init_states = system.sample_initial_states(batch_size=batch_size, key=key, state_specs=init_state_specs)

    # Generate batch of trajectories
    start = time.time()
    trajs, cntrls = system.generate_trajectory_batch(initial_states=init_states, target_states=target_state, key=key, batch_size=batch_size, low_level_controller=tvlqr_controller, plan_update_freq=10)#, control_sequence=cntrl_sequence)
    trajs.block_until_ready()
    print(f"Time: {time.time() - start:.2f}s")

    system.plot_xy_trajectories(trajs, "XY Trajectories")
    return cntrls, trajs


@app.cell
def _(Trainer, cntrls, dynamics, jrandom, system, trajs):
    nn_inputs, nn_outputs = system.gen_train_data(trajs,cntrls)
    peak_lr = 3e-4
    num_epochs = 50
    trainer = Trainer(dynamics.num_states, dynamics.num_controls, nn_inputs, nn_outputs, key=jrandom.key(1234), n_epochs=num_epochs, lr=peak_lr)

    trainer.setup_network(depth=2, neurons=16)
    return (trainer,)


@app.cell
def _(trainer):
    # Define training params
    trainer.training_params['CHECKPOINT_AFTER'] = 10
    trainer.training_params['SAVEPOINT_AFTER'] = 100
    trainer.training_params['FILENAME'] = 'model.eqx'
    trainer.training_params['RUN_NAME'] = "FFNet-run-double-integrator"

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
