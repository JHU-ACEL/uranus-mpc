# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.2",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train magnetic field models for use in ```Spacecraft_Control.py```
    Requires ```wandb```
    """)
    return


@app.cell
def _():
    import os
    # Prevent memory pre-allocation for flexible memory management
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8' # Use 80% of GPU memory default is '.75'

    import numpy as np
    import time

    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import jax.extend.backend as jeb
    import jax.tree_util as jtu

    import equinox as eqx
    import optax

    import matplotlib.pyplot as plt
    import marimo as mo
    import pdb
    import gc

    from utils.learning import Trainer, load_model, save_model, adapt_mag_model
    from utils.propagate import TrajectoryGenerator, sample_initial_states
    from utils.coord_transforms import coord

    from dynamics.spacecraft_dynamics import SpacecraftDynamics
    from dynamics.orbit_dynamics import OrbitDynamics
    from dynamics.planetary_params import Earth, Uranus, Neptune, Earth_low_order, Uranus_low_order
    from dynamics.magnetic_field import MagneticFieldModel

    return (
        Earth,
        Earth_low_order,
        MagneticFieldModel,
        OrbitDynamics,
        Trainer,
        TrajectoryGenerator,
        Uranus,
        Uranus_low_order,
        adapt_mag_model,
        coord,
        gc,
        jax,
        jeb,
        jnp,
        jrandom,
        jtu,
        load_model,
        mo,
        sample_initial_states,
        save_model,
    )


@app.cell
def _(jax):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.devices()
    return


@app.cell
def _(gc, jax, jeb):
    # Monitor GPU memory
    device = jax.devices("gpu")[0]
    memory_stats = device.memory_stats()
    print(f"Available memory: {memory_stats}")

    # Clean up when needed
    jeb.clear_backends()     # Clear GPU memory
    jax.clear_caches()
    gc.collect()  # Python garbage collection
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate magnetic field data
    """)
    return


@app.cell
def _(Earth, Earth_low_order, Uranus, Uranus_low_order):
    # Define parameters that must be consistent for orbit and spacecraft attitude dynamics
    dt = 0.1
    planet = Uranus
    learn_coarse = False

    if planet is Earth:
        planet_coarse = Earth_low_order
    elif planet is Uranus:
        planet_coarse = Uranus_low_order
    return dt, learn_coarse, planet, planet_coarse


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
):
    # ==================== Orbit Dynamics =================== #
    orbit_dynamics = OrbitDynamics(Planet=planet)
    orbit_system = TrajectoryGenerator(dynamics=orbit_dynamics, dt=dt)

    key_orbit = jrandom.key(1234)
    r_planet = planet.radius

    orbit_batch_size = 20000

    # Generate random initial states
    orbit_init_state_specs = [
                {'dist': 'uniform', 'min': r_planet + 200, 'max': r_planet + 400}, # a (semimajor axis)
                {'dist': 'uniform', 'min': 0.0, 'max': 0.2}, # eccentricity
                {'dist': 'uniform', 'min': 0, 'max': jnp.pi}, # inclination
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # right ascension of the ascending node
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # argument of periapsis
                {'dist': 'uniform', 'min': 0.0, 'max': 2 * jnp.pi}, # true anomaly
    ]

    # Uniform normal distr of states in orbital parameters
    orbit_init_states_k = sample_initial_states(batch_size=orbit_batch_size, key=key_orbit, state_specs=orbit_init_state_specs)

    # Uniform normal distr of states in PCPF
    orbit_init_states = coord.orbital_elements_to_pci(orbit_init_states_k,planet=planet)

    # Uniform normal distr of states in 4D spherical coords
    orbit_init_states_s = coord.cartesian_to_spherical_4D(orbit_init_states[:,:3]) 
    return key_orbit, orbit_init_states, orbit_init_states_s, orbit_system


@app.cell
def _(jnp, orbit_init_states, orbit_system):
    orbit_system.plot_3D(orbit_init_states[:100,jnp.newaxis,:])
    return


@app.cell
def _(MagneticFieldModel, planet, planet_coarse):
    mag_model = MagneticFieldModel(planet)
    mag_model_coarse = MagneticFieldModel(planet_coarse)
    return mag_model, mag_model_coarse


@app.cell
def _(mag_model, mag_model_coarse, orbit_init_states):
    b_trajs = mag_model.compute_b_pcpf(r_pcpf=orbit_init_states[:,:3])
    b_trajs_coarse = mag_model_coarse.compute_b_pcpf(r_pcpf=orbit_init_states[:,:3])
    return b_trajs, b_trajs_coarse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train FFNN Models
    """)
    return


@app.cell
def _(
    Trainer,
    b_trajs,
    b_trajs_coarse,
    jrandom,
    learn_coarse,
    orbit_init_states_s,
):
    nn_inputs = orbit_init_states_s # [N_samples, 4] # Use 4D spherical coords as inputs
    if learn_coarse:
        nn_outputs = b_trajs_coarse # Coarse model
    else:
        nn_outputs = b_trajs # [N_samples, 3] # Full order model
    print(f"N_samples: {nn_inputs.shape[0]}")
    print(f"Input dim: {nn_inputs.shape[1]}")
    peak_lr = 3e-3
    num_epochs = 100
    trainer = Trainer(nn_inputs, nn_outputs, key=jrandom.key(12), n_epochs=num_epochs, lr=peak_lr)

    trainer.setup_network(depth=5, neurons=64)
    return (trainer,)


@app.cell
def _(Earth, Uranus, learn_coarse, planet, trainer):
    # Define training params
    trainer.training_params['CHECKPOINT_AFTER'] = 100
    trainer.training_params['SAVEPOINT_AFTER'] = 200
    if planet is Earth:
        if learn_coarse:
            trainer.training_params['FILENAME'] = 'models/earth_b_4d_coarse.eqx'
            trainer.training_params['RUN_NAME'] = "earth-coarse-run1"
        else:
            trainer.training_params['FILENAME'] = 'models/earth_b_4d.eqx'
            trainer.training_params['RUN_NAME'] = "earth-run1"
    elif planet is Uranus:
        if learn_coarse:
            trainer.training_params['FILENAME'] = 'models/uranus_b_4d_coarse.eqx'
            trainer.training_params['RUN_NAME'] = "uranus-coarse-run1"
        else:
            trainer.training_params['FILENAME'] = 'models/uranus_b_4d.eqx'
            trainer.training_params['RUN_NAME'] = "uranus-run1"
    #trainer.training_params['NUM_STEPS'] = 3000

    # Run training loop
    train_losses, val_losses = trainer.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate learned models
    """)
    return


@app.cell
def _(
    coord,
    dt,
    jnp,
    key_orbit,
    mag_model,
    orbit_init_states,
    orbit_system,
    planet,
):
    init_state_idx = 10
    long_horizon = 50000
    t = jnp.linspace(0, dt*long_horizon, long_horizon+1)

    # Generate trajectory in PCI coords
    orbit_traj_pci, _ = orbit_system.generate_trajectory(initial_state=orbit_init_states[init_state_idx], target_state=None, key=key_orbit,num_steps=long_horizon)


    orbit_system.plot_3D(orbit_traj_pci)

    # Convert to PCPF coords and 4D spherical (in PCPF)
    orbit_traj = coord.pci_to_pcpf(orbit_traj_pci[:,:3],t_seconds=t, planet=planet)
    orbit_traj_s = coord.cartesian_to_spherical_4D(orbit_traj)

    # B-field in PCPF coords
    b_traj = mag_model.compute_b_pcpf(orbit_traj)

    # B-field in PCI coords
    b_traj_pci = coord.pcpf_to_pci(b_traj,t) # this is equivalent to above
    return b_traj, orbit_traj_s


@app.cell
def _(Earth, Uranus, jax, load_model, orbit_traj_s, planet):
    # Use 4D spherical coords
    if planet is Earth:
        model_s, _ = load_model(filename='models/earth_b_4d.eqx') # Earth
        model_coarse, hyperparams = load_model(filename='models/earth_b_4d_coarse.eqx')
    elif planet is Uranus:
        model_s, _ = load_model(filename='models/uranus_b_4d.eqx') # Uranus
        model_coarse, hyperparams = load_model(filename='models/uranus_b_4d_coarse.eqx')

    b_test_s = jax.vmap(model_s)(orbit_traj_s) # PCPF coords
    b_test_coarse = jax.vmap(model_coarse)(orbit_traj_s)
    return b_test_coarse, b_test_s, hyperparams, model_coarse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fine-tune model with last-layer update (proof of concept for online adaptation)
    """)
    return


@app.cell
def _(
    adapt_mag_model,
    b_traj,
    hyperparams,
    jax,
    jtu,
    model_coarse,
    orbit_traj_s,
    save_model,
):
    # # # Set up fine-tuning from single-orbit data (https://docs.kidger.site/equinox/examples/frozen_layer/)
    original_model = model_coarse
    model = adapt_mag_model(model_coarse, orbit_traj_s[:50], b_traj[:50])

    print(
        f"Parameters of last layer at initialisation:\n"
        f"{jtu.tree_leaves(original_model.layers[-1])}\n"
    )
    print(
        f"Parameters of last layer at end of training:\n"
        f"{jtu.tree_leaves(model.layers[-1])}\n"
    )
    b_updated = jax.vmap(model)(orbit_traj_s)
    save_model('models/uranus_b_4d_updated.eqx', hyperparams, model)
    return


@app.cell
def _(jax, load_model, orbit_traj_s):
    model_updated,_j = load_model('models/uranus_b_4d_updated.eqx')
    b_updated2 = jax.vmap(model_updated)(orbit_traj_s)
    return (b_updated2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare magnetic field models
    """)
    return


@app.cell
def _(b_test_coarse, b_test_s, b_traj, b_updated2, jnp, orbit_system):
    rmse_s = jnp.sqrt(jnp.mean((b_traj - b_test_s) ** 2, axis=0))
    print(f"Spherical: {jnp.mean(rmse_s)}")

    rmse_coarse = jnp.sqrt(jnp.mean((b_traj - b_test_coarse) ** 2, axis=0))
    print(f"Coarse: {jnp.mean(rmse_coarse)}")

    rmse_adj = jnp.sqrt(jnp.mean((b_traj - b_updated2) ** 2, axis=0))
    print(f"Adjusted: {jnp.mean(rmse_adj)}")

    orbit_system.plot_traj(jnp.array([b_traj, b_test_s, b_test_coarse]),labels_states=["Bx [nT]","By [nT]","Bz [nT]"],legend=["True", "Learned Full-order", "Learned Low-order"])
    return


if __name__ == "__main__":
    app.run()
