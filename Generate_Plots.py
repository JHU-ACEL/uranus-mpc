import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate plots from saved dataframes
    ## First, data must be generated using the ```Spacecraft_Control.py``` script.
    """)
    return


@app.cell
def _():
    import os
    # Prevent memory pre-allocation for flexible memory management
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9' # Use 70% of GPU memory default is '.75'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    import pandas as pd
    import seaborn as sns
    import marimo as mo

    from utils.propagate import TrajectoryGenerator
    from dynamics.spacecraft_dynamics import SpacecraftDynamics
    from dynamics.planetary_params import Earth, Uranus, Neptune
    from utils.learning import load_model
    from planetmagfields import Planet

    return (
        Earth,
        Planet,
        SpacecraftDynamics,
        TrajectoryGenerator,
        Uranus,
        load_model,
        mo,
        pd,
        sns,
    )


@app.cell
def _(sns):
    sns.set_theme(context='paper', style='whitegrid', font='serif', font_scale=1.7)
    return


@app.cell
def _(Earth, Uranus, load_model):
    dt = 0.1
    batch_size = 100
    num_steps_batch = 1500

    planet = Earth # Earth, Uranus

    # Load learned magnetic field models
    if planet is Earth:
        model_s, _ = load_model(filename='models/earth_b_4d.eqx') 
        model_coarse, _ = load_model(filename='models/earth_b_4d_coarse.eqx')
    elif planet is Uranus:
        model_s, _ = load_model(filename='models/uranus_b_4d.eqx')
        model_coarse, _ = load_model(filename='models/uranus_b_4d_coarse.eqx')
    return batch_size, dt, model_coarse, model_s, num_steps_batch, planet


@app.cell
def _(
    SpacecraftDynamics,
    TrajectoryGenerator,
    dt,
    model_coarse,
    model_s,
    planet,
):
    spacecraft_dynamics = SpacecraftDynamics(mag_model=model_s,planet=planet)
    spacecraft_dynamics_coarse = SpacecraftDynamics(mag_model=model_coarse,planet=planet)

    system = TrajectoryGenerator(dynamics=spacecraft_dynamics, dt=dt)
    return (system,)


@app.cell
def _(pd):
    # Load dataframes
    uranus_df_mrp = pd.read_pickle('data/uranus_mrp.pkl')
    earth_df_mrp = pd.read_pickle('data/earth_mrp.pkl')
    lqr_df = pd.read_pickle('data/lqr.pkl')

    earth_df_50 = pd.read_pickle('data/earth_50.pkl')
    earth_df_100 = pd.read_pickle('data/earth_100.pkl')
    earth_df_150 = pd.read_pickle('data/earth_150.pkl')
    earth_df_200 = pd.read_pickle('data/earth_200.pkl')
    earth_df_250 = pd.read_pickle('data/earth_250.pkl')

    uranus_df_50 = pd.read_pickle('data/uranus_50.pkl')
    uranus_df_100 = pd.read_pickle('data/uranus_100.pkl')
    uranus_df_150 = pd.read_pickle('data/uranus_150.pkl')
    uranus_df_200 = pd.read_pickle('data/uranus_200.pkl')
    uranus_df_250 = pd.read_pickle('data/uranus_250.pkl')

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
        earth_df_100,
        earth_df_150,
        earth_df_200,
        earth_df_250,
        earth_df_50,
        lqr_df,
        omega_hist_max,
        omega_threshold,
        omega_tol,
        tail_length,
        uranus_df_mrp,
    )


@app.cell
def _(
    angle_threshold,
    batch_size,
    lqr_df,
    num_steps_batch,
    omega_threshold,
    system,
):
    # Torque vs Dipole input LQR comparison
    _ = system.plot_costs(df=lqr_df,batch_size=batch_size, N=num_steps_batch+1, angle_threshold=angle_threshold, omega_threshold=omega_threshold, plot_stats=True,time_max=80)
    return


@app.cell
def _(
    angle_threshold,
    batch_size,
    num_steps_batch,
    omega_threshold,
    system,
    uranus_df_mrp,
):
    _ = system.plot_costs(df=uranus_df_mrp,batch_size=batch_size, N=num_steps_batch+1, angle_threshold=angle_threshold, omega_threshold=omega_threshold, plot_stats=False)#,filename='uranus_mrp.png')
    return


@app.cell
def _(
    angle_hist_max,
    angle_threshold,
    angle_tol,
    batch_size,
    earth_df_100,
    earth_df_150,
    earth_df_200,
    earth_df_250,
    earth_df_50,
    num_steps_batch,
    omega_hist_max,
    omega_threshold,
    omega_tol,
    system,
    tail_length,
):
    df_list = [earth_df_50, earth_df_100, earth_df_150,earth_df_200, earth_df_250]
    filename = 'earth_N.png'
    # df_list = [uranus_df_50, uranus_df_100, uranus_df_150, uranus_df_200, uranus_df_250]
    # filename = 'uranus_N.png'
    labels = ["N=50", "N=100", "N=150", "N=200", "N=250"]

    # df_list = [earth_df_200, uranus_df_200]
    # labels = ["Earth", "Uranus"]
    # filename='planets_mrp.png'

    system.plot_violin_and_bar(df_list=df_list,batch_size=batch_size,N=num_steps_batch+1,angle_threshold=angle_threshold,omega_threshold=omega_threshold,angle_stability_tol=angle_tol, omega_stability_tol=omega_tol, tail_length=tail_length, angle_hist_max=angle_hist_max, omega_hist_max=omega_hist_max, verbose=True, dataset_labels=labels)#,filename=filename)
    return


@app.cell
def _(Planet):
    earth = Planet(name='earth')
    earth.plot(r=1,proj='Mollweide')
    return


@app.cell
def _(Planet):
    uranus = Planet(name='uranus')
    uranus.plot(r=1,proj='Mollweide')
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
