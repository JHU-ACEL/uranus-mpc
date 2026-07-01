import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import jax.random as jrandom
import equinox as eqx
from typing import Callable, Iterable, Tuple, Optional, Dict, Any
import pdb

import moreau
from moreau.jax import Solver

from .base_controller import Controller
from dynamics.base_dynamics import Dynamics
from utils.propagate import TrajectoryGenerator

class SQPController(Controller):
  propagator: TrajectoryGenerator
  dynamics: Dynamics
  
  # Parameters
  Q: jax.Array
  Qf: jax.Array
  R: jax.Array
  horizon: int
  P_indptr: jax.Array
  P_indices: jax.Array

  solver: Solver
  state_limits: jax.Array
  control_limits: jax.Array

  nx: int
  nu: int
  sqp_iter: int

  def __init__(self, propagator, Q, Qf, R, state_limits, control_limits, horizon, sqp_iter=5):
    object.__setattr__(self, "propagator", propagator)
    object.__setattr__(self, "dynamics", self.propagator.dynamics)
    object.__setattr__(self, "Q", (Q + Q.T)/2)
    object.__setattr__(self, "Qf", (Qf + Qf.T)/2)
    object.__setattr__(self, "R", (R+R.T)/2)
    object.__setattr__(self, "horizon", horizon)
    object.__setattr__(self, "nu", self.dynamics.num_controls)
    object.__setattr__(self, "state_limits", state_limits)
    object.__setattr__(self, "control_limits", control_limits)
    object.__setattr__(self, "sqp_iter", sqp_iter)
    
    N = self.horizon
    nx = self.dynamics.num_states
    nu = self.nu

    if not self.dynamics.E(jnp.ones(nx)).shape[0] == self.dynamics.E(jnp.ones(nx)).shape[1]:
      nx = self.dynamics.num_states - 1

    object.__setattr__(self, "nx", nx)

    P_indptr, P_indices = self.get_P_csr_idx()
    object.__setattr__(self, "P_indptr", P_indptr)
    object.__setattr__(self, "P_indices", P_indices)
    A_indptr, A_indices = self.get_A_csr_idx()

    # Set up moreau solver
    n_eq = N * nx + nx          # From A_eq
    n_ineq = 2 * (N + 1) * nx + 2 * N * nu  # From G_ineq
    cones = moreau.Cones(
        num_zero_cones=n_eq, 
        num_nonneg_cones=n_ineq
    )
    solver = Solver(
        n=(N + 1) * nx + N * nu, 
        m=n_eq + n_ineq, 
        P_row_offsets=P_indptr,
        P_col_indices=P_indices,
        A_row_offsets=A_indptr,
        A_col_indices=A_indices,
        cones=cones,
    )
    object.__setattr__(self, "solver", solver)
    
  def get_P_csr_idx(self):
    N = self.horizon
    nx = self.nx
    nu = self.nu
    
    # 2. Construct P_col_indices and P_row_offsets
    # We will determine the structure of a single Q, R, and Qf block first
    
    # Local column patterns for individual blocks
    Q_col_local = jnp.tile(jnp.arange(nx), nx)
    R_col_local = jnp.tile(jnp.arange(nu), nu)
    Qf_col_local = jnp.tile(jnp.arange(nx), nx)
    
    # Local row count tracking per individual block
    Q_nnz_per_row = jnp.full(nx, nx)
    R_nnz_per_row = jnp.full(nu, nu)
    Qf_nnz_per_row = jnp.full(nx, nx)

    def scan_body(current_idx, t):
        # Calculate shifts for this specific horizon step
        q_cols = Q_col_local + current_idx
        current_idx += nx
        
        r_cols = R_col_local + current_idx
        current_idx += nu
        
        # Package the iteration's outputs
        # We concatenate Q and R for this specific step to preserve interleaving
        step_cols = jnp.concatenate([q_cols, r_cols])
        step_nnz  = jnp.concatenate([Q_nnz_per_row, R_nnz_per_row])
        
        return current_idx, (step_cols, step_nnz)

    # --- 2. Run the loop over N horizons ---
    # scan will loop N times, carrying 'current_idx' and stacking the outputs
    init_idx = 0
    dummy_xs = jnp.arange(N)
    final_idx, (stacked_cols, stacked_nnz) = jax.lax.scan(scan_body, init_idx, dummy_xs)
    
    # --- 3. Process the final Qf block ---
    qf_cols = Qf_col_local + final_idx
    
    # --- 4. Flatten the looped results and append the final Qf block ---
    # stacked_cols has shape (N, len(Q) + len(R)). .ravel() flattens it strictly in order.
    P_col_indices = jnp.concatenate([stacked_cols.ravel(), qf_cols])
    total_nnz_per_row = jnp.concatenate([stacked_nnz.ravel(), Qf_nnz_per_row])
    
    # P_row_offsets is the cumulative sum of non-zeros per row, starting at 0
    P_row_offsets = jnp.zeros(len(total_nnz_per_row) + 1)#, dtype=jnp.int32)
    P_row_offsets = P_row_offsets.at[1:].set(jnp.cumsum(total_nnz_per_row))

    return P_row_offsets.astype(jnp.int32), P_col_indices.astype(jnp.int32)

  def get_P_csr_data(self, Q, Qf, R):
    N = self.horizon
    nx = self.nx
    nu = self.nu
    
    Q_flat = Q.ravel()
    R_flat = R.ravel()
    Qf_flat = Qf.ravel()
    
    # Repeat the [Q, R] blocks N times, then append Qf
    # We use jnp.tile to repeat the sequences efficiently in JAX
    QR_flat_sequence = jnp.concatenate([Q_flat, R_flat])
    P_data = jnp.concatenate([
        jnp.tile(QR_flat_sequence, N), 
        Qf_flat
    ])
    
    # Get dense version of P
    P_sparse = jsparse.BCSR((P_data, self.P_indices, self.P_indptr), shape=((N+1)*nx + N*nu, (N+1)*nx + N*nu))
    P_dense = P_sparse.todense()

    return P_data, P_dense

  def get_A_csr_idx(self):
    """
    Generates the static CSR row offsets and column indices for the constraint matrix.
    Depends ONLY on the structural dimensions N, nx, and nu.
    """
    N = self.horizon
    nx = self.nx
    nu = self.nu
    # ===========================
    # 1. A_dynamics Columns
    # ===========================
    # Each row contains: nx (for A) + nu (for B) + 1 (for -I) elements
    t_idx = jnp.arange(N)[:, None, None]
    x_t_start = t_idx * (nx + nu)
    u_t_start = x_t_start + nx
    x_tp1_start = jnp.where(t_idx < N - 1, (t_idx + 1) * (nx + nu), N * (nx + nu))

    range_nx = jnp.arange(nx)[None, None, :]
    range_nu = jnp.arange(nu)[None, None, :]

    col_A = jnp.tile(x_t_start + range_nx, (1, nx, 1))  # (N, nx, nx)
    col_B = jnp.tile(u_t_start + range_nu, (1, nx, 1))  # (N, nx, nu)
    
    row_i = jnp.arange(nx)[None, :, None]
    col_I = x_tp1_start + row_i                        # (N, nx, 1)
    
    dyn_col_indices = jnp.concatenate([col_A, col_B, col_I], axis=2).ravel()

    # ===========================
    # 2. A_init Columns
    # ===========================
    init_col_indices = jnp.arange(nx)

    # ===========================
    # 3. A_state_bounds Columns
    # ===========================
    t_sb = jnp.arange(N + 1)[:, None]
    x_idx_sb = jnp.where(t_sb < N, t_sb * (nx + nu), N * (nx + nu))
    state_cols = x_idx_sb + jnp.arange(nx)[None, :]   # (N+1, nx)
    sb_col_indices = jnp.concatenate([state_cols, state_cols], axis=1).ravel()

    # ===========================
    # 4. A_input_bounds Columns
    # ===========================
    t_ib = jnp.arange(N)[:, None]
    u_idx_ib = t_ib * (nx + nu) + nx
    input_cols = u_idx_ib + jnp.arange(nu)[None, :]   # (N, nu)
    ib_col_indices = jnp.concatenate([input_cols, input_cols], axis=1).ravel()

    # ===========================
    # Combine Columns & Compute Row Offsets
    # ===========================
    A_col_indices = jnp.concatenate([
        dyn_col_indices, init_col_indices, sb_col_indices, ib_col_indices
    ])
    
    # Elements per row for each section
    dyn_nnz = jnp.full(N * nx, nx + nu + 1, dtype=jnp.int32)
    init_nnz = jnp.ones(nx, dtype=jnp.int32)
    sb_nnz = jnp.ones(2 * (N + 1) * nx, dtype=jnp.int32)
    ib_nnz = jnp.ones(2 * N * nu, dtype=jnp.int32)

    total_nnz_per_row = jnp.concatenate([dyn_nnz, init_nnz, sb_nnz, ib_nnz])
    
    A_row_offsets = jnp.zeros(len(total_nnz_per_row) + 1, dtype=jnp.int32)
    A_row_offsets = A_row_offsets.at[1:].set(jnp.cumsum(total_nnz_per_row))

    return A_row_offsets.astype(jnp.int32), A_col_indices.astype(jnp.int32)

  def get_A_data(self, A, B):
    """
    Extracts the numerical data vector for matrix A. 
    Maintains structural zeros for compatibility with fixed sparsity structures.
    """
    N = self.horizon
    nx = self.nx
    nu = self.nu
    # 1. Dynamics Data: Link A, B, and a column of -1.0 per row
    I_neg = jnp.full((N, nx, 1), -1.0)
    dyn_data = jnp.concatenate([A, B, I_neg], axis=2).ravel()

    # 2. Initialization Data (Identity diagonal)
    init_data = jnp.ones(nx)

    # 3. State Bounds Data (Alternates -1.0 rows and 1.0 rows per step)
    sb_data_step = jnp.concatenate([-jnp.ones(nx), jnp.ones(nx)])
    sb_data = jnp.tile(sb_data_step, N + 1)

    # 4. Input Bounds Data (Alternates -1.0 rows and 1.0 rows per step)
    ib_data_step = jnp.concatenate([-jnp.ones(nu), jnp.ones(nu)])
    ib_data = jnp.tile(ib_data_step, N)

    # Combine everything into a single flat vector matching the sparsity structure
    A_data = jnp.concatenate([dyn_data, init_data, sb_data, ib_data])
    return A_data
   
  def form_ocp_moreau(self, x0, xg, nom_traj, nom_cntrl, ext_params, Q, Qf, R):
    A, B = self.dynamics.linearize_and_discretize(nom_traj, nom_cntrl, ext_params, self.propagator.dt)

    A_data = self.get_A_data(A, B)
    
    b_eq = jnp.concatenate([
        jnp.zeros(self.horizon * self.nx),  # dynamics
        x0,            # initial condition
    ])

    x_min = self.state_limits[:,0]
    x_max = self.state_limits[:,1]
    u_min = self.control_limits[:,0] - nom_cntrl
    u_max = self.control_limits[:,1] - nom_cntrl

    u_bounds = jnp.stack([-u_min, u_max], axis=1).flatten()

    b_ineq = jnp.concatenate([
        jnp.tile(-x_min, self.horizon + 1),  # -x <= -x_min
        jnp.tile(x_max, self.horizon + 1),   # x <= x_max
        u_bounds,
    ])

    b = jnp.concatenate([b_eq, b_ineq])

    P_data, P_dense = self.get_P_csr_data(Q, Qf, R)

    cntrl_goal = jnp.zeros((self.horizon, self.nu))
    blocks = jnp.hstack((cntrl_goal, xg[1:] ))
    full_vector = jnp.concatenate((xg[0], blocks.ravel()))
    q = -P_dense @ full_vector

    return P_data, A_data, q, b


  @eqx.filter_jit
  def __call__(self, x0: jax.Array, x_goal: jax.Array, key: jax.Array, external_dynamics_params: jax.Array, nominal_traj: jax.Array, nominal_cntrl: jax.Array, Q: Optional[jax.Array] = jnp.eye(7), Qf: Optional[jax.Array] = jnp.eye(7), R: Optional[jax.Array] = jnp.eye(3)) -> Tuple[jax.Array, jax.Array]:
    """
    PDIP solver algorithm for tracking a nominal trajectory

    Returns:
        nominal state trajectory: (T+1, nx)
        nominal control trajectory: (T, nu)
    """
    N = self.horizon
    nx = self.nx
    nu = self.nu
    #quat_start = self.quat_start

    # Get nonlinear trajectory from previous controls to linearize around
    nominal_traj, _ = self.propagator._generate_trajectory_seq(
                                initial_state = x0,
                                target_state = None,
                                key = key,
                                noise_std = 0.0,
                                num_steps = self.horizon,
                                external_dynamics_params = external_dynamics_params,
                                control_sequence = nominal_cntrl)

    # --- SQP Loop Body ---
    def body_fun(i, loop_state):
        nominal_traj, nominal_cntrl, _, _ = loop_state

        # Generate nonlinear trajectory
        nominal_traj, _ = self.propagator._generate_trajectory_seq(
            initial_state = x0,
            target_state = None,
            key = key,
            noise_std = 0.0,
            num_steps = self.horizon,
            external_dynamics_params = external_dynamics_params,
            control_sequence = nominal_cntrl
        )

        # Convert to error coords
        dx0 = self.dynamics.get_error_coords(x0, nominal_traj[0])
        xg = jax.vmap(self.dynamics.get_error_coords, in_axes=(None, 0))(x_goal, nominal_traj)

        # Solve QP
        P_data, A_data, q, b = self.form_ocp_moreau(dx0, xg, nominal_traj, nominal_cntrl, external_dynamics_params, self.Q, self.Qf, self.R)
        solution = self.solver.solve(P_data, A_data, q, b)
        info = self.solver.info
        reshaped = solution.x[nx:].reshape(N, nx + nu)

        # Extract the new control trajectory
        dx = reshaped[:, nu : nu + nx]
        dx = jnp.concatenate([dx0[None,:], dx], axis=0)
    
        state_traj = jax.vmap(self.dynamics.get_true_coords)(dx, nominal_traj)
        du = reshaped[:, :nu]
        alpha = 0.8
        cntrl_traj = du*alpha + nominal_cntrl

        # # Generate nonlinear trajectory
        # state_traj, _ = self.propagator._generate_trajectory_seq(
        #     initial_state = x0,
        #     target_state = None,
        #     key = key,
        #     noise_std = 0.0,
        #     num_steps = self.horizon,
        #     external_dynamics_params = external_dynamics_params,
        #     control_sequence = jnp.nan_to_num(cntrl_traj)
        # )

        # Calculate convergence metric
        diff_cntrl = jnp.max(jnp.abs(cntrl_traj - nominal_cntrl))
        diff_state = jnp.max(jnp.abs(state_traj - nominal_traj))
        max_diff = jnp.maximum(diff_cntrl, diff_state)
        #jax.debug.print("iter: {i}, status: {x}, diff_state: {ds}, diff_cntrl: {dc}",i=i, x=info.status, ds=diff_state, dc=diff_cntrl)

        return (state_traj, cntrl_traj, max_diff, info.status)

    # --- Initialization ---
    # The initial diff is set arbitrarily high (1e6) to guarantee the loop runs at least once
    init_val = (nominal_traj, nominal_cntrl, 1e6, 0.0)

    # --- Run SQP ---
    final_state_traj, final_cntrl_traj, final_diff, final_status = jax.lax.fori_loop(0,self.sqp_iter,body_fun,init_val)

    final_state_traj = jnp.nan_to_num(final_state_traj)
    final_cntrl_traj = jnp.nan_to_num(final_cntrl_traj)

    return final_state_traj, final_cntrl_traj
    