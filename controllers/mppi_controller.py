import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from typing import Callable, Iterable, Tuple, Optional, Dict, Any

from .base_controller import Controller
from utils.propagate import TrajectoryGenerator
from dynamics.base_dynamics import Dynamics

class MppiController(Controller):
  propagator: TrajectoryGenerator
  dynamics: Dynamics
  
  # Parameters
  Q: jax.Array
  Qf: jax.Array
  R: jax.Array
  control_limits: jax.Array
  
  # Hyperparameters (Static) (ints floats default to static)
  num_samples: int = eqx.field(static=True)
  num_iter: int = eqx.field(static=True)
  horizon: int = eqx.field(static=True)
  kappa: float

  # Useful for working with dynamics
  quat_start: int = eqx.field(static=True)
  nx: int = eqx.field(static=True)
  G: Callable = eqx.field(static=True) # transition matrix from quaternion to 3 param representation

  def __init__(self, propagator, Q, Qf, R, control_limits, num_samples, num_iter, horizon, kappa):
    self.propagator = propagator
    self.dynamics = self.propagator.dynamics
    self.Q = Q
    self.Qf = Qf
    self.R = R
    self.control_limits = control_limits
    self.num_samples = num_samples
    self.num_iter = num_iter
    self.horizon = horizon
    self.kappa = kappa

    self.nx = self.dynamics.num_states
    quat_start = self.dynamics.params.get("quat_start")
   
    if quat_start is not None:
      G = self.compute_G_map(quat_start)
    else:
      quat_start = 0 # placeholder if states don't contain quaternion
      G = lambda q: jnp.eye(self.nx)

    object.__setattr__(self, "quat_start", quat_start)
    object.__setattr__(self, "G", G)

  def compute_G_map(self, quat_start):
    nx = self.nx

    def G(q):
      q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
      G_quat = jnp.array([
          [-q2,  q1,  q4, -q3],
          [-q3, -q4,  q1,  q2],
          [-q4,  q3, -q2,  q1]
      ])
      G = jnp.eye(nx-1, nx)
      G = G.at[quat_start:quat_start + 3, quat_start:quat_start+4].set(G_quat)
      return G
      
    return G

    
  def compute_cost(self, states: jax.Array, controls: jax.Array, target_state: jax.Array) -> float:
    """Computes the total quadratic cost for a trajectory and control sequence, including terminal cost.
        Inputs:
        states: Sequence of states, shape (T+1, nx)
        controls: Sequence of controls, shape (T, nu)
        Outputs:
        cost: Scalar total cost (float)
    """
    # Costs from # "MAGNETORQUER-ONLY ATTITUDE CONTROL OF SMALL SATELLITES USING TRAJECTORY OPTIMIZATION" - Gatherer
    # Terminal cost
    J_terminal = 0.5 * states[-1, :].T @ self.G(target_state[self.quat_start:self.quat_start+4]).T @ self.Qf @ self.G(target_state[self.quat_start:self.quat_start+4]) @ states[-1, :]

    # Stage cost (vectorized over time) 
    diff_stage = states[:-1, :] - target_state
    
    def stage_step_cost(x_k, u_k):
      return (0.5 * x_k.T @ self.G(target_state[self.quat_start:self.quat_start+4]).T @ self.Q @ self.G(target_state[self.quat_start:self.quat_start+4]) @ x_k + 0.5 * u_k.T @ self.R @ u_k)*self.propagator.dt
        
    J_stage = jax.vmap(stage_step_cost)(states[:-1], controls)
    
    return J_terminal + jnp.sum(J_stage)
    
  # def __call__(self, x0: jax.Array, x_goal: jax.Array, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
  #   """Handle both single (nx,) and batched (B, nx) inputs."""
  #   # Check if batched
  #   is_batched = x0.ndim == 2
    
  #   if is_batched:
  #       batch_size = x0.shape[0]
  #       keys = jrandom.split(key, batch_size)
  #       # vmap the single-input version
  #       return jax.vmap(self._single_call)(x0, x_goal, keys)
  #   else:
  #       return self._single_call(x0, x_goal, key)


  # @eqx.filter_jit # always going to be called within another jitted function
  def __call__(self, x0: jax.Array, x_goal: jax.Array, key: jax.Array, external_dynamics_params: jax.Array, nominal_traj: Optional[jax.Array] = None, nominal_cntrl: Optional[jax.Array] = None) -> Tuple[jax.Array, jax.Array]:
    """
    MPPI algorithm.
    Returns:
        nominal state trajectory: (T+1, nx)
        nominal control trajectory: (T, nu)
    """
    # Get appropriate standard deviations for noise to be added
    control_range = self.control_limits[:, 1] - self.control_limits[:, 0]
    noise_std = jnp.abs(control_range) / 10.0

    # Initialize nominal control sequence to zeros
    u_nominal = jnp.zeros((self.horizon, self.dynamics.num_controls))
    
    # Optimization Loop
    def scan_step(carry, _):
        u_nom, key = carry
        
        # Split key for noise generation
        key, noise_key, traj_key = jrandom.split(key, 3)
        
        # Generate noise: (num_samples, T, nu)
        noise = jrandom.normal(noise_key, shape=(self.num_samples, self.horizon, self.dynamics.num_controls)) * noise_std
        
        # Create perturbed control sequences
        u_candidates = u_nom + noise # u_nom broadcasted across num_samples dimension
        
        # Clip controls
        u_candidates = jnp.clip(u_candidates, self.control_limits[:, 0], self.control_limits[:, 1])

        # Rollout batch of trajectories from different u_candidates
        states_batch, _ = self.propagator._generate_trajectory_batch_seq(
                                initial_states = x0,
                                target_states = x_goal,
                                key = traj_key,
                                batch_size = self.num_samples,
                                noise_std = 0.0,
                                num_steps = self.horizon,
                                external_dynamics_params = external_dynamics_params,
                                control_sequence = u_candidates)
        
        # Compute costs (vmap over the batch dimension)
        costs = jax.vmap(self.compute_cost, in_axes=(0, 0, None))(
            states_batch, u_candidates, x_goal)

        # Compute weights
        min_cost = jnp.min(costs)
        exp_costs = jnp.exp(-1.0 / self.kappa * (costs - min_cost))
        denom = jnp.sum(exp_costs) + 1e-10 
        weights = exp_costs / denom
        
        # Update nominal control: weighted sum of candidates
        # weights: (K,), u_candidates: (K, T, nu) -> sum over K -> (T, nu)
        u_update = jnp.sum(weights[:, None, None] * u_candidates, axis=0)
        
        u_new = jnp.clip(u_update, self.control_limits[:, 0], self.control_limits[:, 1])
        
        return (u_new, key), carry

    # Run optimization
    init_carry = (u_nominal, key)
    (final_u_nominal, final_key), _ = jax.lax.scan(scan_step, init_carry, length=self.num_iter)

    best_states, best_controls = self.propagator._generate_trajectory_seq(
                                initial_state = x0,
                                target_state = x_goal,
                                key = final_key,
                                noise_std = 0.0,
                                num_steps = self.horizon,
                                external_dynamics_params = external_dynamics_params,
                                control_sequence = final_u_nominal)
    

    return best_states, best_controls