import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from typing import Callable, Iterable, Tuple, Optional, Dict, Any

from .base_controller import Controller
from dynamics.base_dynamics import Dynamics
from utils.propagate import TrajectoryGenerator

import pdb

class CemController(Controller):
  propagator: TrajectoryGenerator
  dynamics: Dynamics
  
  # Parameters
  Q: jax.Array
  Qf: jax.Array
  R: jax.Array
  w: jax.Array
  wf: jax.Array
  control_limits: jax.Array
  
  # Hyperparameters (Static) (ints floats default to static)
  num_samples: int = eqx.field(static=True)
  num_iter: int = eqx.field(static=True)
  horizon: int = eqx.field(static=True)
  elite_percent: float = eqx.field(static=True)
  init_std: float = eqx.field(static=True)

  # Useful for working with dynamics
  quat_start: int = eqx.field(static=True)
  nx: int = eqx.field(static=True)
  G: Callable = eqx.field(static=True) # transition matrix from quaternion to 3 param representation

  def __init__(self, propagator, Q, Qf, R, control_limits, num_samples, num_iter, horizon, elite_percent=0.1, init_std=0.1):
    self.propagator = propagator
    self.dynamics = self.propagator.dynamics
    self.Q = jnp.diag(jnp.array([0,0,0,0,Q[1,1], Q[2,2], Q[3,3]]))
    self.Qf = jnp.diag(jnp.array([0,0,0,0,Qf[1,1], Qf[2,2], Qf[3,3]]))
    self.R = R
    self.w = Q[0,0]
    self.wf = Qf[0,0]
    self.control_limits = control_limits
    self.num_samples = num_samples
    self.num_iter = num_iter
    self.horizon = horizon
    self.elite_percent = elite_percent
    self.init_std = init_std

    self.nx = self.dynamics.num_states
    quat_start = self.dynamics.params.get("quat_start")
   
    if quat_start is not None:
      G = self.compute_G_map(quat_start)
    else:
      quat_start = 0 # placeholder if states don't contain quaternion
      G = lambda q: jnp.eye(self.nx)

    object.__setattr__(self, "quat_start", quat_start)# if quat_start is not None else 0)
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
      G_mat = jnp.zeros((nx-1, nx))
      # Quaternion error mapping (rows 0:3, cols quat_start:quat_start+4)
      G_mat = G_mat.at[0:3, quat_start:quat_start+4].set(G_quat)
      # Angular velocity passthrough
      G_mat = G_mat.at[3:6, quat_start+4:quat_start+7].set(jnp.eye(3))
      return G_mat
      
    return G
    
  def compute_cost(self, states: jax.Array, controls: jax.Array, target_state: jax.Array) -> float:
    """Computes the total quadratic cost for a trajectory and control sequence, including terminal cost. Unless w and wf are set, they default to zero (i.e. state does not have quaternion)
        Inputs:
        states: Sequence of states, shape (T+1, nx)
        controls: Sequence of controls, shape (T, nu)
        Outputs:
        cost: Scalar total cost (float)
    """
    quat_goal = target_state[self.quat_start:self.quat_start+4]
    quat_cur = states[:, self.quat_start:self.quat_start+4]

    J_terminal = self.wf*(1 - jnp.abs(jnp.dot(quat_goal,quat_cur[-1,:]))) + 0.5 * states[-1, :].T @ self.Qf @ states[-1,:]

    def stage_step_cost(x_k, u_k):
      q_k = x_k[self.quat_start:self.quat_start+4]
      J_quat = self.w*(1 - jnp.abs(jnp.dot(quat_goal,q_k)))
      return (0.5 * x_k.T @ self.Q @ x_k + 0.5 * u_k.T @ self.R @ u_k + J_quat)*self.propagator.dt

    J_stage = jax.vmap(stage_step_cost)(states[:-1], controls)
    return J_terminal + jnp.sum(J_stage)

  @eqx.filter_jit
  def __call__(self, x0: jax.Array, x_goal: jax.Array, key: jax.Array, external_dynamics_params: jax.Array, nominal_traj: jax.Array, nominal_cntrl: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Cross-Entropy (CEM) algorithm.
    Returns:
        nominal state trajectory: (T+1, nx)
        nominal control trajectory: (T, nu)
    """
    n_elites = int(self.num_samples * self.elite_percent)
    nu = self.dynamics.num_controls

    # Initialize mean and covariance (std dev)
    mu = jnp.zeros((self.horizon, nu)) # init to 0
    # mu = nominal_cntrl # warm-start
    sigma = self.init_std * jnp.ones((self.horizon, nu)) # Initial wide exploration
    
    # Optimization Loop
    def scan_step(carry, _):
      curr_mu, curr_sigma, key = carry
      
      key, sample_key, traj_key = jrandom.split(key, 3)
      
      # Sample controls: (K, T, nu)
      # z ~ N(0, 1) -> u = mu + sigma * z
      z = jrandom.normal(sample_key, shape=(self.num_samples, self.horizon, nu))
      u_candidates = curr_mu + curr_sigma * z
      
      # Clip
      u_candidates = jnp.clip(
          u_candidates, 
          self.control_limits[:, 0], 
          self.control_limits[:, 1]
      )

      # Rollout batch of trajectories from different u_candidates
      states_batch, _ = self.propagator._generate_trajectory_batch_seq(
                              initial_states = x0,
                              target_states = x_goal,
                              key = traj_key,
                              batch_size = self.num_samples,
                              noise_std = 0.0, #0.001, #jnp.array([0.001]*4 + [0.001]*3)
        
                              num_steps = self.horizon,
                              external_dynamics_params = external_dynamics_params,
                              control_sequence = u_candidates)
      
      # Compute costs (vmap over the batch dimension)
      costs = jax.vmap(self.compute_cost, in_axes=(0, 0, None))(states_batch, u_candidates, x_goal)
      
      # Select Elites
      _, idx_elites = jax.lax.top_k(-costs, n_elites)
      
      # Gather elite samples: (n_elites, T, nu)
      elites = u_candidates[idx_elites]
      elite_traj = states_batch[idx_elites]
      
      # Update Distribution
      new_mu = jnp.mean(elites, axis=0)
      new_sigma = jnp.std(elites, axis=0)

      # Get mean cost of elites
      cost = jnp.mean(costs[idx_elites])
  
      return (new_mu, new_sigma,  key), carry #  [[ 1 ]] For running in closed loop
      #return (new_mu, new_sigma,  key), (cost, elites, states_batch, elite_traj) # [[ 2 ]] Store all samples for debugging


    # Run optimization
    init_carry = (mu, sigma, key)
    (final_mu, final_sigma, final_key), _ = jax.lax.scan(scan_step, init_carry, length=self.num_iter) # [[ 1 ]]
    #(final_mu, final_sigma, final_key), (costs, elite_cntrls, trajs, elite_trajs) = jax.lax.scan(scan_step, init_carry, length=self.num_iter) # [[ 2 ]]

    final_u_nominal = jnp.clip(final_mu, self.control_limits[:, 0], self.control_limits[:, 1])

    best_states, best_controls = self.propagator._generate_trajectory_seq(
                                initial_state = x0,
                                target_state = x_goal,
                                key = final_key,
                                noise_std = 0.0,
                                num_steps = self.horizon,
                                external_dynamics_params = external_dynamics_params,
                                control_sequence = final_u_nominal)
    
    return best_states, best_controls # [[ 1 ]]
    #return best_states, best_controls, costs, elite_cntrls, trajs, elite_trajs # [[ 2 ]]