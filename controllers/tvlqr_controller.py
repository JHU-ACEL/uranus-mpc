import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from typing import Callable, Iterable, Tuple, Optional, Dict, Any

from .base_controller import Controller
from dynamics.base_dynamics import Dynamics
from utils.propagate import TrajectoryGenerator

import pdb

class TVLQRController(Controller):
  propagator: TrajectoryGenerator
  dynamics: Dynamics
  
  # Parameters
  Q: jax.Array
  Qf: jax.Array
  R: jax.Array
  control_limits: jax.Array

  # Useful for working with dynamics
  quat_start: int = eqx.field(static=True)
  nx: int = eqx.field(static=True)
  G: Callable = eqx.field(static=True) # transition matrix from quaternion to 3 param representation

  def __init__(self, propagator, Q, Qf, R, control_limits):
    self.propagator = propagator
    self.dynamics = self.propagator.dynamics
    self.Q = Q
    self.Qf = Qf.astype('float32')
    self.R = R
    self.control_limits = control_limits

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
    
  # Linearize about point
  def _linearize_and_discretize(self,x,u,t, ext_param):
      A = jax.jacfwd(self.dynamics.state_dot,argnums=0)(x,u,t,ext_param)
      B = jax.jacfwd(self.dynamics.state_dot,argnums=1)(x,u,t,ext_param)
      # Do quaternion conversion to 3 param representation
      G  = self.G(x[self.quat_start:self.quat_start+4])
      A = G @ A @ G.T
      B = G @ B

      Ad = jnp.eye(self.nx-1) + A*self.propagator.dt 
      Bd = B*self.propagator.dt

      
      return Ad, Bd

  @eqx.filter_jit
  def __call__(self, x0: jax.Array, u0: jax.Array, x_goal: jax.Array, external_dynamics_params: jax.Array, cntrl_param: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    TV LQR algorithm for tracking a nominal trajectory

    From  "MAGNETORQUER-ONLY ATTITUDE CONTROL OF SMALL SATELLITES USING TRAJECTORY OPTIMIZATION" - Gatherer
    
    Returns:
        nominal state trajectory: (T+1, nx)
        nominal control trajectory: (T, nu)
    """
    A, B = self._linearize_and_discretize(x0, u0, 0.0, external_dynamics_params)
    Q = self.Q
    R = self.R 
    S = cntrl_param

    omega_goal = x_goal[4:]
    omega = x0[4:]
    q_goal = x_goal[self.quat_start: self.quat_start+4]
    q = x0[self.quat_start: self.quat_start + 4]

    # Calculate dx
    omega_error = omega - omega_goal

    # Quaternion error
    e1 = q_goal[0]*q[0] + q_goal[1:].T @ q[1:]
    e24 = q_goal[0]*q[1:] - q[0]*q_goal[1:] - jnp.cross(q_goal[1:],q[1:])

    phi = e24 / (1+ e1)

    dx = jnp.concatenate((omega_error, 2*phi))

    # For when states don't have quaternion (double integrator test)
    #dx = x0 - x_goal

    # Ricatti equation
    eps = 1e-8*jnp.ones(R.shape) # to avoid singular matrix
    K = jnp.linalg.solve(R + B.T @ S @ B + eps, B.T @ S @ A)
    S = Q + K.T @ R @ K + (A - B@K).T @ S @ (A - B@K)

    # Control law
    u = u0 - K @ dx
    
    # Clip controls
    u = jnp.clip(u, self.control_limits[:, 0], self.control_limits[:, 1])

    return u, S