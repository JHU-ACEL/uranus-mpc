import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from typing import Callable, Iterable, Tuple, Optional, Dict, Any # TODO: maybe do jax.typing?

import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
import numpy as np

from .base_controller import Controller
from dynamics.base_dynamics import Dynamics
from utils.propagate import TrajectoryGenerator

import pdb

class SpacecraftAcadosModel(eqx.Module):
  """
  Constructs the AcadosModel used for defining the system dynamics.
  """
  propagator: TrajectoryGenerator
  dynamics: Dynamics
  
  name: str
  mass: float
  inertia: jax.Array
  inertia_inv: jax.Array
  state_limits: jax.Array
  control_limits: jax.Array

  dt: float
  
  def __init__(self, propagator, state_limits, control_limits):
    """
    Constructor for the spacecraft model
    
    Attributes:
      name (str): model name.
      mass (float): robot mass [kg].
      inertia (np.ndarray): 3x3 inertia tensor [kg*m^2].
      omega_max (float): maximum (and defines minimum) angular rotation [rad/s].
      moment_max (float): maximum moment applied along any axis [N*m].
      dt (float): integration time step.
    """
    self.name = 'spacecraft_acados_model'
    self.propagator = propagator
    self.dynamics = propagator.dynamics
    self.dt = propagator.dt
    self.mass = self.dynamics.dynamics_params['mass']
    self.inertia = np.array(self.dynamics.dynamics_params['inertia'])
    self.inertia_inv = np.array(self.dynamics.dynamics_params['inertia_inv'])

    self.state_limits = state_limits
    self.control_limits = control_limits

  @staticmethod
  def S(u):
    """
    Skew symmetric map (lie algebra of SO(3), solves a x b = S(a)b)
    """
    return cs.vertcat(
        cs.horzcat(0, -u[2], u[1]),
        cs.horzcat(u[2], 0, -u[0]),
        cs.horzcat(-u[1], u[0], 0))

  def q_left(self, q):
      """
      Left quaternion product matrix.
      Assumes q is [qs, qx, qy, qz] (scalar first).
      """
      qs = q[0]
      qv = q[1:]
      
      # Top row: [qs, -qv.T]
      row_top = cs.horzcat(qs, -qv.T)
      
      # Bottom block: [qv, qs*I + S(qv)]  (3x3)
      row_bottom = cs.horzcat(qv, qs * cs.MX.eye(3) + self.S(qv))
      
      return cs.vertcat(row_top, row_bottom)

  @staticmethod
  def get_rotation(q):
      """Quaternion to rotation matrix"""
      qw, qx, qy, qz = q[0], q[1], q[2], q[3]
      qw2 = qw * qw
      qx2 = qx * qx
      qy2 = qy * qy
      qz2 = qz * qz
      return cs.vertcat(
          cs.horzcat(qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
          cs.horzcat(2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)),
          cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2))

  @staticmethod
  def q_conj(q):
    """Conjugate quaternion"""
    return cs.vertcat(q[0], -q[1], -q[2], -q[3])
 
  def get_acados_model(self) -> AcadosModel:
    """
      Constructs the AcadosModel used for the 3-DoF attitude
      planning problem for spacecraft. 
      Constructed w help from https://github.com/acados/acados/blob/main/examples/acados_python/getting_started/pendulum_model.py

      Returns:
        model (AcadosModel): model for spacecraft slewing problem.
    """
    # Set up states
    q = cs.MX.sym('q',4,1)
    v = cs.MX.sym('v', 3, 1)
    x = cs.vertcat(q,v)

    # Set up control input (moment)
    M = cs.MX.sym('M', 3, 1)

    # Set up params (b-field and target_state
    b = cs.MX.sym('b',3,1)
    x_g = cs.MX.sym('xg',7,1)
    p = cs.vertcat(b,x_g)

    # Magnetorquer torque: rotate b-field from PCI to body frame
    b_body = self.get_rotation(self.q_conj(q)) @ b
    tau = cs.cross(M, b_body) * 1e-9  # convert nT to T

    # Set up discrete dynamics
    qdot = 0.5*self.q_left(q) @ cs.vertcat(0.0, v)
    J = cs.MX(self.inertia)
    J_inv = cs.MX(self.inertia_inv)

    # Use torque directly (M) or use dipole (tau)
    vdot = J_inv @ (tau - cs.cross(v, J @ v)) # use dipole
    #vdot = J_inv @ (M - cs.cross(v, J @ v)) # use torque directly

    f_disc = cs.vertcat((q + qdot*self.dt)/cs.norm_2(q + qdot*self.dt), v + vdot*self.dt)

    model = AcadosModel()

    model.disc_dyn_expr = f_disc
    model.x = x
    model.u = M
    model.p = p
    model.name = self.name

    return model
    
class AcadosController(eqx.Module):
  model: SpacecraftAcadosModel
  horizon: int

  Q: jax.Array
  Qf: jax.Array
  R: jax.Array

  max_iter: int
  ocp_solver: AcadosOcpSolver = eqx.field(static=True)
  
  def __init__(self, model, Q, Qf, R, horizon, max_iter):
    """
    Attributes:
      model (AcadosModel): acados model.
      ocp_solver (AcadosOcpSolver): acados solver.
      N (int): number of time steps in optimal control horizon.
      x0 (np.ndarray): initial state of system.
      Q_mat (list[float]): diagonal elements of the Q cost term.
      R_mat (list[float]): diagonal elements of the R cost term.
    """
    self.model = model
    self.horizon = horizon
    
    # set cost
    self.Q = Q 
    self.Qf = Qf 
    self.R = R 
    
    self.max_iter = max_iter
    self.ocp_solver = self.setup(self.horizon)

  def setup(self, N_horizon:int) -> AcadosOcpSolver:
    """
    Method to set up the AcadosOcpSolver; this is where the constraints and cost terms
    are explicitly defined for the system using the AcadosModel.

    Parameters:
      x0 (np.ndarray): initial state of system. 
      N_horizon (int): number of time steps in optimal control horizon.

    Returns:
      ocp_solver (AcadosOcpSolver): acados solver.
    """
    # Create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    ocp.model = self.model.get_acados_model()

    # set prediction horizon
    ocp.solver_options.N_horizon = N_horizon
    ocp.solver_options.tf = N_horizon*self.model.dt

    # init param
    ocp.parameter_values =  np.zeros(10) #b-field (3) and target_state (7)
    target_state = ocp.model.p[3:]
    q_g = target_state[:4]
    omega_g = target_state[4:]
    
    # path cost
    ocp.cost.cost_type = 'NONLINEAR_LS'

    q_error = 1 - cs.dot(ocp.model.x[:4], q_g)**2
    omega_error = omega_g - ocp.model.x[4:7]
    
    ocp.model.cost_y_expr = cs.vertcat(q_error, omega_error, ocp.model.u)
    ocp.cost.yref = np.zeros(7)
    ocp.cost.W = np.array(jax.scipy.linalg.block_diag(self.Q, self.R))

    # terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = cs.vertcat(q_error, omega_error)
    ocp.cost.yref_e = np.zeros(4)
    ocp.cost.W_e = np.array(self.Qf)
    
    # set constraints
    ocp.constraints.x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # placeholder, gets set in solve
    ocp.constraints.ubu = np.array(self.model.control_limits[:,1])
    ocp.constraints.lbu =  np.array(self.model.control_limits[:,0])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    ocp.constraints.ubx = np.array(self.model.state_limits[:,1]) 
    ocp.constraints.lbx = np.array(self.model.state_limits[:,0])
    ocp.constraints.idxbx = np.arange(3)
    
    # Set solver details
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = self.max_iter

    #ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    #ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    #ocp.solver_options.qp_solver_iter_max = 1000
    #ocp.solver_options.tol = 1e-4

    ocp_solver = AcadosOcpSolver(ocp, verbose=False)

    return ocp_solver

  
  def solve(self, x0:jax.Array, target_state:jax.Array, key:jax.Array, external_params:jax.Array, nominal_traj: jax.Array, nominal_cntrl: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Solves the optimal control problem starting at an initial state x0.

    Parameters:
      x0 (np.ndarray: initial state.
      verbose (bool): verbosity level of solver.

    Returns:
      x_traj (np.ndarray): state trajectory for system.
      u_traj (np.ndarray): control trajectory for system.
    """
    x_traj = np.zeros((self.horizon+1, len(x0)))
    u_traj = np.zeros((self.horizon, 3))

    self.ocp_solver.set(0, 'lbx', np.array(x0))
    self.ocp_solver.set(0, 'ubx', np.array(x0))

    nominal_traj = np.array(nominal_traj)
    nominal_cntrl = np.array(nominal_cntrl)

    for i in range(self.horizon):
      # Different parameter at each node
      p = jnp.concatenate([external_params[i,:], target_state])
      self.ocp_solver.set(i, 'p', np.array(p))

      # initialize to init traj
      self.ocp_solver.set(i, 'x', nominal_traj[i, :])
      self.ocp_solver.set(i, 'u', nominal_cntrl[i, :])

    # terminal condition
    p = jnp.concatenate([external_params[-1,:], target_state])
    self.ocp_solver.set(self.horizon, 'p', np.array(p))
    self.ocp_solver.set(self.horizon, 'x', nominal_traj[-1, :])
    
    status = self.ocp_solver.solve()

    verbose=False
    
    if verbose:
      if status == 0:
        print("Success")
      elif status == 1:
        print("Acados failure or nan")
      elif status == 2:
        print("Max iterations")
      elif status == 3:
        print("Acados minstep")
      elif status == 4:
        print("QP Failure")
    
    for i in range(self.horizon):
      x_traj[i,:] = self.ocp_solver.get(i, 'x')
      u_traj[i,:] = self.ocp_solver.get(i, 'u')
    x_traj[self.horizon,:] = self.ocp_solver.get(self.horizon, 'x')

    return jnp.array(x_traj), jnp.array(u_traj)
    
  def __call__(self, x0, target_state, key, external_params, nominal_traj, nominal_cntrl):
    # Ensure shapes use concrete integers (self.horizon)
    shape1 = jax.ShapeDtypeStruct((self.horizon + 1, x0.shape[-1]), jnp.float32)
    shape2 = jax.ShapeDtypeStruct((self.horizon, 3), jnp.float32)
    
    # Wrap the call to avoid hashing 'self'
    def callback_wrapper(x0, target, k, ext, nom_t, nom_c):
        return self.solve(x0, target, k, ext, nom_t, nom_c)

    return jax.pure_callback(
        callback_wrapper,
        (shape1, shape2),
        x0, target_state, key, external_params, nominal_traj, nominal_cntrl,
        vmap_method='sequential')