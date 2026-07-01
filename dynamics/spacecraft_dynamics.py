"""Spacecraft dynamics class. With rotation defined using a right-handed quaternion (not JPL convention)

Setup from "Differentiable Model Predictive Control on the GPU" by Emre Adabag, Marcus Greiff, John Subosits, and Thomas Lew
https://github.com/ToyotaResearchInstitute/diffmpc
"""
from typing import Any, Dict, Optional
from copy import deepcopy

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx 

import pdb

from .base_dynamics import Dynamics
from .planetary_params import PlanetParams
from .quaternion_functions import S, q_left, q_conj, get_rotation, q_to_mrp, mrp_to_q

spacecraft_parameters: Dict[str, Any] = {
    "num_states": 7,
    "num_controls": 3,
    "names_states": [
        "q_0", # scalar part
        "q_1",
        "q_2",
        "q_3",
        "omega_x",
        "omega_y",
        "omega_z",
    ],
    "names_controls": ["dipole_x", "dipole_y", "dipole_z"], # [A m^2]
    "quat_start": 0,
}

# 1U properties from "MAGNETORQUER-ONLY ATTITUDE CONTROL OF SMALL SATELLITES USING TRAJECTORY OPTIMIZATION" - Gatherer
spacecraft_dynamics_parameters = {
    "mass": 0.75,
    "inertia": jnp.array([0.00125, 0.0001, 0.0001, 0.0001, 0.00125, 0.0001, 0.0001, 0.0001, 0.00125]).reshape((3, 3)),
}
spacecraft_dynamics_parameters["inertia_inv"] = jnp.linalg.inv(spacecraft_dynamics_parameters["inertia"]) 

class SpacecraftDynamics(Dynamics):
    mag_model: eqx.Module
    filter_spec: eqx.Module
    planet: PlanetParams
    """Spacecraft dynamics class."""
    def __init__(self, parameters: Dict[str, Any] = None, dynamics_params: Dict[str, Any] = None, mag_model: eqx.Module = None, planet: PlanetParams = None):
      """
      Initializes the class.
      Args:
          parameters:  parameters of the class.
              (str, Any) dictionary
          dynamics_parameters:  parameters for the dynamics of the class.
              (str, Any) dictionary
      """

      if parameters is None:
          parameters = deepcopy(spacecraft_parameters)
      if dynamics_params is None:
          dynamics_params = deepcopy(spacecraft_dynamics_parameters)
      object.__setattr__(self, "mag_model", mag_model)
      object.__setattr__(self, "planet", planet)

      # Define filter to freeze all params of mag_model except last layer
      filter_spec = jtu.tree_map(lambda _: False, mag_model)
      filter_spec = eqx.tree_at(
          lambda tree: (tree.layers[-1].weight, tree.layers[-1].bias),
          filter_spec,
          replace=(True, True),
      )
      object.__setattr__(self, "filter_spec", filter_spec)
    
      super().__init__(parameters, dynamics_params)
      
    def state_dot(
        self,
        state: jax.Array,
        control: jax.Array,
        t: float = 0.0,
        external_param: Optional[jax.Array] = None,
        disturbance_force: Optional[jax.Array] = jnp.array([0])) -> jax.Array:
        """
        Computes the time derivative of the state of the system.

        Returns x_dot = f(x, u) where f describes the dynamics of the system.

        Args:
            state: state of the system (see names_states)
                (_num_states, ) array
            control: control input applied to the system (see names_controls)
                (_num_controls, ) array
            params: parameters of the state_dot function of the dynamics.
                (str, Any) dictionary

        Returns:
            state_dot: time derivative of the state
                (_num_states, ) array
        """
        inertia = self.dynamics_params["inertia"]
        inertia_inverse = self.dynamics_params["inertia_inv"]

        # Get b-field (planet-centered inertial coordinate frame)
        b_pci = external_param
      
        # Extract states
        q = jnp.array(state[0:4]) 
        w = jnp.array(state[4:7]) # [rad/s]
        dipole = control[0:3] # [A m^2] magnetic dipoles are in body frame
      
        # Magnetorquer torque
        b_body = get_rotation(q_conj(q)) @ b_pci # using q_conj goes from pci to body
        tau = jnp.cross(dipole, b_body)*1e-9 + disturbance_force # convert nT to T, A*m^2 * T results in Nm
        # tau = control[0:3] + disturbance_force # [Nm] (use torque directly as control input)

        # Calculate angular velocities and accelerations
        # Conventions/formulation from "Planning with Attitude" - Jackson et. al.
        q_dot = 0.5 * q_left(q) @ jnp.concatenate((jnp.array([0.0]), w))
        w_dot = inertia_inverse @ (tau - S(w)@ inertia@w) # cross product a x b = a_skew_symmetric @ b
      
        state_dot = jnp.concatenate((q_dot, w_dot))
        return state_dot

    def quaternion_projection(self, state):
      quat_start = self.params["quat_start"]
      q = state[quat_start:quat_start+4]
      return state.at[quat_start:quat_start+4].set(q / jnp.linalg.norm(q))

    def get_error_coords(self, x, xbar):
      quat_start = self.params["quat_start"]
      # Convert to error coords
      dq = q_to_mrp(x[quat_start:quat_start+4], xbar[quat_start:quat_start+4])
      domega = x[quat_start+4:] - xbar[quat_start+4:]
      return jnp.concatenate((dq, domega))
    
    def get_true_coords(self, dx, xbar):
      quat_start = self.params["quat_start"]
      q = mrp_to_q(dx[quat_start:quat_start+3], xbar[quat_start:quat_start+4])
      omega = dx[quat_start+3:] + xbar[quat_start+4:]
      return jnp.concatenate((q, omega))

    def E(self, x: jax.Array):
      nx = self.params["num_states"]
      quat_start = self.params["quat_start"]
      q1, q2, q3, q4 = x[0], x[1], x[2], x[3]
      G = jnp.array([
          [-q2, -q3, -q4],
          [q1, -q4, q3],
          [q4, q1, -q2],
          [-q3, q2, q1]
      ])
      E = jnp.zeros((nx, nx-1))
      # Quaternion error mapping (rows 0:3, cols quat_start:quat_start+4)
      E = E.at[quat_start:quat_start+4, quat_start:quat_start+3].set(G)
      # Angular velocity passthrough
      E = E.at[quat_start+4:quat_start+7, quat_start+3:quat_start+6].set(jnp.eye(3))
      return E

    def linearize_and_discretize(self, nom_traj, nom_cntrl, ext_param, dt):
      nx = self.params["num_states"]-1
      def linearize_and_discretize_single(x,xp,u,ext_param):
        A = jax.jacfwd(self.state_dot,argnums=0)(x,u,0.0,ext_param) #df/dx
        B = jax.jacfwd(self.state_dot,argnums=1)(x,u,0.0,ext_param) # df/du
  
        # xp is x_k+1, transformation in "Planning with Attitude"
        A = self.E(xp).T @ A @ self.E(x)
        B = self.E(xp).T @ B
        
        Ad = jnp.eye(nx) + A*dt 
        Bd = B*dt
  
        return Ad, Bd
        
      Ad, Bd = jax.vmap(linearize_and_discretize_single)(nom_traj[:-1], nom_traj[1:], nom_cntrl, ext_param)
  
      return Ad, Bd



########### Dynamics w/o defining attitude jacobian, treats quats as normal states ###
class SpacecraftDynamicsQuat(Dynamics):
    mag_model: eqx.Module
    filter_spec: eqx.Module
    planet: PlanetParams
    """Spacecraft dynamics class."""
    def __init__(self, parameters: Dict[str, Any] = None, dynamics_params: Dict[str, Any] = None, mag_model: eqx.Module = None, planet: PlanetParams = None):
      """
      Initializes the class.
      Args:
          parameters:  parameters of the class.
              (str, Any) dictionary
          dynamics_parameters:  parameters for the dynamics of the class.
              (str, Any) dictionary
      """

      if parameters is None:
          parameters = deepcopy(spacecraft_parameters)
      if dynamics_params is None:
          dynamics_params = deepcopy(spacecraft_dynamics_parameters)
      object.__setattr__(self, "mag_model", mag_model)
      object.__setattr__(self, "planet", planet)

      # Define filter to freeze all params of mag_model except last layer
      filter_spec = jtu.tree_map(lambda _: False, mag_model)
      filter_spec = eqx.tree_at(
          lambda tree: (tree.layers[-1].weight, tree.layers[-1].bias),
          filter_spec,
          replace=(True, True),
      )
      object.__setattr__(self, "filter_spec", filter_spec)
    
      super().__init__(parameters, dynamics_params)
      
    def state_dot(
        self,
        state: jax.Array,
        control: jax.Array,
        t: float = 0.0,
        external_param: Optional[jax.Array] = None,
        disturbance_force: Optional[jax.Array] = jnp.array([0])) -> jax.Array:
        """
        Computes the time derivative of the state of the system.

        Returns x_dot = f(x, u) where f describes the dynamics of the system.

        Args:
            state: state of the system (see names_states)
                (_num_states, ) array
            control: control input applied to the system (see names_controls)
                (_num_controls, ) array
            params: parameters of the state_dot function of the dynamics.
                (str, Any) dictionary

        Returns:
            state_dot: time derivative of the state
                (_num_states, ) array
        """
        inertia = self.dynamics_params["inertia"]
        inertia_inverse = self.dynamics_params["inertia_inv"]

        # Get b-field (planet-centered inertial coordinate frame)
        b_pci = external_param
      
        # Extract states
        q = jnp.array(state[0:4]) 
        w = jnp.array(state[4:7]) # [rad/s]
        dipole = control[0:3] # [A m^2] magnetic dipoles are in body frame
      
        # Magnetorquer torque
        b_body = get_rotation(q_conj(q)) @ b_pci # using q_conj goes from pci to body
        tau = jnp.cross(dipole, b_body)*1e-9 + disturbance_force # convert nT to T, A*m^2 * T results in Nm
        # tau = control[0:3] + disturbance_force # [Nm] (use torque directly as control input)

        # Calculate angular velocities and accelerations
        # Conventions/formulation from "Planning with Attitude" - Jackson et. al.
        q_dot = 0.5 * q_left(q) @ jnp.concatenate((jnp.array([0.0]), w))
        w_dot = inertia_inverse @ (tau - S(w)@ inertia@w) # cross product a x b = a_skew_symmetric @ b
      
        state_dot = jnp.concatenate((q_dot, w_dot))
        return state_dot

    def quaternion_projection(self, state):
      quat_start = self.params["quat_start"]
      q = state[quat_start:quat_start+4]
      return state.at[quat_start:quat_start+4].set(q / jnp.linalg.norm(q))

      



############## Dynamics w/ torque as direct input      

class SpacecraftDynamicsTorque(Dynamics):
    mag_model: eqx.Module
    filter_spec: eqx.Module
    planet: PlanetParams
    """Spacecraft dynamics class."""
    def __init__(self, parameters: Dict[str, Any] = None, dynamics_params: Dict[str, Any] = None, mag_model: eqx.Module = None, planet: PlanetParams = None):
      """
      Initializes the class.
      Args:
          parameters:  parameters of the class.
              (str, Any) dictionary
          dynamics_parameters:  parameters for the dynamics of the class.
              (str, Any) dictionary
      """

      if parameters is None:
          parameters = deepcopy(spacecraft_parameters)
      if dynamics_params is None:
          dynamics_params = deepcopy(spacecraft_dynamics_parameters)
      object.__setattr__(self, "mag_model", mag_model)
      object.__setattr__(self, "planet", planet)

      # Define filter to freeze all params of mag_model except last layer
      filter_spec = jtu.tree_map(lambda _: False, mag_model)
      filter_spec = eqx.tree_at(
          lambda tree: (tree.layers[-1].weight, tree.layers[-1].bias),
          filter_spec,
          replace=(True, True),
      )
      object.__setattr__(self, "filter_spec", filter_spec)
    
      super().__init__(parameters, dynamics_params)
      
    def state_dot(
        self,
        state: jax.Array,
        control: jax.Array,
        t: float = 0.0,
        external_param: Optional[jax.Array] = None,
        disturbance_force: Optional[jax.Array] = jnp.array([0])) -> jax.Array:
        """
        Computes the time derivative of the state of the system.

        Returns x_dot = f(x, u) where f describes the dynamics of the system.

        Args:
            state: state of the system (see names_states)
                (_num_states, ) array
            control: control input applied to the system (see names_controls)
                (_num_controls, ) array
            params: parameters of the state_dot function of the dynamics.
                (str, Any) dictionary

        Returns:
            state_dot: time derivative of the state
                (_num_states, ) array
        """
        inertia = self.dynamics_params["inertia"]
        inertia_inverse = self.dynamics_params["inertia_inv"]

        # Get b-field (planet-centered inertial coordinate frame)
        b_pci = external_param
      
        # Extract states
        q = jnp.array(state[0:4]) 
        w = jnp.array(state[4:7]) # [rad/s]
        dipole = control[0:3] # [A m^2] magnetic dipoles are in body frame
      
        # Magnetorquer torque
        b_body = get_rotation(q_conj(q)) @ b_pci # using q_conj goes from pci to body
        #tau = jnp.cross(dipole, b_body)*1e-9 # convert nT to T, A*m^2 * T results in Nm
        tau = control[0:3] + disturbance_force # [Nm] (use torque directly as control input)

        # Calculate angular velocities and accelerations
        # Conventions/formulation from "Planning with Attitude" - Jackson et. al.
        q_dot = 0.5 * q_left(q) @ jnp.concatenate((jnp.array([0.0]), w))
        w_dot = inertia_inverse @ (tau - S(w)@ inertia@w) # cross product a x b = a_skew_symmetric @ b
      
        state_dot = jnp.concatenate((q_dot, w_dot))
        return state_dot

    def quaternion_projection(self, state):
      quat_start = self.params["quat_start"]
      q = state[quat_start:quat_start+4]
      return state.at[quat_start:quat_start+4].set(q / jnp.linalg.norm(q))

    def get_error_coords(self, x, xbar):
      quat_start = self.params["quat_start"]
      # Convert to error coords
      dq = q_to_mrp(x[quat_start:quat_start+4], xbar[quat_start:quat_start+4])
      domega = x[quat_start+4:] - xbar[quat_start+4:]
      return jnp.concatenate((dq, domega))
    
    def get_true_coords(self, dx, xbar):
      quat_start = self.params["quat_start"]
      q = mrp_to_q(dx[quat_start:quat_start+3], xbar[quat_start:quat_start+4])
      omega = dx[quat_start+3:] + xbar[quat_start+4:]
      return jnp.concatenate((q, omega))

    def E(self, x: jax.Array):
      nx = self.params["num_states"]
      quat_start = self.params["quat_start"]
      q1, q2, q3, q4 = x[0], x[1], x[2], x[3]
      G = jnp.array([
          [-q2, -q3, -q4],
          [q1, -q4, q3],
          [q4, q1, -q2],
          [-q3, q2, q1]
      ])
      E = jnp.zeros((nx, nx-1))
      # Quaternion error mapping (rows 0:3, cols quat_start:quat_start+4)
      E = E.at[quat_start:quat_start+4, quat_start:quat_start+3].set(G)
      # Angular velocity passthrough
      E = E.at[quat_start+4:quat_start+7, quat_start+3:quat_start+6].set(jnp.eye(3))
      return E

    def linearize_and_discretize(self, nom_traj, nom_cntrl, ext_param, dt):
      nx = self.params["num_states"]-1
      def linearize_and_discretize_single(x,xp,u,ext_param):
        A = jax.jacfwd(self.state_dot,argnums=0)(x,u,0.0,ext_param) #df/dx
        B = jax.jacfwd(self.state_dot,argnums=1)(x,u,0.0,ext_param) # df/du
  
        # xp is x_k+1, transformation in "Planning with Attitude"
        A = self.E(xp).T @ A @ self.E(x)
        B = self.E(xp).T @ B
        
        Ad = jnp.eye(nx) + A*dt 
        Bd = B*dt
  
        return Ad, Bd
        
      Ad, Bd = jax.vmap(linearize_and_discretize_single)(nom_traj[:-1], nom_traj[1:], nom_cntrl, ext_param)
  
      return Ad, Bd