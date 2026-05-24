"""Spacecraft dynamics class. With rotation defined using a right-handed quaternion (not JPL convention)

Setup from "Differentiable Model Predictive Control on the GPU" by Emre Adabag, Marcus Greiff, John Subosits, and Thomas Lew
https://github.com/ToyotaResearchInstitute/diffmpc
"""
from typing import Any, Dict, Optional
from copy import deepcopy

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx 

from .magnetic_field import MagneticFieldModel
from .planetary_params import PlanetParams, Earth
from .base_dynamics import Dynamics

import pdb

orbit_parameters: Dict[str, Any] = {
    "num_states": 6,
    "num_controls": 3,
    "names_states": [
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
    ],
    "names_controls": ["thrust_x", "thrust_y", "thrust_z"], #[kN]
}
orbit_dynamics_parameters = {
    "mass": 0.1,
    "inertia": jnp.array([0.1, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 0.01, 0.1]),
}


# # Skew symmetric map (lie algebra of SO(3), solves a x b = S(a)b)
# def S(u):
#     return jnp.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])


# # Left quaternion product (used in the quaternion dynamics)
# def q_left(q):
#     qw = q[0]
#     qv = q[1:]
#     qL_B = jnp.concatenate((-qv.reshape(1, -1), S(qv)), axis=0)
#     qL_A = jnp.concatenate((jnp.array([[0]]), qv.reshape(-1, 1)), axis=0)
#     qL = jnp.concatenate((qL_A, qL_B), axis=1)
#     for ii in range(4):
#         qL = qL.at[ii, ii].set(qw)
#     return qL


# # Conjugate quaternion (used in the quaternion dynamics)
# def q_conj(q):
#     return jnp.concatenate((q[:1], -q[1:]))

# # Convert the quaternion into a rotation (rotates vector r as:  r' = get_rotation(q) @ r)
# # converts vector from body -> inertial frame (use q_conj(q) as input to go from inertial -> body)
# def get_rotation(q):
#     qw, qx, qy, qz = q
#     qw2 = qw * qw
#     qx2 = qx * qx
#     qy2 = qy * qy
#     qz2 = qz * qz
#     return jnp.array(
#         [
#             [qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
#             [2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)],
#             [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2],
#         ]
#     )

class OrbitDynamicsKeplerian(Dynamics):
    """Orbit dynamics class for generating orbit/magnetic field data"""
    mag_model: MagneticFieldModel
    planet: PlanetParams

    def __init__(self, parameters: Dict[str, Any] = None, dynamics_params: Dict[str, Any] = None, Planet: PlanetParams = None):
        """
        Initializes the class.
        Args:
            parameters:  parameters of the class.
                (str, Any) dictionary
            dynamics_parameters:  parameters for the dynamics of the class.
                (str, Any) dictionary
        """

        if parameters is None:
            parameters = deepcopy(orbit_parameters)
        if dynamics_params is None:
            dynamics_params = deepcopy(orbit_dynamics_parameters)

        # Default to Earth
        if Planet is None:
          Planet = Earth
        object.__setattr__(self, "mag_model", MagneticFieldModel(Planet))
        object.__setattr__(self, "planet", Planet)

        super().__init__(parameters, dynamics_params)
      
    def state_dot(
        self,
        state: jax.Array,
        control: jax.Array,
        t: float = 0.0,
        external_param: Optional[jax.Array] = None) -> jax.Array:
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
        mu = self.planet.mu
        a = state[0]
        
        # Mean motion: n = sqrt(mu / a^3)
        n = jnp.sqrt(mu / a**3)
        
        # In the simplest form, the rate of change of True Anomaly (nu_dot)
        # is derived from the angular momentum conservation:
        # h = r^2 * nu_dot -> nu_dot = h / r^2
        
        e = state[1]
        nu = state[5]
        r = a * (1 - e**2) / (1 + e * jnp.cos(nu))
        h = jnp.sqrt(mu * a * (1 - e**2))
        
        nu_dot = h / r**2
        
        # Elements [a, e, i, Omega, omega] are constant (dot = 0)
        return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, nu_dot])

    def get_magnetometer_reading(self, state, t, key, noise_std, bias, mag_model = None):
        """
        Simulates a magnetometer reading in the PCI frame [nT].
        
        Args:
            state: The current spacecraft state (contains quaternion q at state[6:10])
            t: time since epoch (seconds)
            key: JAX random key for noise generation
            noise_std: Standard deviation of sensor noise (nT)
            bias: jnp.array(3,) constant offset (nT)
        """
        if mag_model is None:
          mag_model = self.mag_model
        r = state[0:3]

        # True Physics
        b_pci = mag_model.compute_b_pci(r, t) # [nT]
        #b_body = get_rotation(q_conj(q)) @ b_eci

        if bias is not None:
          b_pci += bias
        
        # Sensor Noise
        noise = jrandom.normal(key, shape=(3,)) * noise_std
        return b_pci + noise

    @eqx.filter_jit
    def generate_magnetometer_data(self, trajectory, dt, num_steps, key, noise_std, bias, batch_size=0, mag_model = None):
      """Generates magnetometer data for 1 or more trajectories
      
      Args:
          trajectory: Either shape [num_steps+1, state_dim] for single trajectory
                     or shape [N, num_steps+1, state_dim] for N trajectories
          dt: Time step
          num_steps: Number of steps
          key: JAX random key
          noise_std: noise stdev for magnetometer
          bias: bias for magnetometer
          batch_size: size to chunk vmap, default is to process all in parallel, separate into batches if memory issue
          
      Returns:
          Magnetometer readings with shape matching input trajectory structure (in nT)
      """
      # Converts 2D array (single traj) to 3D with batch_size 1, doesn't change 3D array
      if trajectory.ndim == 2:
        batched_traj = trajectory[jnp.newaxis, :, :]
      else:
        batched_traj = trajectory
  
      n_traj = batched_traj.shape[0]
      batch_keys = jrandom.split(key, n_traj)
      times = jnp.linspace(0, dt * num_steps, num_steps + 1)
        

      if batch_size==0:
        batch_size = n_traj

      def single_trajectory(args):
        traj, key = args
        step_keys = jrandom.split(key, num_steps+1)
        return jax.vmap(self.get_magnetometer_reading, in_axes=(0,0,0,None,None,None))(traj, times, step_keys, noise_std, bias, mag_model)
      
      mag_data = jax.lax.map(single_trajectory, (batched_traj, batch_keys), batch_size=batch_size)

      return mag_data.reshape(trajectory.shape[:-1] + (-1,))

      def eci_to_orbital_elements(self, states):
        mu = self.planet.mu
        def cartesian_to_elements_single(state):
          r_vec = state[0:3]
          v_vec = state[3:6]
          
          r = jnp.linalg.norm(r_vec)
          v = jnp.linalg.norm(v_vec)
          
          h_vec = jnp.cross(r_vec, v_vec)
          h = jnp.linalg.norm(h_vec)
          
          # Robust Inclination
          inc = jnp.arccos(jnp.clip(h_vec[2] / h, -1.0, 1.0))
          
          n_vec = jnp.array([-h_vec[1], h_vec[0], 0])
          n = jnp.linalg.norm(n_vec)
          
          # Robust RAAN
          raan = jnp.where(n > 1e-9, jnp.arctan2(n_vec[1], n_vec[0]), 0.0)
          raan = jnp.mod(raan, 2 * jnp.pi)
          
          e_vec = ((v**2 - mu/r) * r_vec - jnp.dot(r_vec, v_vec) * v_vec) / mu
          ecc = jnp.linalg.norm(e_vec)
          
          energy = (v**2 / 2) - (mu / r)
          sma = -mu / (2 * energy)
          
          # Robust Argument of Periapsis
          # Avoid division by zero for circular orbits
          arg_p_cos = jnp.clip(jnp.dot(n_vec, e_vec) / (n * ecc + 1e-12), -1.0, 1.0)
          arg_p = jnp.where(n > 1e-9, jnp.arccos(arg_p_cos), 0.0)
          arg_p = jnp.where(e_vec[2] < 0, 2 * jnp.pi - arg_p, arg_p)
          
          # Robust True Anomaly
          nu_cos = jnp.clip(jnp.dot(e_vec, r_vec) / (ecc * r + 1e-12), -1.0, 1.0)
          nu = jnp.arccos(nu_cos)
          nu = jnp.where(jnp.dot(r_vec, v_vec) < 0, 2 * jnp.pi - nu, nu)
          nu = jnp.mod(nu, 2 * jnp.pi) # wrap to stay within 0 and 2pi
          return jnp.array([sma, ecc, inc, raan, arg_p, nu])
        return jax.vmap(cartesian_to_elements_single)(states)