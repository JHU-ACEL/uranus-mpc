"""Quadrotor dynamics class. With rotation defined using a right-handed quaternion (not JPL convention)

"Differentiable Model Predictive Control on the GPU" by Emre Adabag, Marcus Greiff, John Subosits, and Thomas Lew
https://github.com/ToyotaResearchInstitute/diffmpc
"""
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from copy import deepcopy

from .base_dynamics import Dynamics
from .quaternion_functions import S, q_left, q_conj, get_rotation, q_to_mrp, mrp_to_q

quadrotor_parameters: Dict[str, Any] = {
    "num_states": 13,
    "num_controls": 4,
    "names_states": [
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "q_0",
        "q_1",
        "q_2",
        "q_3",
        "omega_x",
        "omega_y",
        "omega_z",
    ],
    "names_controls": ["thrust", "torque_x", "torque_y", "torque_z"],
    "quat_start": 6, # start idx of quaternion
}
quadrotor_dynamics_parameters = {
    "mass": 0.1,
    "inertia": jnp.array([0.1, 0.01, 0.01, 0.01, 0.1, 0.01, 0.01, 0.01, 0.1]),
}

class QuadrotorDynamics(Dynamics):
    """Quadrotor dynamics class."""

    def __init__(self, parameters: Dict[str, Any] = None, dynamics_params: Dict[str, Any] = None):
        """
        Initializes the class.
        Args:
            parameters:  parameters of the class.
                (str, Any) dictionary
            dynamics_parameters:  parameters for the dynamics of the class.
                (str, Any) dictionary
        """

        if parameters is None:
            parameters = deepcopy(quadrotor_parameters)
        if dynamics_params is None:
            dynamics_params = deepcopy(quadrotor_dynamics_parameters)

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
        inertia = self.dynamics_params["inertia"].reshape((3, 3))
        inertia_inverse = jnp.linalg.inv(
            inertia
        )  # We might want to do this computation elsewhere.
        m = self.dynamics_params["mass"]
        g = 9.81

        # Extract states (no checks are done here)
        v = jnp.array(state[3:6])
        q = jnp.array(state[6:10])
        w = jnp.array(state[10:13])
        f = control[0] # +z direction in body frame
        tau = control[1:4] # in body frame

        # Normalize the quaternion (this should be done post integration, this is a hack)
        #q = q / jnp.linalg.norm(q) # now done post integration

        # Express dynamics on R9 x H, with velocity in global cooridnates (convention of scaramuzza)
        e3 = jnp.array([0, 0, 1])
        R = get_rotation(q)
        p_dot = v
        v_dot = -g * e3 + (f / m) * R @ e3 # * R @ e3 converts thrust from local frame +z to global frame +z
        q_dot = 0.5 * q_left(q) @ jnp.concatenate((jnp.array([0.0]), w))
        w_dot = inertia_inverse @ (S(inertia @ w) @ w + tau)

        state_dot = jnp.concatenate((p_dot, v_dot, q_dot, w_dot))
        return state_dot

    def quaternion_projection(self, state):
      quat_start = self.params["quat_start"]
      q = state[quat_start:quat_start+4]
      return state.at[quat_start:quat_start+4].set(q / jnp.linalg.norm(q))