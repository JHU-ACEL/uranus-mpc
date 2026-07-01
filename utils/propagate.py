import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from typing import Callable, Iterable, Tuple, Optional, Dict, Any # TODO: maybe do jax.typing?
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pdb

from dynamics.base_dynamics import Dynamics
from controllers.base_controller import Controller
from .coord_transforms import coord
from .learning import adapt_mag_model
from .plotting import plot_traj, plot_costs, plot_3D, plot_kde, plot_violin_and_bar, create_stats_df, save_data_to_pd_df

def sample_initial_states(batch_size, key, state_specs):
    """
    Generates a batch of random initial states based on provided specs.

    Args:
        batch_size: Number of states to sample.
        key: JAX key.
        state_specs: List of dicts, e.g., 
            [{'name': 'pos', 'shape': (3,), 'dist': 'uniform', 'min': -1, 'max': 1},
             {'name': 'rot', 'shape': (4,), 'dist': 'quaternion'}]
    """
    states = []
    
    for spec in state_specs:
        key, subkey = jrandom.split(key)
        dist_type = spec.get('dist', 'uniform')
        shape = (batch_size,) + spec.get('shape', (1,))

        if dist_type == 'uniform':
            val = jrandom.uniform(subkey, shape=shape, 
                                 minval=spec['min'], maxval=spec['max'])
        
        elif dist_type == 'normal':
            val = spec.get('mean', 0.0) + spec.get('std', 1.0) * jrandom.normal(subkey, shape=shape)

        elif dist_type == 'quaternion':
            # Specialized sampler for unit quaternions
            q = jrandom.normal(subkey, shape=shape)
            val = q / jnp.linalg.norm(q, axis=-1, keepdims=True)

        elif dist_type == 'constant':
            return jnp.broadcast_to(jnp.array(spec['value']), shape)

        states.append(val.reshape(batch_size, -1))
        #TODO: Potentially set trajectory up as pytrees (traj.pos instead of traj[:,0:3])
    return jnp.concatenate(states, axis=-1)

class TrajectoryGenerator(eqx.Module):
  dynamics: Dynamics
  dt: float

  def rk4_step(self, state, control, t, external_dynamics_param, key, noise_std):
    def dynamics_wrapper(s,k):
      key, noise_key = jrandom.split(k)
      u_noise = jrandom.normal(noise_key, shape=control.shape) * noise_std
      return self.dynamics.state_dot(s, control, t, external_dynamics_param, u_noise)

    # Note: quaternion projection is just identity unless defined in Dynamics class
    k1 = dynamics_wrapper(state,key)
    k2 = dynamics_wrapper(self.dynamics.quaternion_projection(state + 0.5 * self.dt * k1),key)
    k3 = dynamics_wrapper(self.dynamics.quaternion_projection(state + 0.5 * self.dt * k2),key)
    k4 = dynamics_wrapper(self.dynamics.quaternion_projection(state + self.dt * k3),key)
  
    dx = (self.dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # dx = self.dt*dynamics_wrapper(state) # euler

    # key1, key = jrandom.split(key)
    # noise = jrandom.normal(key, shape=state.shape) * noise_std * jnp.sqrt(self.dt)

    return self.dynamics.quaternion_projection(state + dx)

  def generate_trajectory(
    self,
    initial_state: jax.Array, 
    key: jax.Array,
    target_state: Optional[jax.Array] = None, 
    noise_std: Optional[jax.Array] = 0.0,
    num_steps: Optional[int] = 200,
    external_dynamics_params: Optional[jax.Array] = None, # Ex: Magnetic field [Bx, By, Bz] for each timestep
    external_dynamics_params_est: Optional[jax.Array] = None,
    high_level_controller: Optional[Controller] = None,
    replan_freq: Optional[int] = 10, 
    adapt_freq: Optional[int] = None,
    history: Optional[int] = 100,
    low_level_controller: Optional[Controller] = None,
    control_sequence: Optional[jax.Array] = None # this overides controller if provided
    ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory using either controller function or control sequence.
    
    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """    
    if high_level_controller is not None and high_level_controller.horizon >num_steps:
      if external_dynamics_params is None:
        external_dynamics_params = jnp.zeros((high_level_controller.horizon, 1))
      else:
        assert external_dynamics_params.shape[0] >= high_level_controller.horizon, \
            f"external_dynamics_params sequence must have at least length {num_steps}"
        # trim external params to num_steps length
        ext_dyn_params = external_dynamics_params[:high_level_controller.horizon]
    else:
      if external_dynamics_params is None:
        ext_dyn_params = jnp.zeros((num_steps, 1))
      else:
        assert external_dynamics_params.shape[0] >= num_steps, \
            f"external_dynamics_params sequence must have at least length {num_steps}"
        # trim external params to num_steps length
        ext_dyn_params = external_dynamics_params[:num_steps]

    if control_sequence is not None:
      assert control_sequence.shape[0] == num_steps, \
          f"Control sequence must have length {num_steps}"
      return self._generate_trajectory_seq(initial_state, target_state, key, noise_std, num_steps, ext_dyn_params,
    control_sequence)
    elif low_level_controller and high_level_controller is not None:
      if adapt_freq is None:
        if external_dynamics_params_est is None:
          return self._generate_trajectory_high_and_low_level_control(initial_state, target_state, key, noise_std, num_steps, ext_dyn_params, high_level_controller, low_level_controller, replan_freq)
        else:
          return self._generate_trajectory_high_and_low_level_control_param_est(initial_state, target_state, key, noise_std, num_steps, ext_dyn_params, external_dynamics_params_est[:num_steps], high_level_controller, low_level_controller, replan_freq)
      else:
        return self._generate_trajectory_high_and_low_level_control_adapt(initial_state, target_state, key, noise_std, num_steps, ext_dyn_params, external_dynamics_params_est[:num_steps], high_level_controller, low_level_controller, replan_freq, adapt_freq, history)
    elif low_level_controller is not None:
      if target_state.ndim < 2:
        target_state = jnp.tile(target_state,(num_steps+1,1))
      return self._generate_trajectory_low_level_control(initial_state, target_state, key, noise_std, num_steps, ext_dyn_params, low_level_controller)
    elif high_level_controller is not None:
      if adapt_freq is None:
        if external_dynamics_params_est is None:
          return self._generate_trajectory_high_level_control(initial_state, target_state, key, noise_std, num_steps, ext_dyn_params, high_level_controller, replan_freq)
        else:
          return self._generate_trajectory_high_level_control_param_est(initial_state, target_state, key, noise_std, num_steps, ext_dyn_params, external_dynamics_params_est[:num_steps], high_level_controller, replan_freq)
      else:
        return self._generate_trajectory_high_level_control_adapt(initial_state, target_state, key, noise_std, num_steps, ext_dyn_params, external_dynamics_params_est[:num_steps], high_level_controller, replan_freq, adapt_freq, history)
    else:
      control_sequence = jnp.zeros((num_steps, self.dynamics.num_controls))
      return self._generate_trajectory_seq(initial_state, target_state, key, noise_std, num_steps, ext_dyn_params, control_sequence)

  def generate_trajectory_batch(self,
                                initial_states: jax.Array,
                                key: jax.Array,
                                batch_size: int,
                                target_states: Optional[jax.Array] = None,
                                noise_std: Optional[jax.Array] = 0.0,
                                num_steps: Optional[int] = 200,
                                external_dynamics_params: Optional[jax.Array] = None,
                                external_dynamics_params_est: Optional[jax.Array] = None,
                                high_level_controller: Optional[Controller] = None,
                                replan_freq: Optional[int] = 10,
                                adapt_freq: Optional[int] = None,
                                history: Optional[int] = 100,
                                low_level_controller: Optional[Controller] = None,
                                control_sequence: Optional[jax.Array] = None,
                                chunk_size: Optional[int] = 0):
      
    keys = jrandom.split(key, batch_size)
    actual_chunk_size = chunk_size or batch_size

    # Broadcast scalars/unbatched arrays to [batch_size, ...]
    def prepare_input(x, dummy_shape=(0,)):
        if x is None:
            return jnp.zeros((batch_size, *dummy_shape))
        if x.ndim == (2 if len(dummy_shape) == 1 else 3): # Already batched
            return x
        return jnp.broadcast_to(x, (batch_size, *x.shape))
    
    b_init = prepare_input(initial_states, (self.dynamics.num_states,))
    b_target = prepare_input(target_states, (self.dynamics.num_states,))
    b_extern = prepare_input(external_dynamics_params, (num_steps, 1))
    if external_dynamics_params_est is not None:
      b_extern_est = prepare_input(external_dynamics_params_est, (num_steps,1))
    else:
      b_extern_est = jnp.zeros(b_extern.shape) # just placeholder, not used
    b_ctrl_seq = prepare_input(control_sequence, (num_steps, self.dynamics.num_controls))

    # Logic for a single trajectory
    def map_fn(args):
        # Unpack the single-trajectory slices
        init, target, k, extern, extern_est, c_seq = args
        
        if control_sequence is not None:
            return self._generate_trajectory_seq(init, target, k, noise_std, num_steps, extern, c_seq)
        
        elif low_level_controller is not None and high_level_controller is not None:
          if adapt_freq is None:
            if external_dynamics_params_est is None:
              return self._generate_trajectory_high_and_low_level_control(
                init, target, k, noise_std, num_steps, extern, 
                high_level_controller, low_level_controller, replan_freq)
            else:
              return self._generate_trajectory_high_and_low_level_control_param_est(
                init, target, k, noise_std, num_steps, extern, extern_est, 
                high_level_controller, low_level_controller, replan_freq)
          else:
            return self._generate_trajectory_high_and_low_level_control_adapt(
                init, target, k, noise_std, num_steps, extern, extern_est, 
                high_level_controller, low_level_controller, replan_freq, adapt_freq, history)
            
        elif low_level_controller is not None:
            if target.ndim < 2:
              target = jnp.tile(target,(num_steps+1,1))
            return self._generate_trajectory_low_level_control(
                init, target, k, noise_std, num_steps, extern, low_level_controller
            )
            
        elif high_level_controller is not None:
          if adapt_freq is None:
            if external_dynamics_params_est is None:
              return self._generate_trajectory_high_level_control(
                init, target, k, noise_std, num_steps, extern, high_level_controller, replan_freq)
            else:
              return self._generate_trajectory_high_level_control_param_est(
                init, target, k, noise_std, num_steps, extern, extern_est, high_level_controller, replan_freq)
          else:
            return self._generate_trajectory_high_level_control_adapt(
                init, target, k, noise_std, num_steps, extern, extern_est, high_level_controller, replan_freq, adapt_freq, history)
            
        else:
            return self._generate_trajectory_seq(init, target, k, noise_std, num_steps, extern, c_seq)

    # Executes actual_chunk_size items in parallel using vmap internally, then loops through the rest of the batch.
    #return jax.vmap(map_fn)((b_init, b_target, keys, b_extern, b_extern_est, b_ctrl_seq))
    return jax.lax.map(map_fn, (b_init, b_target, keys, b_extern, b_extern_est, b_ctrl_seq), batch_size=actual_chunk_size)

  @eqx.filter_jit
  def _generate_trajectory_seq(
    self,
    initial_state: jax.Array, 
    target_state: jax.Array, 
    key: jax.Array,
    noise_std: jax.Array,
    num_steps: int,
    external_dynamics_params: jax.Array,
    control_sequence: jax.Array
    ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory using  control sequence.
    
    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """  
    scan_inputs = (control_sequence, external_dynamics_params)
    
    def scan_step(carry, scan_input):
      state, key, i = carry
      control_input, extern_dyn_param = scan_input
      t_current = i * self.dt
      key, rk4_key = jrandom.split(key)
      next_state = self.rk4_step(state, control_input, t_current, extern_dyn_param, rk4_key, noise_std)
      return (next_state, key, i+1), (state, control_input)
  
    # Run scan
    init_carry = (initial_state, key, 0)
    (final_state, final_key, _), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, scan_inputs)
  
    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])
  
    return trajectory, controls

  @eqx.filter_jit
  def _generate_trajectory_low_level_control(
    self,
    initial_state: jax.Array, 
    target_state: jax.Array, 
    key: jax.Array,
    noise_std: jax.Array,
    num_steps: int,
    external_dynamics_params: jax.Array, 
    low_level_controller: Controller,
    ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory using low-level controller (PID, LQR, etc)
    
    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """  
    ext_params = external_dynamics_params[:target_state.shape[0]-1]
    nominal_traj = target_state
    nominal_cntrl = jnp.zeros((target_state.shape[0]-1, self.dynamics.num_controls))

    scan_inputs = ext_params

    def scan_step(carry, scan_input):
      state, key, i, cntrl_param = carry
      extern_dyn_param = scan_input
      t_current = i * self.dt
      key, rk4_key, cntrl_key = jrandom.split(key,3)
      u0 = jnp.zeros((self.dynamics.num_controls,))
      # control_input, cntrl_param = low_level_controller(state, target_state, cntrl_key, external_dynamics_params, nominal_traj, nominal_cntrl, cntrl_param)
      control_input, cntrl_param = low_level_controller(state, u0, target_state[i], extern_dyn_param, cntrl_param)
      next_state = self.rk4_step(state, control_input, t_current, extern_dyn_param, rk4_key, noise_std) 
      return (next_state, key, i+1, cntrl_param), (state, control_input)

    # Run scan
    cntrl_param = low_level_controller.cntrl_param_init
    init_carry = (initial_state, key, 0, cntrl_param)
    (final_state, final_key, _, _), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, scan_inputs)
  
    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])
  
    return trajectory, controls

  @eqx.filter_jit
  def _generate_trajectory_high_level_control(
    self,
    initial_state: jax.Array, 
    target_state: jax.Array, 
    key: jax.Array,
    noise_std: jax.Array,
    num_steps: int,
    external_dynamics_params: jax.Array, 
    high_level_controller: Controller,
    replan_freq: int, 
    ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory using high-level controller (MPPI, CEM, etc)
    
    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """  

    def scan_step(carry, _):
      state, key, i, nominal_traj, nominal_cntrl = carry
      t_current = i * self.dt
      key, rk4_key, cntrl_key = jrandom.split(key, 3)

      def controller_wrapper(operand):
        return high_level_controller(*operand)
        
      def no_update(operand):
        *_ ,nominal_traj, nominal_cntrl = operand
        # Shift nominal trajectory and nominal cntrl
        nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
        nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
        return nominal_traj, nominal_cntrl

      params_dim = external_dynamics_params.shape[1]
      horizon = high_level_controller.horizon

      # True magnetic field
      dyn_params_slice = jax.lax.dynamic_slice(
          external_dynamics_params,
          (i, 0),
          (horizon, params_dim)
      )

      nominal_traj, nominal_cntrl = jax.lax.cond(
        i % replan_freq == 0,
        controller_wrapper,
        no_update,
        operand=(state, target_state, cntrl_key, dyn_params_slice, nominal_traj, nominal_cntrl)
      )
      #control_input = nominal_cntrl[i % replan_freq]
      control_input = nominal_cntrl[0] # always 0 because no_update fn shifts
      next_state = self.rk4_step(state, control_input, t_current, dyn_params_slice[0], rk4_key, noise_std)
      return (next_state, key, i+1, nominal_traj, nominal_cntrl), (state, control_input)
      
    # Run scan
    init_nom_traj = jnp.tile(initial_state,(high_level_controller.horizon+1,1))
    #t = jnp.linspace(0, self.dt*high_level_controller.horizon, high_level_controller.horizon + 1).reshape(-1, 1)  # Shape: (horizon+1, 1)
    #init_nom_traj = initial_state + t * (target_state - initial_state) 
    key, init_cntrl_key = jrandom.split(key)
    init_nom_cntrl = 0.001*jrandom.normal(init_cntrl_key, shape=((high_level_controller.horizon, self.dynamics.num_controls)))
    init_carry = (initial_state, key, 0, init_nom_traj, init_nom_cntrl)
    (final_state, final_key, _, nominal_traj, nominal_cntrl), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, length=num_steps)
  
    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])
  
    return trajectory, controls#, nominal_traj, nominal_cntrl

  @eqx.filter_jit
  def _generate_trajectory_high_and_low_level_control(
    self,
    initial_state: jax.Array, 
    target_state: jax.Array, 
    key: jax.Array,
    noise_std: jax.Array,
    num_steps: int,
    external_dynamics_params: jax.Array, 
    high_level_controller: Controller,
    low_level_controller: Controller,
    replan_freq: int, 
    ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory by planning high-level controller (MPC, MPPI, CEM, etc) and then using a 
    low-level controller (PID, LQR, etc) to track nominal trajectory 
    
    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """  
    
    def scan_step(carry, _):
      state, key, i, cntrl_param, nominal_traj, nominal_cntrl, mag_model = carry
      t_current = i * self.dt
      key, rk4_key, cntrl_key = jrandom.split(key, 3)

      def controller_wrapper(operand):
        return high_level_controller(*operand)
        
      def no_update(operand):
        *_ ,nominal_traj, nominal_cntrl = operand 
        # Shift nominal trajectory and nominal cntrl
        nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
        nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
        return nominal_traj, nominal_cntrl

      params_dim = external_dynamics_params.shape[1]
      horizon = high_level_controller.horizon

      # True magnetic field in PCI coords
      dyn_params_slice = jax.lax.dynamic_slice(
          external_dynamics_params,
          (i, 0),
          (horizon, params_dim)
      )

      nominal_traj, nominal_cntrl = jax.lax.cond(
        i % replan_freq == 0,
        controller_wrapper,
        no_update,
        operand=(state, target_state, cntrl_key, dyn_params_slice, nominal_traj, nominal_cntrl)
      )
      # control_input, cntrl_param = low_level_controller(state, target_state, cntrl_key, dyn_params_slice, nominal_traj, nominal_cntrl, cntrl_param)
      control_input, cntrl_param = low_level_controller(state, nominal_cntrl[0], nominal_traj[0], dyn_params_slice[i % replan_freq], cntrl_param)
      next_state = self.rk4_step(state, control_input, t_current, dyn_params_slice[0], rk4_key, noise_std)
      return (next_state, key, i+1, cntrl_param, nominal_traj, nominal_cntrl, mag_model), (state, control_input)
      
    # Run scan
    init_nom_traj = jnp.tile(initial_state,(high_level_controller.horizon+1,1))
    #t = jnp.linspace(0, self.dt*high_level_controller.horizon, high_level_controller.horizon + 1).reshape(-1, 1)  # Shape: (horizon+1, 1)
    #init_nom_traj = initial_state + t * (target_state - initial_state) 
    key, init_cntrl_key = jrandom.split(key)
    init_nom_cntrl = 0.001*jrandom.normal(init_cntrl_key, shape=((high_level_controller.horizon, self.dynamics.num_controls)))
    init_cntrl_param = low_level_controller.cntrl_param_init
    init_carry = (initial_state, key, 0, init_cntrl_param, init_nom_traj, init_nom_cntrl, self.dynamics.mag_model)
    
    (final_state, final_key, _, _, nominal_traj, nominal_cntrl, mag_model), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, length=num_steps)
  
    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])
  
    return trajectory, controls #, nominal_traj, nominal_cntrl

  @eqx.filter_jit
  def _generate_trajectory_high_and_low_level_control_param_est(
    self,
    initial_state: jax.Array, 
    target_state: jax.Array, 
    key: jax.Array,
    noise_std: jax.Array,
    num_steps: int,
    external_dynamics_params: jax.Array, 
    external_dynamics_params_est: jax.Array,
    high_level_controller: Controller,
    low_level_controller: Controller,
    replan_freq: int, 
    ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory by planning high-level controller (MPC, MPPI, CEM, etc) and then using a 
    low-level controller (PID, LQR, etc) to track nominal trajectory 
    
    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """  
    
    def scan_step(carry, _):
      state, key, i, cntrl_param, nominal_traj, nominal_cntrl, mag_model = carry
      t_current = i * self.dt
      key, rk4_key, cntrl_key = jrandom.split(key, 3)

      def controller_wrapper(operand):
        return high_level_controller(*operand)
        
      def no_update(operand):
        *_ ,nominal_traj, nominal_cntrl = operand
        # Shift nominal trajectory and nominal cntrl
        nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
        nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
        return nominal_traj, nominal_cntrl

      params_dim = external_dynamics_params.shape[1]
      est_params_dim = external_dynamics_params_est.shape[1]
      horizon = high_level_controller.horizon

      # True magnetic field in PCI coords
      dyn_params_slice = jax.lax.dynamic_slice(
          external_dynamics_params,
          (i, 0),
          (horizon, params_dim)
      )

      # This is XYZ coords in PCI coords
      dyn_params_est_slice = jax.lax.dynamic_slice(
          external_dynamics_params_est,
          (i, 0), # start idx
          (horizon, est_params_dim) # slice size (must use fixed values)
      )
 
      ############### Use Learned Model #########################
      # Convert XYZ in PCI to PCPF
      t_slice = jnp.linspace(t_current, t_current + self.dt*horizon, horizon)
      dyn_params_est_slice = coord.pci_to_pcpf(dyn_params_est_slice, t_slice, self.dynamics.planet) 
      # Convert dyn_params_est_slice (xyz pos) to PCPF -> 4D spherical
      dyn_params_est_slice = coord.cartesian_to_spherical_4D(dyn_params_est_slice)
      # Compute b-field (4D spherical -> B-field in PCPF)
      dyn_params_est_slice = jax.vmap(mag_model)(dyn_params_est_slice)
      # Convert b-field back to PCI and save as dyn_params_est_slice
      dyn_params_est_slice = coord.pcpf_to_pci(dyn_params_est_slice, t_slice, self.dynamics.planet)
      #############################################################
      
      nominal_traj, nominal_cntrl = jax.lax.cond(
        i % replan_freq == 0,
        controller_wrapper,
        no_update,
        operand=(state, target_state, cntrl_key, dyn_params_est_slice, nominal_traj, nominal_cntrl)
      )
      # control_input, cntrl_param = low_level_controller(state, target_state, cntrl_key, dyn_params_slice, nominal_traj, nominal_cntrl, cntrl_param)
      control_input, cntrl_param = low_level_controller(state, nominal_cntrl[0], nominal_traj[0], dyn_params_slice[i % replan_freq], cntrl_param)
      next_state = self.rk4_step(state, control_input, t_current, dyn_params_slice[0], rk4_key, noise_std)
      return (next_state, key, i+1, cntrl_param, nominal_traj, nominal_cntrl, mag_model), (state, control_input)
      
    # Run scan
    init_nom_traj = jnp.tile(initial_state,(high_level_controller.horizon+1,1))
    key, init_cntrl_key = jrandom.split(key)
    init_nom_cntrl = 0.001*jrandom.normal(init_cntrl_key, shape=((high_level_controller.horizon, self.dynamics.num_controls)))
    init_cntrl_param = low_level_controller.cntrl_param_init
    init_carry = (initial_state, key, 0, init_cntrl_param, init_nom_traj, init_nom_cntrl, self.dynamics.mag_model)
    
    (final_state, final_key, _, _, nominal_traj, nominal_cntrl, mag_model), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, length=num_steps)
  
    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])
  
    return trajectory, controls#, nominal_traj, nominal_cntrl

  @eqx.filter_jit
  def _generate_trajectory_high_level_control_param_est(
    self,
    initial_state: jax.Array, 
    target_state: jax.Array, 
    key: jax.Array,
    noise_std: jax.Array,
    num_steps: int,
    external_dynamics_params: jax.Array, 
    external_dynamics_params_est: jax.Array,
    high_level_controller: Controller,
    replan_freq: int, 
    ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory using high-level controller (MPC, MPPI, CEM, etc)
    
    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """  

    def scan_step(carry, _):
      state, key, i, nominal_traj, nominal_cntrl = carry
      t_current = i * self.dt
      key, rk4_key, cntrl_key = jrandom.split(key, 3)

      def controller_wrapper(operand):
        return high_level_controller(*operand)
        
      def no_update(operand):
        *_ ,nominal_traj, nominal_cntrl = operand
        # Shift nominal trajectory and nominal cntrl
        nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
        nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
        return nominal_traj, nominal_cntrl

      params_dim = external_dynamics_params.shape[1]
      est_params_dim = external_dynamics_params_est.shape[1]
      horizon = high_level_controller.horizon

      dyn_params_slice = jax.lax.dynamic_slice(
          external_dynamics_params,
          (i, 0),
          (horizon, params_dim)
      )

      # This is XYZ coords in PCI coords
      dyn_params_est_slice = jax.lax.dynamic_slice(
          external_dynamics_params_est,
          (i, 0),
          (horizon, est_params_dim)
      )

      # # # Compute B-field from learned model to pass to controller 
      # Convert XYZ in PCI to PCPF
      t_slice = jnp.linspace(t_current, t_current + self.dt*horizon, horizon)
      dyn_params_est_slice = coord.pci_to_pcpf(dyn_params_est_slice, t_slice, self.dynamics.planet) 
      # Convert dyn_params_est_slice (xyz pos) to PCPF -> 4D spherical
      dyn_params_est_slice = coord.cartesian_to_spherical_4D(dyn_params_est_slice)
      # Compute b-field (4D spherical -> B-field in PCPF)
      dyn_params_est_slice = jax.vmap(self.dynamics.mag_model)(dyn_params_est_slice)
      # Convert b-field back to PCI and save as dyn_params_est_slice
      dyn_params_est_slice = coord.pcpf_to_pci(dyn_params_est_slice, t_slice, self.dynamics.planet)

      nominal_traj, nominal_cntrl = jax.lax.cond(
        i % replan_freq == 0,
        controller_wrapper,
        no_update,
        operand=(state, target_state, cntrl_key, dyn_params_est_slice, nominal_traj, nominal_cntrl)
      )
      control_input = nominal_cntrl[0]
      next_state = self.rk4_step(state, control_input, t_current, dyn_params_slice[0], rk4_key, noise_std)
      return (next_state, key, i+1, nominal_traj, nominal_cntrl), (state, control_input)
      
    # Run scan
    init_nom_traj = jnp.tile(initial_state,(high_level_controller.horizon+1,1))
    key, init_cntrl_key = jrandom.split(key)
    init_nom_cntrl = 0.001*jrandom.normal(init_cntrl_key, shape=((high_level_controller.horizon, self.dynamics.num_controls)))
    init_carry = (initial_state, key, 0, init_nom_traj, init_nom_cntrl)
    (final_state, final_key, _, nominal_traj, nominal_cntrl), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, length=num_steps)
  
    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])
  
    return trajectory, controls#, nominal_traj, nominal_cntrl

  @eqx.filter_jit
  def _generate_trajectory_high_level_control_adapt(
    self,
    initial_state: jax.Array, 
    target_state: jax.Array, 
    key: jax.Array,
    noise_std: jax.Array,
    num_steps: int,
    external_dynamics_params: jax.Array, 
    external_dynamics_params_est: jax.Array,
    high_level_controller: Controller,
    replan_freq: int, 
    adapt_freq: int,
    history: int,
    ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory using high-level controller (MPC, MPPI, CEM, etc)
    
    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """  

    def scan_step(carry, _):
      state, key, i, nominal_traj, nominal_cntrl = carry
      t_current = i * self.dt
      key, rk4_key, cntrl_key, mag_key = jrandom.split(key, 4)

      def controller_wrapper(operand):
        return high_level_controller(*operand)
        
      def no_update(operand):
        *_ ,nominal_traj, nominal_cntrl = operand
        # Shift nominal trajectory and nominal cntrl
        nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
        nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
        return nominal_traj, nominal_cntrl

      params_dim = external_dynamics_params.shape[1]
      est_params_dim = external_dynamics_params_est.shape[1]
      horizon = high_level_controller.horizon

      dyn_params_slice = jax.lax.dynamic_slice(
          external_dynamics_params,
          (i, 0),
          (horizon, params_dim)
      )

      # This is XYZ coords in PCI coords
      dyn_params_est_slice = jax.lax.dynamic_slice(
          external_dynamics_params_est,
          (i, 0),
          (horizon, est_params_dim)
      )

      ############ Adaptation ##################
      # Simulate noisy magnetometer measurement
      magnetometer_measurements = jax.lax.dynamic_slice(
        external_dynamics_params + jrandom.normal(mag_key, shape=external_dynamics_params.shape) * 1e4,
        (i-history, 0), # uses history (no knowledge of future mag measurements)
        (history, params_dim),
        allow_negative_indices=False # will throw error if cond. not set up correctly which would allow it to use future measurements
      )

      dyn_params_est_adapt_slice = jax.lax.dynamic_slice(
          external_dynamics_params_est,
          (i-history, 0),
          (history, est_params_dim),
          allow_negative_indices=False
      )
      
      t_adapt_slice = jnp.linspace( (i-history)*self.dt, (i-history)*self.dt + self.dt*history, history)
      pcpf_positions = coord.pci_to_pcpf(dyn_params_est_adapt_slice, t_adapt_slice, self.dynamics.planet)
      pcpf_positions = coord.cartesian_to_spherical_4D(pcpf_positions)
      def adapt_wrapper(operand):
        return adapt_mag_model(*operand)

      def no_adapt_update(operand):
        model, *_ = operand
        return model
        
      mag_model = jax.lax.cond(
        (i % adapt_freq == 0) & (i >= history), 
        adapt_wrapper,
        no_adapt_update,
        operand=(mag_model, pcpf_positions, magnetometer_measurements)
      )
      ########################################################

      ############### Use Learned Model #########################
      # Convert XYZ in PCI to PCPF
      t_slice = jnp.linspace(t_current, t_current + self.dt*horizon, horizon)
      dyn_params_est_slice = coord.pci_to_pcpf(dyn_params_est_slice, t_slice, self.dynamics.planet) 
      # Convert dyn_params_est_slice (xyz pos) to PCPF -> 4D spherical
      dyn_params_est_slice = coord.cartesian_to_spherical_4D(dyn_params_est_slice)
      # Compute b-field (4D spherical -> B-field in PCPF)
      dyn_params_est_slice = jax.vmap(mag_model)(dyn_params_est_slice)
      # Convert b-field back to PCI and save as dyn_params_est_slice
      dyn_params_est_slice = coord.pcpf_to_pci(dyn_params_est_slice, t_slice, self.dynamics.planet)
      #############################################################

      nominal_traj, nominal_cntrl = jax.lax.cond(
        i % replan_freq == 0,
        controller_wrapper,
        no_update,
        operand=(state, target_state, cntrl_key, dyn_params_est_slice, nominal_traj, nominal_cntrl)
      )
      control_input = nominal_cntrl[0]
      next_state = self.rk4_step(state, control_input, t_current, dyn_params_slice[0], rk4_key, noise_std)
      return (next_state, key, i+1, nominal_traj, nominal_cntrl), (state, control_input)
      
    # Run scan
    init_nom_traj = jnp.tile(initial_state,(high_level_controller.horizon+1,1))
    key, init_cntrl_key = jrandom.split(key)
    init_nom_cntrl = 0.001*jrandom.normal(init_cntrl_key, shape=((high_level_controller.horizon, self.dynamics.num_controls)))
    init_carry = (initial_state, key, 0, init_nom_traj, init_nom_cntrl)
    (final_state, final_key, _, nominal_traj, nominal_cntrl), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, length=num_steps)
  
    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])
  
    return trajectory, controls#, nominal_traj, nominal_cntrl
      
  @eqx.filter_jit
  def _generate_trajectory_high_and_low_level_control_adapt(
    self,
    initial_state: jax.Array, 
    target_state: jax.Array, 
    key: jax.Array,
    noise_std: jax.Array,
    num_steps: int,
    external_dynamics_params: jax.Array, 
    external_dynamics_params_est: jax.Array,
    high_level_controller: Controller,
    low_level_controller: Controller,
    replan_freq: int, 
    adapt_freq: int, 
    history: int,
    ) -> Tuple[jax.Array, jax.Array]: 
    """
    Generate trajectory by planning high-level controller (MPC, MPPI, CEM, etc) and then using a 
    low-level controller (PID, LQR, etc) to track nominal trajectory 
    
    Returns:
        trajectory: Shape (N + 1, nx)
        controls: Shape (N, nu)
    """  
    
    def scan_step(carry, _):
      state, key, i, cntrl_param, nominal_traj, nominal_cntrl, mag_model = carry
      t_current = i * self.dt
      key, rk4_key, cntrl_key, mag_key = jrandom.split(key, 4)

      def controller_wrapper(operand):
        return high_level_controller(*operand)
        
      def no_update(operand):
        *_ ,nominal_traj, nominal_cntrl = operand
        # Shift nominal trajectory and nominal cntrl
        nominal_traj = jnp.concatenate((nominal_traj[1:],jnp.expand_dims(nominal_traj[-1],axis=0)),axis=0)
        nominal_cntrl = jnp.concatenate((nominal_cntrl[1:],jnp.expand_dims(nominal_cntrl[-1],axis=0)),axis=0)
        return nominal_traj, nominal_cntrl

      params_dim = external_dynamics_params.shape[1]
      est_params_dim = external_dynamics_params_est.shape[1]
      horizon = high_level_controller.horizon

      # True magnetic field
      dyn_params_slice = jax.lax.dynamic_slice(
          external_dynamics_params,
          (i, 0),
          (horizon, params_dim)
      )

      # This is XYZ coords in PCI coords
      dyn_params_est_slice = jax.lax.dynamic_slice(
          external_dynamics_params_est,
          (i, 0), # start idx
          (horizon, est_params_dim) # slice size (must use fixed values)
      )

      ############ Adaptation ##################
      # Simulate noisy magnetometer measurement
      magnetometer_measurements = jax.lax.dynamic_slice(
        external_dynamics_params + jrandom.normal(mag_key, shape=external_dynamics_params.shape) * 1e4,
        (i-history, 0), # uses history (no knowledge of future mag measurements)
        (history, params_dim),
        allow_negative_indices=False # will throw error if cond. not set up correctly which would allow it to use future measurements
      )

      dyn_params_est_adapt_slice = jax.lax.dynamic_slice(
          external_dynamics_params_est,
          (i-history, 0),
          (history, est_params_dim),
          allow_negative_indices=False
      )
      
      t_adapt_slice = jnp.linspace( (i-history)*self.dt, (i-history)*self.dt + self.dt*history, history)
      pcpf_positions = coord.pci_to_pcpf(dyn_params_est_adapt_slice, t_adapt_slice, self.dynamics.planet)
      pcpf_positions = coord.cartesian_to_spherical_4D(pcpf_positions)
      def adapt_wrapper(operand):
        return adapt_mag_model(*operand)

      def no_adapt_update(operand):
        model, *_ = operand
        return model
        
      mag_model = jax.lax.cond(
        (i % adapt_freq == 0) & (i >= history), 
        adapt_wrapper,
        no_adapt_update,
        operand=(mag_model, pcpf_positions, magnetometer_measurements)
      )
      ########################################################
 
      ############### Use Learned Model #########################
      # Convert XYZ in PCI to PCPF
      t_slice = jnp.linspace(t_current, t_current + self.dt*horizon, horizon)
      dyn_params_est_slice = coord.pci_to_pcpf(dyn_params_est_slice, t_slice, self.dynamics.planet) 
      # Convert dyn_params_est_slice (xyz pos) to PCPF -> 4D spherical
      dyn_params_est_slice = coord.cartesian_to_spherical_4D(dyn_params_est_slice)
      # Compute b-field (4D spherical -> B-field in PCPF)
      dyn_params_est_slice = jax.vmap(mag_model)(dyn_params_est_slice)
      # Convert b-field back to PCI and save as dyn_params_est_slice
      dyn_params_est_slice = coord.pcpf_to_pci(dyn_params_est_slice, t_slice, self.dynamics.planet)
      #############################################################
      
      nominal_traj, nominal_cntrl = jax.lax.cond(
        i % replan_freq == 0,
        controller_wrapper,
        no_update,
        operand=(state, target_state, cntrl_key, dyn_params_est_slice, nominal_traj, nominal_cntrl)
      )
      # control_input, cntrl_param = low_level_controller(state, target_state, cntrl_key, dyn_params_slice[i % replan_freq], nominal_traj, nominal_cntrl, cntrl_param)
      control_input, cntrl_param = low_level_controller(state, nominal_cntrl[0], nominal_traj[0], dyn_params_slice[i % replan_freq], cntrl_param)
      next_state = self.rk4_step(state, control_input, t_current, dyn_params_slice[0], rk4_key, noise_std)
      return (next_state, key, i+1, cntrl_param, nominal_traj, nominal_cntrl, mag_model), (state, control_input)
      
    # Run scan
    init_nom_traj = jnp.tile(initial_state,(high_level_controller.horizon+1,1))
    key, init_cntrl_key = jrandom.split(key)
    init_nom_cntrl = 0.001*jrandom.normal(init_cntrl_key, shape=((high_level_controller.horizon, self.dynamics.num_controls)))
    init_cntrl_param = low_level_controller.cntrl_param_init
    init_carry = (initial_state, key, 0, init_cntrl_param, init_nom_traj, init_nom_cntrl, self.dynamics.mag_model)
    
    (final_state, final_key, _, _, nominal_traj, nominal_cntrl, mag_model), (trajectory, controls) = jax.lax.scan(scan_step, init_carry, length=num_steps)
  
    # Add final state to end of trajectory
    trajectory = jnp.vstack([trajectory, final_state[None, :]])
  
    return trajectory, controls#, nominal_traj, nominal_cntrl

  @eqx.filter_jit
  def _generate_trajectory_batch_seq(self,
                                initial_states: jax.Array,
                                target_states: jax.Array,
                                key: jax.Array,
                                batch_size: int,
                                noise_std: jax.Array,
                                num_steps: int,
                                external_dynamics_params: jax.Array,
                                control_sequence: jax.Array):
    """ Batch trajectory generator without conditionals (for computational graph simplicity for jit) for multiple control sequence inputs (for use in MPPI or CEM)"""
    keys = jrandom.split(key, batch_size)
    batch_fn = jax.vmap(self._generate_trajectory_seq, in_axes=(None, None, 0, None, None, None, 0))
    return batch_fn(initial_states, target_states, keys, noise_std, num_steps, external_dynamics_params, control_sequence)

  def plot_traj(self, trajectory, controls=None, target_state=None, labels_states=None, labels_controls=None, legend=None, title=None):
    """
    Plot one or more trajectories, and optionally controls.
    Each trajectory is shaped (N+1, nx).
    Controls, if provided, are shaped (N, nu).
    labels_states and labels_controls optionally override y-axis labels.
    Defaults use self.dynamics.names_states and self.dynamics.names_controls.
    """
    # Labels (defaults to dynamics names)
    labels_states = labels_states if labels_states is not None else self.dynamics.names_states
    labels_controls = labels_controls if labels_controls is not None else (
        self.dynamics.names_controls if controls is not None else None)
    dt = self.dt
    
    plot_traj(dt, trajectory, controls, target_state, labels_states, labels_controls, legend, title)

  def plot_costs(self, trajectory=None, target_state=None, df=None, batch_size=1, N = 100, angle_threshold=15.0, omega_threshold=5.0, time_max=None, title=None, legend=None, plot_stats=False,filename=None):
    """
    Plots quaternion and angular velocity costs over the horizon.
    Supports trajectory shapes: [N, nx] or [batch, N, nx]
    """
    quat_start = self.dynamics.params.get("quat_start")
    dt = self.dt

    if df is None:
      df = save_data_to_pd_df(trajectory, target_state, filename=None, labels=legend, dt=dt, quat_start=quat_start)
      if trajectory.ndim == 2:
          trajectory = trajectory[None, None, :, :]
      elif trajectory.ndim == 3:
          trajectory = trajectory[None, :, :, :]
      
      batch_size = trajectory.shape[1]
      N = trajectory.shape[2]

    return plot_costs(df, batch_size, N, title=title, legend=legend, angle_threshold=angle_threshold, omega_threshold=omega_threshold, time_max=time_max, plot_stats=plot_stats,filename=filename)
  

  def plot_kde(self, trajectories=None, target_states=None, df=None, batch_size=1, N = 100, angle_threshold=15.0, omega_threshold=5.0, angle_stability_tol=5.0, omega_stability_tol=5.0, tail_length=100, time_hist_max = None, angle_hist_max = None, omega_hist_max = None, legend = None, title = None, verbose=False, filename=None):
    """
    Plots histograms for Slew Time, Final Quat Error, and Final Omega Error.
    """
    dt = self.dt
    quat_start = self.dynamics.params.get("quat_start")

    if df is None:
      df = save_data_to_pd_df(trajectories, target_states, filename=None, labels=legend, dt=dt, quat_start=quat_start)
      if trajectories.ndim == 2:
          trajectories = trajectories[None, None, :, :]
      elif trajectories.ndim == 3:
          trajectories = trajectories[None, :, :, :]
      
      batch_size = trajectories.shape[1]
      N = trajectories.shape[2]

    stats_df = create_stats_df(df, batch_size, N, angle_stability_tol, omega_stability_tol, tail_length, labels=None)

    if time_hist_max is None:
        time_hist_max = df["Time"].max()
    if angle_hist_max is None:
        angle_hist_max = angle_threshold
    if omega_hist_max is None:
        omega_hist_max = omega_threshold

    plot_kde(stats_df, angle_threshold, omega_threshold, time_hist_max, angle_hist_max, omega_hist_max, legend, title, verbose, filename)

  def plot_violin_and_bar(self, trajectories=None, target_states=None, df_list=None, batch_size=1, N = 100, angle_threshold=15.0, omega_threshold=5.0, angle_stability_tol=5.0, omega_stability_tol=5.0, tail_length=100, time_hist_max = None, angle_hist_max = None, omega_hist_max = None, legend = None, dataset_labels=None, title = None, verbose=False, plot_bar_by_group=False, filename=None):
    """
    Plots histograms for Slew Time, Final Quat Error, and Final Omega Error.
    """
    dt = self.dt
    quat_start = self.dynamics.params.get("quat_start")

    if df_list is None:
      df = save_data_to_pd_df(trajectories, target_states, filename=None, labels=legend, dt=dt, quat_start=quat_start)
      df_list = [df]
      if trajectories.ndim == 2:
          trajectories = trajectories[None, None, :, :]
      elif trajectories.ndim == 3:
          trajectories = trajectories[None, :, :, :]
      
      batch_size = trajectories.shape[1]
      N = trajectories.shape[2]

    stats_df_list = []
    if dataset_labels is None:
      datasets = []
    else:
      datasets = dataset_labels
      
    for i in range(len(df_list)):
      if dataset_labels is None:
        datasets.append(f"Dataset_{i+1}")
      stats_df_list.append(create_stats_df(df_list[i], batch_size, N, angle_stability_tol, omega_stability_tol, tail_length, labels=None))

    if time_hist_max is None:
        time_hist_max = df_list[0]["Time"].max()
    if angle_hist_max is None:
        angle_hist_max = angle_threshold
    if omega_hist_max is None:
        omega_hist_max = omega_threshold

    plot_violin_and_bar(stats_df_list, angle_threshold, omega_threshold, time_hist_max, angle_hist_max, omega_hist_max, datasets, title, verbose, plot_bar_by_group, filename)


  def plot_3D(self, trajectory):
    """
    Plot 3D trajectories.
    
    Parameters:
    -----------
    trajectory : jax.Array
        Trajectory data of shape (batch_size, N, 13) or (N, 13)
        where the first 3 states are position (x, y, z).
    """
    plot_3D(trajectory)


