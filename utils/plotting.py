import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

from dynamics.spacecraft_dynamics import q_left, q_conj

def _euler_angle_error_core(q_current, q_goal):
    """Core mathematical operations on 1D arrays of shape (4,)."""
    q_curr_inv = q_conj(q_current)
    delta_q = q_left(q_curr_inv) @ q_goal
  
    # Shortest path correction
    delta_q = jnp.where(delta_q[0] < 0.0, -delta_q, delta_q)
    
    dw, dx, dy, dz = delta_q
    
    # Convert error quaternion to Euler angles (Z-Y-X sequence)
    sinr_cosp = 2.0 * (dw * dx + dy * dz)
    cosr_cosp = 1.0 - 2.0 * (dx * dx + dy * dy)
    error_roll = jnp.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (dw * dy - dz * dx)
    sinp = jnp.clip(sinp, -1.0, 1.0)
    error_pitch = jnp.asin(sinp)

    siny_cosp = 2.0 * (dw * dz + dx * dy)
    cosy_cosp = 1.0 - 2.0 * (dy * dy + dz * dz)
    error_yaw = jnp.atan2(siny_cosp, cosy_cosp)
    
    return jnp.stack([error_roll, error_pitch, error_yaw])

# Automatically handles inputs of shapes:
# (4,), (N, 4), or (B, N, 4) by defining the signature of the core inputs
euler_angle_error = jnp.vectorize(
    _euler_angle_error_core, 
    signature='(4),(4)->(3)'
)

def plot_traj(dt, trajectory, controls, target_state, labels_states, labels_controls, legend, title):
  # Ensure batch dimension exists
  if trajectory.ndim < 3:
      trajectory = jnp.expand_dims(trajectory, 0)
  n_traj = trajectory.shape[0]
  
  # Optional controls
  if controls is not None and controls.ndim < 3:
      controls = jnp.expand_dims(controls, 0)
  
  # Colors for trajectories
  colors = plt.cm.tab10(jnp.linspace(0.0, 1.0, n_traj))
  
  num_states = trajectory.shape[2]
  num_controls = controls.shape[2] if controls is not None else 0
  
  fig, axes = plt.subplots(nrows=int(num_states + num_controls), figsize=(8, 12))
  
  # Normalize axes to a list
  try:
      iter(axes)
      axes_list = list(axes)
  except TypeError:
      axes_list = [axes]
  
  for i, ax in enumerate(axes_list):
      if i < num_states:
          for j in range(n_traj):
            if legend is None:
              ax.plot(trajectory[j, :, i], color=colors[j])
            else:
              ax.plot(trajectory[j, :, i], color=colors[j],label=legend[j])
          if target_state is not None:
            ax.plot(target_state[:,i], linestyle='--',color='k')
          if labels_states is not None and i < len(labels_states):
              ax.set_ylabel(labels_states[i])
      else:
          ci = i - num_states
          if controls is not None:
              for j in range(n_traj):
                if legend is None:
                  ax.plot(controls[j, :, ci], color=colors[j])
                else:
                  ax.plot(controls[j, :, ci], color=colors[j], label=legend[j])
              if labels_controls is not None and ci < len(labels_controls):
                  ax.set_ylabel(labels_controls[ci])
      ax.grid(True)
      if legend is not None:
        ax.legend()
  if title is not None:
    axes_list[0].set_title(title)
  plt.show()
  
def plot_costs(dt, quat_start, trajectory, target_state, title, legend):
  """
  Plots quaternion, angular velocity costs, and Euler error norms over time.
  Supports trajectory shapes: [N, nx] or [batch, N, nx]
  """
  sns.set_theme(style="whitegrid", context="talk",font_scale=.8)
  
  if trajectory.ndim == 2:
      trajectory = jnp.expand_dims(trajectory, 0)
  
  if target_state.ndim == 1:
      target_state = jnp.expand_dims(target_state, 0)
  
  num_batches = trajectory.shape[0]
  horizon_N = trajectory.shape[1]
  time_vector = jnp.arange(horizon_N) * dt

  colors = sns.color_palette("deep", n_colors=max(num_batches, 1))

  # Extract tracking metrics
  quats = trajectory[:, :, quat_start:quat_start+4]
  quat_goals = target_state[:, None, :4]
  
  quat_costs = 1.0 - jnp.abs(jnp.sum(quat_goals * quats, axis=-1))
  
  euler_errors = euler_angle_error(quats, quat_goals)
  euler_error_norm = jnp.linalg.norm(euler_errors, axis=-1)*180/jnp.pi

  omegas = trajectory[:, :, 4:]
  omega_goals = target_state[:, None, 4:]
  omega_costs = jnp.linalg.norm(omegas - omega_goals, axis=-1)*180/jnp.pi

  fig, ax = plt.subplots(2, 1, sharex=True, dpi=150)
  
  # Loop through each batch run
  for b in range(num_batches):
      # Parse legend labeling strategy
      if legend is None:
          lbl = None
      elif isinstance(legend, (list, tuple, jnp.ndarray)):
          # If a list of strings is passed, label each line explicitly
          lbl = legend[b] if b < len(legend) else f"Run {b+1}"
      else:
          # If a single string is passed, label the first line only to avoid clutter
          lbl = legend if b == 0 else None
      
      ax[0].plot(time_vector, quat_costs[b], label=lbl, color=colors[b], linewidth=2)
      #ax[0].plot(time_vector, euler_error_norm[b], label=lbl, color=colors[b], linewidth=2)
      ax[1].plot(time_vector, omega_costs[b], label=lbl, color=colors[b], linewidth=2)

  ax[0].set_ylabel("Quaternion Cost\n$1 - |q^T q_d|$")
  #ax[0].set_ylabel("$\\|\\theta_e\\|_2$ (deg)")
  ax[1].set_ylabel("$\\|\\omega\\|_2$ (deg/s)")
  ax[1].set_xlabel("Time (s)")

  if title:
      fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
  for a in ax:
      a.grid(True, linestyle="--", alpha=0.6)
      sns.despine(ax=a, left=False, bottom=False)
      
      # Only render the legend block if handles exist on that specific axis
      handles, labels = a.get_legend_handles_labels()
      if legend is not None and len(labels) > 0:
          a.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
        
  plt.tight_layout()
  plt.show()

def plot_hist(dt, quat_start, trajectories, target_states, quat_threshold, omega_threshold, bins, time_hist_max, quat_hist_max, omega_hist_max, legend, title):
    """
    Plots KDE distributions for Slew Time, Final Quat Error, and Final Omega Error.
    Supports single trajectory arrays or a stacked array of multiple groups.
    """
    sns.set_context("talk", font_scale=1.1)
    sns.set_style("whitegrid") 
    # 1. Normalize dimensions to 4D: [groups, n_trajs, time_steps, state_dim]
    if trajectories.ndim == 2:
        trajectories = trajectories[None, None, :, :]
    elif trajectories.ndim == 3:
        trajectories = trajectories[None, :, :, :]
    
    n_groups = trajectories.shape[0]

    # Normalize target_states to 3D: [groups, n_trajs, state_dim]
    if target_states.ndim == 1:
        target_states = target_states[None, None, :]
    elif target_states.ndim == 2:
        target_states = target_states[None, :, :]

    if time_hist_max is None:
        time_hist_max = trajectories.shape[2]  # Note: index 2 is time dim in 4D
    if quat_hist_max is None:
        quat_hist_max = quat_threshold
    if omega_hist_max is None:
        omega_hist_max = omega_threshold

    # 2. Process data and compile into Pandas DataFrame for Seaborn
    all_data = []
    
    for g in range(n_groups):
        trajs = trajectories[g]
        # Match targets to this group
        t_states = target_states[g] if target_states.shape[0] == n_groups else target_states[0]

        # Calculate Costs
        quats = trajs[:, :, quat_start:quat_start+4]
        quat_goals = t_states[:, None, :4]
        quat_costs = 1.0 - jnp.abs(jnp.sum(quat_goals * quats, axis=-1))

        euler_errors = euler_angle_error(quats, quat_goals)
        euler_error_norms = jnp.linalg.norm(euler_errors, axis=-1)*180/jnp.pi
        
        omegas = trajs[:, :, 4:]
        omega_goals = t_states[:, None, 4:]
        omega_norms = jnp.linalg.norm(omegas - omega_goals, axis=-1)*180/jnp.pi

        # Stability Logic
        cond = (omega_norms < omega_threshold) & (quat_costs < quat_threshold)
        mask = jnp.logical_and.accumulate(cond[:, ::-1], axis=1)[:, ::-1]
        
        stability_start_idx = jnp.argmax(mask, axis=1)
        ever_stable = jnp.any(mask, axis=1)
        slew_time = stability_start_idx * dt
        
        # Replace non-stable slew times with NaN so Seaborn ignores them
        slew_time_arr = np.array(slew_time)
        slew_time_arr[~np.array(ever_stable)] = np.nan

        if legend is None:
          group_label = f"Group {g+1}" if n_groups > 1 else "Trajectories"
        else:
          group_label = legend[g]
        
        # Append to DataFrame structure
        df_group = pd.DataFrame({
            "Group": group_label,
            "Slew Time (s)": slew_time_arr,
            "Final Quaternion Error": np.array(quat_costs[:, -1]),
            "Final Angle Error": np.array(euler_error_norms[:, -1]),
            "Final Omega Error": np.array(omega_norms[:, -1])
        })
        all_data.append(df_group)

        # Analytics Output per group
        success_rate = jnp.mean(ever_stable) * 100
        num_clipped_time = jnp.sum((slew_time > time_hist_max) & ever_stable)
        # num_clipped_euler = jnp.sum(euler_error_norms[:, -1] > euler_hist_max)
        num_clipped_quat = jnp.sum(quat_costs[:, -1] > quat_hist_max)
        num_clipped_omega = jnp.sum(omega_norms[:, -1] > omega_hist_max)

        print(f"--- {group_label} ---")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"  {num_clipped_time} successful trajs clipped from time plot")
        # print(f"  {num_clipped_euler} trajs clipped from angle plot")
        print(f"  {num_clipped_quat} trajs clipped from quat plot")
        print(f"  {num_clipped_omega} trajs clipped from omega plot\n")

    # Merge all groups into one DataFrame
    df = pd.concat(all_data, ignore_index=True)

    # 3. Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # kwargs to keep seaborn plots consistent
    kde_kwargs = {"hue": "Group", "fill": True, "common_norm": False, "warn_singular": False}

    # Slew Time (ignoring NaNs)
    sns.kdeplot(data=df, x="Slew Time (s)", ax=axs[0], clip=(0, time_hist_max), **kde_kwargs)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_title(f"{title}: Slew Time" if title else "Slew Time")
    axs[0].set_xlim(0, time_hist_max)

    # Final Quat Error
    sns.kdeplot(data=df, x="Final Quaternion Error", ax=axs[1], clip=(0, quat_hist_max), **kde_kwargs)
    axs[1].set_xlabel(r"$||q_{\text{error}}||$")
    axs[1].set_title("Final Quaternion Error")
    axs[1].set_xlim(0, quat_hist_max)

    # # Final Euler Angle Error
    # sns.kdeplot(data=df, x="Final Angle Error", ax=axs[1], clip=(0, euler_hist_max), **kde_kwargs)
    # axs[1].set_title("Final Angle Norm")
    # axs[1].set_xlabel(r"$||\theta_e||$ (deg)")
    # axs[1].set_xlim(0, euler_hist_max)

    # Final Omega Error
    sns.kdeplot(data=df, x="Final Omega Error", ax=axs[2], clip=(0, omega_hist_max), **kde_kwargs)
    axs[2].set_title("Final Angular Velocity Norm")
    axs[2].set_xlabel(r"$||\omega||$ (deg/s)")
    axs[2].set_xlim(0, omega_hist_max)


    # Histograms (uncomment to see histograms instead of kde plots)
    # sns.histplot(data=df, x="Slew Time (s)", ax=axs[0], hue="Group", kde=True, stat="count", common_norm=False)
    # sns.histplot(data=df, x="Final Angle Error", ax=axs[1], hue="Group", kde=True, stat="count", common_norm=False)
    # sns.histplot(data=df, x="Final Omega Error", ax=axs[2], hue="Group", kde=True, stat="count", common_norm=False)

    for a in axs:
        a.grid(axis='y', linestyle='--', alpha=0.4)
        
    # Clean up duplicate legends if multiple groups were plotted
    if n_groups > 1:
        legend = axs[2].get_legend()
        if legend:
            legend.set_title(None)
        if axs[0].get_legend(): axs[0].get_legend().remove()
        if axs[1].get_legend(): axs[1].get_legend().remove()
    elif n_groups == 1: # Remove the "Group" legend altogether if only 1 trajectory was passed
        for a in axs:
            if a.get_legend(): a.get_legend().remove()
    plt.tight_layout()
    plt.show()

def plot_3D(trajectory):
  """
  Plot 3D trajectories.
  
  Parameters:
  -----------
  trajectory : jax.Array
      Trajectory data of shape (batch_size, N, 13) or (N, 13)
      where the first 3 states are position (x, y, z).
  """
  # Handle single trajectory case (N, 13) -> (1, N, 13)
  if (trajectory.ndim < 3):
      trajectory = jnp.expand_dims(trajectory, 0)
  n_traj = trajectory.shape[0]
  
  # Extract positions (first 3 states)
  positions = trajectory[:, :, :3]  # Shape: (batch_size, N, 3)
  
  # Create 3D figure
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')
  
  
  # Color map for multiple trajectories
  colors = plt.cm.tab10(jnp.linspace(0, 1, n_traj))
  
  # Plot each trajectory
  for i in range(n_traj):
      # Convert to numpy for matplotlib (matplotlib doesn't accept jax arrays directly)
      x = jnp.asarray(positions[i, :, 0])
      y = jnp.asarray(positions[i, :, 1])
      z = jnp.asarray(positions[i, :, 2])
      
      # Plot the orbit line
      ax.plot(x, y, z, color=colors[i], label=f'Trajectory {i+1}', linewidth=1.5)
      
      # Mark start point (green) and end point (red)
      ax.scatter(x[0], y[0], z[0], color='green', s=50, marker='o', edgecolors='black')
      ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, marker='s', edgecolors='black')
  
  # Plot origin (central body)
  #ax.scatter(0, 0, 0, color='yellow', s=50, marker='o', edgecolors='orange', label='Central Body')
  
  # Labels and title
  ax.set_xlabel('X [km]')
  ax.set_ylabel('Y [km]')
  ax.set_zlabel('Z [km]')
  ax.set_title('Orbit (ECI coords)')
  
  # Add legend if multiple trajectories
  # if n_traj > 1:
  #     ax.legend()
  
  # Equal aspect ratio
  max_range = float(jnp.max(jnp.abs(positions))) * 1.1
  ax.set_xlim([-max_range, max_range])
  ax.set_ylim([-max_range, max_range])
  ax.set_zlim([-max_range, max_range])
  
  plt.tight_layout()
  plt.show()