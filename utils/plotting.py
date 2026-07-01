import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
import seaborn as sns
import pdb

from dynamics.spacecraft_dynamics import q_left, q_conj

def save_data_to_pd_df(trajectories, target_states, filename=None, labels=None, dt=0.1, quat_start=0):
  """ Save results from multiple different Monte Carlo sims to one Pandas dataframe """

  # 1. Normalize dimensions to 4D: [groups, n_trajs, time_steps, state_dim]
  if trajectories.ndim == 2:
      trajectories = trajectories[None, None, :, :]
  elif trajectories.ndim == 3:
      trajectories = trajectories[None, :, :, :]
  
  n_groups = trajectories.shape[0]
  batch_size = trajectories.shape[1]
  n_steps = trajectories.shape[2]

  # horizon_N = trajectory.shape[1]
  time = jnp.arange(n_steps) * dt
  time_expanded = jnp.broadcast_to(time, (batch_size,n_steps))

  # Normalize target_states to 3D: [groups, n_trajs, state_dim]
  if target_states.ndim == 1:
      target_states = target_states[None, None, :]
  elif target_states.ndim == 2:
      target_states = target_states[None, :, :]

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

      principal_rotation_angles = 2*jnp.arccos(jnp.abs(jnp.sum(quat_goals * quats, axis=-1)))*180/jnp.pi
      
      omegas = trajs[:, :, 4:]
      omega_goals = t_states[:, None, 4:]
      omega_norms = jnp.linalg.norm(omegas - omega_goals, axis=-1)*180/jnp.pi

      if labels is None:
        group_label = f"Group_{g+1}" if n_groups > 1 else "Trajectories"
      else:
        group_label = labels[g]
      
      # Append to DataFrame structure
      df_group = pd.DataFrame({
          "Group": group_label,
          "Time": np.array(time_expanded.ravel()),
          "Quaternion Errors": np.array(quat_costs.ravel()),
          "Angle Errors": np.array(principal_rotation_angles.ravel()),
          "Omega Errors": np.array(omega_norms.ravel()),
      })
      all_data.append(df_group)
      
  # Merge all groups into one DataFrame
  df = pd.concat(all_data, ignore_index=True)
  if filename is not None:
    df.to_pickle('data/'+filename)
  
  return df

def create_stats_df(df, batch_size, N, angle_tol, omega_tol, tail_length, labels=None):
  n_groups = df["Group"].nunique()

  # 2. Process data and compile into Pandas DataFrame for Seaborn
  all_data = []
  
  for g in range(n_groups):
    group = df["Group"].unique()[g]
    time_vector = jnp.array(df.loc[df["Group"] == group]["Time"]).reshape((batch_size, N))
    time_vector = time_vector[0]
    dt = time_vector[1] - time_vector[0]
    quat_costs = jnp.array(df.loc[df["Group"] == group]["Quaternion Errors"]).reshape((batch_size, N))
    principal_rotation_angles = jnp.array(df.loc[df["Group"] == group]["Angle Errors"]).reshape((batch_size, N))
    omega_norms = jnp.array(df.loc[df["Group"] == group]["Omega Errors"]).reshape((batch_size, N))
    # Extract the "converged" trajs using the mean of the tail
    final_omegas = jnp.mean(omega_norms[:, -tail_length:], axis=1, keepdims=True)
    final_angles = jnp.mean(principal_rotation_angles[:, -tail_length:], axis=1, keepdims=True)
    
    # 3. Condition: Is the distance from the *average* final value within the neighborhood?
    cond = (jnp.abs(omega_norms - final_omegas) < omega_tol) & \
           (jnp.abs(principal_rotation_angles - final_angles) < angle_tol)
    
    # 4. Backward accumulation to find where it *stays* within the neighborhood
    mask = jnp.logical_and.accumulate(cond[:, ::-1], axis=1)[:, ::-1]
    
    # 5. Calculate indices and stability checks
    stability_start_idx = jnp.argmax(mask, axis=1)
    
    # We use the tail_length as our minimum required stability steps to ensure 
    # the sequence actually settled around this average for a meaningful duration.
    min_stable_steps = tail_length 
    ever_stable = jnp.sum(mask, axis=1) >= min_stable_steps
    
    slew_time = stability_start_idx * dt
    
    # 6. Apply NaNs for Seaborn plotting
    slew_time_arr = np.array(slew_time, dtype=float) 
    slew_time_arr[~np.array(ever_stable)] = np.nan

    if labels is None:
      group_label = group
    else:
      group_label = labels[g]
    
    # Append to DataFrame structure
    df_group = pd.DataFrame({
      "Group": group_label,
      "Slew Time": slew_time_arr,
      "Final Quaternion Error": np.array(quat_costs[:, -1]),
      "Final Angle Error": np.array(final_angles[:, -1]),
      # "Final Angle Error": np.array(principal_rotation_angles[:, -1]),
      "Final Omega Error": np.array(final_omegas[:, -1]),
    })
    all_data.append(df_group)

  # Merge all groups into one DataFrame
  stats_df = pd.concat(all_data, ignore_index=True)
  
  return stats_df

def plot_traj(dt, trajectory, controls, target_state, labels_states, labels_controls, legend, title):
  # Ensure batch dimension exists
  if trajectory.ndim < 3:
      trajectory = jnp.expand_dims(trajectory, 0)
  n_traj = trajectory.shape[0]

  if target_state is not None:
    if target_state.ndim < 2:
      target_state = jnp.tile(target_state,(trajectory.shape[1],1))
    
  # Optional controls
  if controls is not None and controls.ndim < 3:
      controls = jnp.expand_dims(controls, 0)
  
  # Colors for trajectories
  # colors = plt.cm.tab10(jnp.linspace(0.0, 1.0, n_traj))
  colors = sns.color_palette("deep", n_colors=max(n_traj, 1))
  
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

def plot_costs(df, batch_size, N, title, legend, angle_threshold, omega_threshold, time_max, plot_stats, filename):
    """
    Plots aggregate statistics (Max, 95th, 50th, Mean) or all costs for all trajectories
    Supports trajectory shapes: [N, nx] or [batch, N, nx]
    """
    n_groups = df["Group"].nunique()

    for g in range(n_groups):
        group = df["Group"].unique()[g]
        time_vector = jnp.array(df.loc[df["Group"] == group]["Time"]).reshape((batch_size, N))
        time_vector = time_vector[0]
        quat_costs = jnp.array(df.loc[df["Group"] == group]["Quaternion Errors"]).reshape((batch_size, N))
        angle_costs = jnp.array(df.loc[df["Group"] == group]["Angle Errors"]).reshape((batch_size, N))
        omega_costs = jnp.array(df.loc[df["Group"] == group]["Omega Errors"]).reshape((batch_size, N))
      
        # Define shared labels and configurations to decouple the loop
        metrics_info = [
            {"data": angle_costs, "threshold": angle_threshold, "ylabel": "Angle (deg)", "suffix": "_angle"},
            {"data": omega_costs, "threshold": omega_threshold, "ylabel": r"$\|\omega\|_2$ (deg/s)", "suffix": "_omega"}
        ]

        for idx, m in enumerate(metrics_info):
            # Create a completely separate figure for each metric
            fig, a = plt.subplots(figsize=(8, 4), dpi=150)
            
            if plot_stats:
                blues_palette = sns.light_palette("steelblue", n_colors=4, reverse=False)
                c_max = blues_palette[1]  
                c_95  = blues_palette[2]  
                c_50  = blues_palette[3]  
                c_avg = "red"             
                
                data = m["data"]
                mx = jnp.max(data, axis=0)
                p95 = jnp.percentile(data, 95, axis=0)
                p50 = jnp.percentile(data, 50, axis=0)
                avg = jnp.mean(data, axis=0)
                
                a.fill_between(time_vector, 0, mx, color=c_max, alpha=0.3, label="Max error")
                a.fill_between(time_vector, 0, p95, color=c_95, alpha=0.4, label="95th percentile")
                a.fill_between(time_vector, 0, p50, color=c_50, alpha=0.2, label="50th percentile")
                
                a.plot(time_vector, mx, color=c_max, linewidth=1.5)
                a.plot(time_vector, p95, color=c_95, linewidth=2.0)
                a.plot(time_vector, p50, color=c_50, linewidth=2.5)
                a.plot(time_vector, avg, color=c_avg, linewidth=2.5, label="Average")
                
                a.axhline(y=m["threshold"], color="black", linestyle="--", label=r"$\text{threshold}$")
                
            else:
                colors = sns.color_palette("deep", n_colors=max(batch_size, 1))
                for b in range(batch_size):
                    lbl = legend[b] if isinstance(legend, (list, tuple, jnp.ndarray)) and b < len(legend) else (legend if b == 0 else None)
                    a.plot(time_vector, m["data"][b], label=lbl, color=colors[b], linewidth=2)
        
            # Styling and housekeeping per independent plot
            a.set_ylabel(m["ylabel"])
            a.set_xlabel("Time (seconds)")
        
            if title:
                a.set_title(title, fontsize=14, fontweight='bold', y=1.02)
              
            a.grid(True, linestyle="-", alpha=0.15)
            sns.despine(ax=a)
            if time_max is None:
              a.set_xlim([time_vector[0], time_vector[-1]])
            else:
              a.set_xlim([time_vector[0], time_max])
            a.set_ylim(bottom=0)
            
            handles, labels = a.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if len(by_label) > 0:
                a.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True, framealpha=0.95, edgecolor="black")
            
            plt.tight_layout()
            if filename is not None:
              plt.savefig('figures/' + 'costs_'+ group.lower() + m["suffix"] +  '_' + filename)
            plt.show()

    return df

def plot_kde(stats_df, angle_threshold, omega_threshold, time_hist_max, angle_hist_max, omega_hist_max, legend, title, verbose, filename):
    """
    Plots KDE distributions for Slew Time, Final Quat Error, and Final Omega Error as individual figures.
    Supports single trajectory arrays or a stacked array of multiple groups.
    """
    n_groups = stats_df["Group"].nunique()
    kde_kwargs = {"hue": "Group", "fill": True, "common_norm": False, "warn_singular": False, "alpha": 0.5}
    
    # --- Figure 1: Slew Time ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.kdeplot(data=stats_df, x="Slew Time", ax=ax1, legend=False, clip=(0, time_hist_max), **kde_kwargs)
    ax1.set_xlabel("Time (s)")
    #ax1.set_title(f"{title}: Slew Time" if title else "Slew Time")
    ax1.set_xlim(0, time_hist_max)
    plt.tight_layout()
    if filename is not None:
        plt.savefig('figures/kde_slew_time_' + filename)
    plt.show()

    # --- Figure 2: Final Principal Angle Error ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.kdeplot(data=stats_df, x="Final Angle Error", ax=ax2, legend=False, clip=(0, angle_hist_max), **kde_kwargs)
    #ax2.set_title("Final Principal Angle Error")
    ax2.set_xlabel(r"$\gamma_e$ (deg)")
    ax2.set_xlim(0, angle_hist_max)
    plt.tight_layout()
    if filename is not None:
        plt.savefig('figures/kde_angle_error_' + filename)
    plt.show()

    # --- Figure 3: Final Omega Error ---
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.kdeplot(data=stats_df, x="Final Omega Error", ax=ax3, clip=(0, omega_hist_max), **kde_kwargs)
    #ax3.set_title("Final Angular Velocity Norm")
    ax3.set_xlabel(r"$||\omega||$ (deg/s)")
    ax3.set_xlim(0, omega_hist_max)
    plt.tight_layout()
    if filename is not None:
        plt.savefig('figures/kde_omega_error_' + filename)
    plt.show()

    if verbose:
        for g in range(n_groups):
            group_label = stats_df["Group"].unique()[g]
            final_angles = jnp.array(stats_df.loc[stats_df["Group"] == group_label]["Final Angle Error"])
            final_omegas = jnp.array(stats_df.loc[stats_df["Group"] == group_label]["Final Omega Error"])
            slew_time = jnp.array(stats_df.loc[stats_df["Group"] == group_label]["Slew Time"])
            stable_trajs = ~jnp.isnan(slew_time)

            successful_trajs = ((final_angles < angle_threshold) & (final_omegas < omega_threshold) & (stable_trajs))
            stable_rate = jnp.mean(stable_trajs)*100
            success_rate = jnp.mean(successful_trajs) * 100
            print(f"--- {group_label} ---")
            print(f"Success Rate: {success_rate:.2f}%")
            print(f"Stable Rate: {stable_rate:.2f}%")
            print(f"Mean Converged Angle Error: {jnp.mean(final_angles):.3f} deg")
            print(f"Mean Final Angular Velocity: {jnp.mean(final_omegas):.3f} deg/s")

def plot_violin_and_bar(stats_df_list, angle_threshold, omega_threshold, time_hist_max, angle_hist_max, omega_hist_max, dataset_labels, title, verbose, plot_bar_by_group, filename):
    """
    Plots Violin distributions for Slew Time, Final Quat Error, and Final Omega Error as individual figures.
    Supports single trajectory arrays or a stacked array of multiple groups.
    """
    n_dfs = len(stats_df_list)
    df_combined = pd.concat(stats_df_list, keys=dataset_labels)
    stats_df = df_combined.reset_index()
    stats_df = stats_df.rename(columns={'level_0': 'Dataset'}).drop(columns=['level_1'], errors='ignore')
      
    n_groups = stats_df["Group"].nunique()

    if n_dfs > 1:
        violin_kwargs = {"x": "Group", "hue": "Dataset", "cut": 0, "legend": True}#, "inner": "quartile"} 
    else:
        violin_kwargs = {"x": "Group", "hue": "Group", "cut": 0, "legend": False}#, "inner": "quartile"}

    # Helper function to reduce styling redundancies across independent violins
    def finalize_violin_axes(ax):
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if n_groups == 1:
            ax.set_xticks([])

    # --- Figure 1: Slew Time ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=stats_df, y="Slew Time", ax=ax1, **violin_kwargs)
    ax1.set_ylabel("Time (s)")
    ax1.set_xlabel("")
    ax1.set_ylim(0, time_hist_max)
    #ax1.set_title(f"{title}: Slew Time" if title else "Slew Time")
    if n_dfs > 1: ax1.get_legend().set_title("")
    finalize_violin_axes(ax1)
    plt.tight_layout()
    if filename is not None: plt.savefig('figures/violin_slew_' + filename)
    plt.show()

    # --- Figure 2: Final Principal Angle Error ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=stats_df, y="Final Angle Error", ax=ax2, **violin_kwargs)
    #ax2.set_title("Final Principal Angle Error")
    ax2.set_ylabel(r"$\gamma_e$ (deg)")
    ax2.set_xlabel("")
    ax2.set_ylim(0, angle_hist_max)
    if n_dfs > 1: ax2.get_legend().set_title("")
    finalize_violin_axes(ax2)
    plt.tight_layout()
    if filename is not None: plt.savefig('figures/violin_angle_' + filename)
    plt.show()

    # --- Figure 3: Final Omega Error ---
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=stats_df, y="Final Omega Error", ax=ax3, **violin_kwargs)
    #ax3.set_title("Final Angular Velocity Norm")
    ax3.set_ylabel(r"$||\omega||$ (deg/s)")
    ax3.set_xlabel("")
    ax3.set_ylim(0, omega_hist_max)
    if n_dfs > 1: ax3.get_legend().set_title("")
    finalize_violin_axes(ax3)
    plt.tight_layout()
    if filename is not None: plt.savefig('figures/violin_omega_' + filename)
    plt.show()

    # --- Collect Data for Success Rate & Mean Plots ---
    success_data = []
    grouped = stats_df.groupby(["Dataset", "Group"], sort=False)
    for (dataset_label, group_label), group_data in grouped:
        final_angles = jnp.array(group_data["Final Angle Error"].values)
        final_omegas = jnp.array(group_data["Final Omega Error"].values)
        slew_time    = jnp.array(group_data["Slew Time"].values)
        stable_trajs = ~jnp.isnan(slew_time)

        successful_trajs = ((final_angles < angle_threshold) & (final_omegas < omega_threshold) & (stable_trajs))
        stable_rate = jnp.mean(stable_trajs) * 100
        success_rate = jnp.mean(successful_trajs) * 100
        
        mean_angle_error = jnp.mean(final_angles)
        mean_omega_error = jnp.mean(final_omegas)
        
        # Store metrics for plotting
        success_data.append({
            "Dataset": dataset_label, 
            "Group": group_label, 
            "Success Rate": float(success_rate),
            "Mean Angle Error": float(mean_angle_error),
            "Mean Omega Error": float(mean_omega_error)
        })

        if verbose:
          print(f" ---- {dataset_label} | {group_label} ---- ")
          print(f"Success Rate: {success_rate:.2f}%")
          print(f"Stable Rate: {stable_rate:.2f}%")
          print(f"Mean Converged Angle Error: {mean_angle_error:.3f} deg")
          print(f"Mean Final Angular Velocity: {mean_omega_error:.3f} deg/s")
          print("")

    # --- Convert collected stats to DataFrame ---
    success_df = pd.DataFrame(success_data)
    bar_kwargs = {"x": "Group", "hue": "Dataset" if n_dfs > 1 else "Group", "legend": n_dfs > 1}

    # --- Figure 4: Success Rate Bar Plot ---
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=success_df, y="Success Rate", ax=ax4, **bar_kwargs)
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_xlabel("")
    ax4.set_ylim(0, 100)
    if n_dfs > 1: ax4.get_legend().set_title("")
    finalize_violin_axes(ax4)
    plt.tight_layout()
    if filename is not None: plt.savefig('figures/bar_success_' + filename)
    plt.show()

    if plot_bar_by_group:
      # --- Figure 5: Success Rate Bar Plot by Group ---
      # --- Individual Group Success Rate Bar Plots ---
      for target_group in success_df["Group"].unique():
          group_df = success_df[success_df["Group"] == target_group]
          
          fig_g, ax_g = plt.subplots(figsize=(10, 5))
          
          # Keep y and hue the same; x is just the single group
          sns.barplot(
              data=group_df, 
              x="Group", 
              y="Success Rate", 
              hue="Dataset" if n_dfs > 1 else "Group", 
              ax=ax_g
          )
          
          ax_g.set_ylabel("Success Rate (%)")
          ax_g.set_xlabel("")
          ax_g.set_xticks([]) # Removes the single x-tick label for a cleaner look
          ax_g.set_ylim(0, 100)
          #ax_g.set_title(f"Success Rate: {target_group}")
          
          if n_dfs > 1: 
              ax_g.get_legend().set_title("")
              
          finalize_violin_axes(ax_g)
          plt.tight_layout()
          
          if filename is not None: 
              plt.savefig(f'figures/bar_success_{target_group}_' + filename)
          plt.show()

    # # --- Figure 5: Mean Final Angle Error Bar Plot ---
    # fig5, ax5 = plt.subplots(figsize=(10, 5))
    # sns.barplot(data=success_df, y="Mean Angle Error", ax=ax5, **bar_kwargs)
    # ax5.set_ylabel(r"Mean $\gamma_e$ (deg)")
    # ax5.set_xlabel("")
    # if n_dfs > 1: ax5.get_legend().set_title("")
    # finalize_violin_axes(ax5)
    # plt.tight_layout()
    # if filename is not None: plt.savefig('figures/bar_angle_error_' + filename)
    # plt.show()

    # # --- Figure 6: Mean Final Omega Error Bar Plot ---
    # fig6, ax6 = plt.subplots(figsize=(10, 5))
    # sns.barplot(data=success_df, y="Mean Omega Error", ax=ax6, **bar_kwargs)
    # ax6.set_ylabel(r"Mean $||\omega||$ (deg/s)")
    # ax6.set_xlabel("")
    # if n_dfs > 1: ax6.get_legend().set_title("")
    # finalize_violin_axes(ax6)
    # plt.tight_layout()
    # if filename is not None: plt.savefig('figures/bar_omega_error_' + filename)
    # plt.show()
  
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