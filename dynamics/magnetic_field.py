import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from functools import partial
import equinox as eqx

import pdb

from .planetary_params import PlanetParams
from utils.coord_transforms import coord

# --- Main Class ---
class MagneticFieldModel(eqx.Module):
  planet: PlanetParams

  # From IGRF code: https://github.com/IAGA-VMOD/ppigrf/blob/main/src/ppigrf/ppigrf.py, modified for jax
  def get_legendre(self, theta):
      nmax = self.planet.nmax
      theta_rad = jnp.radians(theta).flatten()
      sinth = jnp.sin(theta_rad)
      costh = jnp.cos(theta_rad)
      
      dims = (nmax + 1, nmax + 1, theta_rad.shape[0])
      
      def inner_loop(m, carry):
          n, P, dP, S = carry
          
          # Diagonal case: n == m
          P_diag = sinth * P[n - 1, m - 1]
          dP_diag = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
          
          # Off-diagonal case: n != m
          Knm = jnp.where(n > 1, 
                          ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3)), 
                          0.0)
          P_off = jnp.where(n == 1,
                            costh * P[n - 1, m],
                            costh * P[n - 1, m] - Knm * P[n - 2, m])
          dP_off = jnp.where(n == 1,
                             costh * dP[n - 1, m] - sinth * P[n - 1, m],
                             costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m])
          
          # Select based on m == n
          is_diag = (m == n)
          P_val = jnp.where(is_diag, P_diag, P_off)
          dP_val = jnp.where(is_diag, dP_diag, dP_off)
          
          P = P.at[n, m].set(P_val)
          dP = dP.at[n, m].set(dP_val)
          
          # Schmidt normalization
          m_idx_term = jnp.where(m == 1, 2.0, 1.0)
          S_val = jnp.where(m == 0,
                           S[n - 1, 0] * (2. * n - 1) / n,
                           S[n, m - 1] * jnp.sqrt((n - m + 1) * m_idx_term / (n + m)))
          S = S.at[n, m].set(S_val)
          
          return (n, P, dP, S)
      
      def outer_loop(n, carry):
          P, dP, S = carry
          _, P, dP, S = jax.lax.fori_loop(0, n + 1, inner_loop, (n, P, dP, S))
          return (P, dP, S)
      
      P = jnp.zeros(dims).at[0, 0].set(jnp.ones(theta_rad.shape[0]))
      dP = jnp.zeros(dims)
      S = jnp.zeros((nmax + 1, nmax + 1)).at[0, 0].set(1.0)
      
      P, dP, S = jax.lax.fori_loop(1, nmax + 1, outer_loop, (P, dP, S))
      
      # Apply normalization
      P = P * S[:, :, None]
      dP = dP * S[:, :, None]
      
      return P, dP

  def compute_b_pcpf(self, r_pcpf):
    """
    Calculates B field in Planet-Centered Planet-Fixed for a batch of points.
    Input r_pcpf: (N, 3) in km
    Output: (N, 3) in nT
    """
    # 1. Coordinate conversion
    # r, theta_rad, and phi_rad will all be (N, 1)
    r_pcpf = jnp.atleast_2d(r_pcpf)
    r = jnp.linalg.norm(r_pcpf, axis=-1, keepdims=True) 
    theta_rad = jnp.arccos(r_pcpf[:, 2:3] / r)          
    phi_rad = jnp.arctan2(r_pcpf[:, 1:2], r_pcpf[:, 0:1]) 
    
    # Flatten theta for the Legendre function grid lookup
    theta_deg_arr = jnp.degrees(theta_rad).flatten() 
    
    # 2. Get Legendre (P_grid, dP_grid shape: [nmax+1, nmax+1, N])
    P_grid, dP_grid = self.get_legendre(theta_deg_arr)
    
    # 3. Extract active coefficients
    ns = self.planet.keys[:, 0] # (K,)
    ms = self.planet.keys[:, 1] # (K,)
    
    # Pulling (K, N) slices from the full grid
    P = P_grid[ns, ms, :] 
    dP = dP_grid[ns, ms, :]

    # 4. Trig and Radial terms
    # Resulting shapes: (K, N)
    m_phi = (phi_rad @ ms[None, :]).T 
    cosmphi = jnp.cos(m_phi)
    sinmphi = jnp.sin(m_phi)
    
    # Broadcasting semi-major axis ratio: (K, 1) / (1, N) -> (K, N)
    radial_term = (self.planet.radius / r.T) ** (ns[:, None] + 2)
    
    # 5. Gradients (Summing over the K dimension to get N results)
    g = self.planet.g[:, None] # (K, 1)
    h = self.planet.h[:, None] # (K, 1)

    # Br: (N,)
    Br = jnp.sum((ns[:, None] + 1) * radial_term * (g * cosmphi + h * sinmphi) * P, axis=0)
    
    # Btheta: (N,)
    Btheta = jnp.sum(-1.0 * radial_term * (g * cosmphi + h * sinmphi) * dP, axis=0)
    
    # Bphi: (N,)
    Bp_terms = -1.0 * radial_term * (g * (-sinmphi * ms[:, None]) + h * (cosmphi * ms[:, None])) * P
    sin_theta = jnp.sin(theta_rad).flatten()
    
    # Use jnp.where for a jit-friendly singularity check
    Bphi = jnp.where(
        jnp.abs(sin_theta) > 1e-10, 
        jnp.sum(Bp_terms, axis=0) / sin_theta, 
        0.0
    )
    
    # 6. Rotate to Cartesian (N,)
    cos_theta = jnp.cos(theta_rad).flatten()
    sin_phi = jnp.sin(phi_rad).flatten()
    cos_phi = jnp.cos(phi_rad).flatten()
    
    bx = Br * (sin_theta * cos_phi) + Btheta * (cos_theta * cos_phi) + Bphi * (-sin_phi)
    by = Br * (sin_theta * sin_phi) + Btheta * (cos_theta * sin_phi) + Bphi * (cos_phi)
    bz = Br * (cos_theta)           + Btheta * (-sin_theta)
    
    return jnp.stack([bx, by, bz], axis=-1)

  def compute_b_pci(self, r_pci, t_seconds):
    """
    Calculates B field in Inertial Frame (Input: km, s, Output: nT)
    """
    r_pcpf = coord.pci_to_pcpf(r_pci, t_seconds, self.planet)
    
    # Get Field in PCPF
    b_pcpf = self.compute_b_pcpf(r_pcpf)

    # Convert to PCI
    return coord.pcpf_to_pci(b_pcpf, t_seconds, self.planet)

  @eqx.filter_jit
  def generate_magnetometer_data(self, trajectory, dt, num_steps, key, noise_std, bias, batch_size=0):
    '''
    Generates magnetic field data in PCI coords [nT] with optional noise
    '''
    if trajectory.ndim == 2:
      batched_traj = trajectory[jnp.newaxis, :, :]
    else:
      batched_traj = trajectory

    if batch_size==0:
       batch_size = n_traj

    n_traj = batched_traj.shape[0]
    if batch_size==0:
       batch_size = n_traj
    #batch_keys = jrandom.split(key, n_traj)
    t = jnp.linspace(0, dt * num_steps, num_steps + 1)
    key, subkey = jrandom.split(key) # ensure key is not resused
    noise = jrandom.normal(subkey, shape=(n_traj,num_steps+1,3)) * noise_std
    bias = jnp.broadcast_to(bias, (num_steps+1, 3))

    def wrapper(args):
      traj, noise = args
      return self.compute_b_pci(traj,t) + noise + bias

    mag_data = jax.lax.map(wrapper, (batched_traj, noise),batch_size=batch_size)

    return mag_data #mag_data.reshape(trajectory.shape[:-1] + (-1,))
    
    