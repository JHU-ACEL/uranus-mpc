import jax
import jax.numpy as jnp
import pdb

from dynamics.planetary_params import PlanetParams, Earth

# Class that stores coordinate system transformation functions
class coord():
  def cartesian_to_spherical_4D(states):
      # states shape: (N, 3)
      r = jnp.linalg.norm(states, axis=-1, keepdims=True)
  
      unit_vectors = states / r
      # Concatenate radius and unit vectors along the last axis
      return jnp.concatenate([r, unit_vectors], axis=-1)
  
  def spherical_4D_to_cartesian(spherical_states):
      r = spherical_states[:, :1]
      unit_vectors = spherical_states[:, 1:]
  
      # Multiply radius across the vector components
      return r * unit_vectors


  def pci_to_pcpf(pci_vec, t_seconds, planet=Earth):
    """ 
    Converts a vector of shape (N, 3) from PCI to PCPF coordinates.
    """
    
    # GST rotation
    theta_gst = planet.omega * t_seconds
    c, s = jnp.cos(theta_gst), jnp.sin(theta_gst)

    # Extract columns: x, y, z are now arrays of shape (N,)
    x = pci_vec[..., 0]
    y = pci_vec[..., 1]
    z = pci_vec[..., 2]
    
    pcpf_x =  c * x + s * y
    pcpf_y = -s * x + c * y
    pcpf_z = z

    # Stack along the last axis to return shape (N, 3)
    return jnp.stack([pcpf_x, pcpf_y, pcpf_z], axis=-1)
  
  def pcpf_to_pci(pcpf_vec, t_seconds, planet=Earth):
    """ 
    Converts a vector of shape (N, 3) from PCPF to PCI coordinates.
    """
    # GST rotation
    theta_gst = planet.omega * t_seconds
    c, s = jnp.cos(theta_gst), jnp.sin(theta_gst)
    
    x = pcpf_vec[..., 0]
    y = pcpf_vec[..., 1]
    z = pcpf_vec[..., 2]
    
    pci_x = c * x - s * y
    pci_y = s * x + c * y
    pci_z = z
    
    # Stack along the last axis to return shape (N, 3)
    return jnp.stack([pci_x, pci_y, pci_z], axis=-1)  

  def orbital_elements_to_pci(elements: jax.Array, planet=Earth):
    """
    Vectorized conversion for multiple sets of orbital elements.
    
    Parameters
    ----------
    elements : jax.Array
        Array of shape (N, 6) with columns [a, e, i, raan, omega, nu]
        a : float
        Semi-major axis (km)
        e : float
            Eccentricity (dimensionless)
        i : float
            Inclination (radians)
        raan : float
            Right Ascension of Ascending Node (radians)
        omega : float
            Argument of Periapsis (radians)
        nu : float
            True Anomaly (radians)
        
    Returns
    -------
    pos_and_vel_eci : jax.Array
        Position and velocity vectors in PCI frame (km) [N, 6]
    """
    mu = planet.mu 
    
    # Use ellipsis to support (6,) or (N, 6)
    a     = elements[..., 0]
    e     = elements[..., 1]
    i     = elements[..., 2]
    raan  = elements[..., 3]
    omega = elements[..., 4]
    nu    = elements[..., 5]
  
    # Semi-latus rectum and magnitude
    p = a * (1 - e**2)
    r_mag = p / (1 + e * jnp.cos(nu))
    
    # Position in perifocal (PQW) frame (components are shape (N,) or scalar)
    r_p = r_mag * jnp.cos(nu)
    r_q = r_mag * jnp.sin(nu)
    # r_w is 0
    
    # Velocity in perifocal (PQW) frame
    v_factor = jnp.sqrt(mu / p)
    v_p = v_factor * -jnp.sin(nu)
    v_q = v_factor * (e + jnp.cos(nu))
    # v_w is 0
    
    # Pre-calculate trig terms
    c_raan, s_raan = jnp.cos(raan), jnp.sin(raan)
    c_i, s_i       = jnp.cos(i), jnp.sin(i)
    c_om, s_om     = jnp.cos(omega), jnp.sin(omega)

    # Manual rotation PQW -> PCI coordinates   
    # Position X, Y, Z
    x_eci = (c_raan * c_om - s_raan * s_om * c_i) * r_p + \
            (-c_raan * s_om - s_raan * c_om * c_i) * r_q
            
    y_eci = (s_raan * c_om + c_raan * s_om * c_i) * r_p + \
            (-s_raan * s_om + c_raan * c_om * c_i) * r_q
            
    z_eci = (s_om * s_i) * r_p + (c_om * s_i) * r_q
    
    # Velocity X, Y, Z
    vx_eci = (c_raan * c_om - s_raan * s_om * c_i) * v_p + \
             (-c_raan * s_om - s_raan * c_om * c_i) * v_q
             
    vy_eci = (s_raan * c_om + c_raan * s_om * c_i) * v_p + \
             (-s_raan * s_om + c_raan * c_om * c_i) * v_q
             
    vz_eci = (s_om * s_i) * v_p + (c_om * s_i) * v_q
    
    # Stack along the last dimension to get (N, 6) or (6,)
    return jnp.stack([x_eci, y_eci, z_eci, vx_eci, vy_eci, vz_eci], axis=-1)
