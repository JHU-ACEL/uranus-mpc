import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

class PlanetParams(eqx.Module):
    """Class to store planetary parameters"""
    name: str
    mu: float          # gravitational param [km^3/s^2]
    J2: float          # oblateness coeff.
    radius: float      # [km]
    omega: float       # angular velocity [rad/s]
    keys: jax.Array    # Gauss coefficient indices [n, m]
    g: jax.Array       # Gauss coefficients [nT]
    h: jax.Array       # Gauss coefficients [nT]
    nmax: int          # highest degree

    def __init__(self, name: str, mu: float, J2: float, radius: float, omega: float, gauss_coeff: dict, nmax: int = None):
        keys_list = list(gauss_coeff.keys())
        g_list = [gauss_coeff[k][0] for k in keys_list]
        h_list = [gauss_coeff[k][1] for k in keys_list]
        if nmax is None:
          nmax = max(k[0] for k in keys_list)
        
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "J2", J2)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "keys", jnp.array(keys_list))
        object.__setattr__(self, "g", jnp.array(g_list))
        object.__setattr__(self, "h", jnp.array(h_list))
        object.__setattr__(self, "nmax", nmax)


# Planet data definitions
_PLANET_DATA = {
    "earth": {
        "mu": 398600.4418,
        "J2": 1.08263e-3,
        "radius": 6371.2,
        "omega": 7.292115e-5,
        "gauss_coeff": {
            (1, 0): (-29351.8, 0.0),
            (1, 1): (-1410.8, 4545.4),
            (2, 0): (-2556.6, 0.0),
            (2, 1): (2951.1, -3133.6),
            (2, 2): (1649.3, -814.5),
            (3, 0): (1361.0, 0.0),
            (3, 1): (-2404.1, -56.6),
            (3, 2): (1243.8, 237.4),
            (3, 3): (453.6, -549.3),
            (4, 0): (895.0, 0.0),
            (4, 1): (799.5, 283.3),
            (4, 2): (55.7, -415.4),
            (4, 3): (-281.1, 13.6),
            (4, 4): (-242.1, 110.0),
            (5, 0): (-231.4, 0.0),
            (5, 1): (327.7, 44.1),
            (5, 2): (208.5, 188.2),
            (5, 3): (-122.0, -141.2),
            (5, 4): (-13.4, -115.2),
            (5, 5): (52.8, -46.2),
            (6, 0): (63.2, 0.0),
            (6, 1): (58.7, -25.4),
            (6, 2): (1.0, 109.6),
            (6, 3): (-95.3, -32.3),
            (6, 4): (9.7, 14.9),
            (6, 5): (68.0, 49.0),
            (6, 6): (-20.3, -19.6),
            (7, 0): (77.5, 0.0),
            (7, 1): (-73.6, -25.5),
            (7, 2): (1.6, 7.4),
            (7, 3): (28.0, 5.5),
            (7, 4): (5.8, -25.1),
            (7, 5): (-3.6, 11.3),
            (7, 6): (18.3, 2.2),
            (7, 7): (-18.5, -12.1),
            (8, 0): (21.6, 0.0),
            (8, 1): (10.6, 21.0),
            (8, 2): (-0.4, 9.9),
            (8, 3): (-11.5, -19.4),
            (8, 4): (-5.3, 5.3),
            (8, 5): (10.3, -17.5),
            (8, 6): (-1.4, 4.2),
            (8, 7): (-0.4, 0.3),
            (8, 8): (3.3, 3.8),
        },
    },
    "uranus": {
        "mu": 5793939.0,
        "J2": 3343.43e-6,
        "radius": 25559.0,
        "omega": 1.012e-4,
        "gauss_coeff": {
            (1, 0): (11855.0, 0.0),
            (1, 1): (11507.0, -15812.0),
            (2, 0): (-5877.0, 0.0),
            (2, 1): (-13085.0, 5851.0),
            (2, 2): (-605.0, 4185.0),
            (3, 0): (4183.0, 0.0),
            (3, 1): (-1336.0, -5817.0),
            (3, 2): (-6776.0, -357.0),
            (3, 3): (-4021.0, -2265.0),
        },
    },
    "neptune": {
        "mu": 6836529.0,
        "J2": 3411.0e-6,
        "radius": 24764.0,
        "omega": 1.083e-4,
        "gauss_coeff": {
            (1, 0): (10336.0, 0.0),
            (1, 1): (3359.0, -9772.0),
            (2, 0): (8566.0, 0.0),
            (2, 1): (-406.0, 11139.0),
            (2, 2): (4644.0, -743.0),
            (3, 0): (-5749.0, 0.0),
            (3, 1): (11632.0, -3905.0),
            (3, 2): (-1889.0, 903.0),
            (3, 3): (-2920.0, -245.0),
        },
    },
}

def get_planet(name: str, nmax: int = None) -> PlanetParams:
    """Get planetary parameters by name"""
    name_lower = name.lower()
    if name_lower not in _PLANET_DATA:
        raise ValueError(f"Unknown planet: {name}. Available: {list(_PLANET_DATA.keys())}")
    data = _PLANET_DATA[name_lower]
    return PlanetParams(
        name=name_lower,
        mu=data["mu"],
        J2=data["J2"],
        radius=data["radius"],
        omega=data["omega"],
        gauss_coeff=data["gauss_coeff"],
        nmax=nmax
    )


# Pre-instantiate common planets for convenience
Earth = get_planet("earth")
Uranus = get_planet("uranus")
Neptune = get_planet("neptune")
Earth_low_order = get_planet("earth", nmax=1)
Uranus_low_order = get_planet("uranus", nmax=1)