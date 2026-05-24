import jax
import jax.numpy as jnp
import equinox as eqx

class Controller(eqx.Module):
  def __call__(self, state: jax.Array, target: jax.Array, key: jax.Array) -> jax.Array:
    raise NotImplementedError("Must define a controller if no control sequence is given")