import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jrandom

from typing import List, Callable
import pdb

# Build network with jax compatibility
class FFNet(eqx.Module):
    layers: List[eqx.nn.Linear]
    in_mean: jax.Array
    in_std: jax.Array
    out_mean: jax.Array
    out_std: jax.Array

    def __init__(self, key, input_size, output_size, width, depth, in_mean, in_std, out_mean, out_std):
        self.layers = []
        self.in_mean = in_mean
        self.in_std = in_std
        self.out_mean = out_mean
        self.out_std = out_std

        hidden_sizes = [width] * depth
        layer_sizes = [input_size, *hidden_sizes, output_size]
        
        for (fan_in, fan_out) in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey = jrandom.split(key)
            self.layers.append(
                eqx.nn.Linear(fan_in, fan_out, use_bias=True, key=subkey)
            )
            
    def __call__(self, x):
        # Normalization
        x = (x - self.in_mean.ravel())/self.in_std.ravel()
        # Step through each layer
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        x = self.layers[-1](x)
        return (x * self.out_std) + self.out_mean # unnormalize

class HyperNet(eqx.Module):
    # with help from https://medium.com/@atulit23/hypernetworks-a-novel-way-to-initialize-weights-e7584385488d
    layers: List[eqx.nn.Linear]
    in_mean: jax.Array
    in_std: jax.Array
    out_mean: jax.Array
    out_std: jax.Array

    def __init__(self, key, input_size, output_size, width, depth, in_mean, in_std, out_mean, out_std):
        self.layers = []
        self.in_mean = in_mean
        self.in_std = in_std
        self.out_mean = out_mean
        self.out_std = out_std

        hidden_sizes = [width] * depth
        layer_sizes = [input_size, *hidden_sizes, output_size]
        
        for (fan_in, fan_out) in zip(layer_sizes[:-1], layer_sizes[1:]):
            key, subkey = jrandom.split(key)
            self.layers.append(
                eqx.nn.Linear(fan_in, fan_out, use_bias=True, key=subkey)
            )
        # Create 2 different heads to output weights and biases from last hidden layer
        # key, k_w, k_b = jrandom.split(key, 3)
        # self.weight_gen = eqx.nn.Linear(current_dim, primary_weight_dim, use_bias=True, key=k_w)
        # self.bias_gen = eqx.nn.Linear(current_dim, primary_bias_dim, use_bias=True, key=k_b)
            
    def __call__(self, x):
        # Step through each layer
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        x = self.layers[-1](x)
        N = 100 # weight size
        weights = x[:N]
        biases = x[N:]
      
        return weights, biases

class GRU(eqx.Module):
    cell: eqx.nn.GRUCell

    def __init__(self, **kwargs):
        self.cell = eqx.nn.GRUCell(**kwargs)

    def __call__(self, xs):
        scan_fn = lambda state, input: (self.cell(input, state), None)
        init_state = jnp.zeros(self.cell.hidden_size)
        final_state, _ = jax.lax.scan(scan_fn, init_state, xs)
        return final_state
