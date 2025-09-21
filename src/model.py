from flax import nnx
import jax.numpy as jnp
from jax import random
class MLP(nnx.Module):

    def __init__(self, rngs, input_dim, hidden_dims, output_dim, dropout_rate, activation):

        self.activation = activation
        self.layers = []
        prev_dim = input_dim
        for i, hdim in enumerate(hidden_dims):
            layer = nnx.Linear(prev_dim, hdim, rngs=rngs)
            self.layers.append(layer)
            prev_dim = hdim

        self.output_layer = nnx.Linear(prev_dim, output_dim, rngs=rngs)

        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, deterministic=False):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            if not deterministic:
                x = self.dropout(x)

        x = self.output_layer(x)
        return x

        
class SDE(MLP):

    def generate(self, source_gen, n_samples, key, dt=0.01):
        '''
        Generate samples from the learned SDE model starting from source_gen samples.

        Args:
            source_gen: Function to generate initial samples.
            dt: Time step for the SDE.
        Returns:
            A JAX array of shape (n_samples, 2) representing generated samples.
        '''

        # Generate initial samples
        z = source_gen(n_samples, key)

        # Sample from the SDE
        for step in range(int(1/dt)):
            t = jnp.ones((n_samples, 1)) * (1 - step * dt)

            z = z + (-t**2 * self(jnp.hstack([z, t]), deterministic=True)) * dt + t * jnp.sqrt(dt) * random.normal(key, shape=z.shape)

        return z
