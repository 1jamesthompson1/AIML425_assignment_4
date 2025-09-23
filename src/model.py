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

    def generate(self, z, n_samples, key, dt=0.01):
        '''
        Generate samples from the learned SDE model starting from source_gen samples.

        Args:
            source_gen: Function to generate initial samples.
            dt: Time step for the SDE.
        Returns:
            A JAX array of shape (n_samples, 2) representing generated samples.
        '''

        # Sample from the SDE
        for step in range(int(1/dt)):
            key , noise_key = random.split(key)
            t = jnp.ones((n_samples, 1)) * (1 - step * dt)

            sigma_min, sigma_max = 0.01, 1.0
            sigma_t = sigma_min * (sigma_max / sigma_min) ** t
            model = self(jnp.hstack([z, t]), deterministic=True)
            score = (1 / sigma_t**2) * model
            noise = random.normal(noise_key, shape=z.shape)

            z = z + sigma_t**2 * score * dt + sigma_t * jnp.sqrt(dt) * noise

        return z
