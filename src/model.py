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

    def euler_maruyama_step(self, x, dt, drift, diffusion, noise):
        '''
        Perform a single Euler-Maruyama step for the SDE  
        X_{t+dt} = X_t + f(X_t, t) * dt + g(X_t, t) * sqrt(dt) * noise
        '''
        return x + drift * dt + diffusion * jnp.sqrt(jnp.abs(dt)) * noise

    def generate(self, z, key, dt=0.01):
        '''
        Generate samples from the learned SDE model starting from source_gen samples.

        Args:
            z: Initial samples from the source distribution.
            dt: Time step for the SDE numerical integration.
        Returns:
            A JAX array of shape (n_samples, 2) representing generated samples.
        '''
        n_samples = z.shape[0]
        num_steps = int(1 / dt)
        t_grid = jnp.linspace(1.0, 0.0, num_steps + 1)

        for step in range(num_steps):
            key, noise_key = random.split(key)

            # t_curr = jnp.clip(t_grid[step], 1e-3, 0.999)
            t_curr = t_grid[step]
            t = jnp.full((n_samples, 1), t_curr)

            beta_t = 2.0 / jnp.clip(1.0 - t, a_min=1e-3)
            dt_step = t_grid[step + 1] - t_grid[step]  # negative value for reverse-time integration

            score = self(jnp.hstack([z, t]), deterministic=True)
            drift = -0.5 * beta_t * z - beta_t * score
            diffusion = jnp.sqrt(beta_t)

            noise = random.normal(noise_key, shape=z.shape)
            z = self.euler_maruyama_step(z, dt=dt_step, drift=drift, diffusion=diffusion, noise=noise)
            

        return z

class ODE(MLP):

    def euler_step(self, x, t, dt):
        '''
        Perform a single Euler step for the ODE  
        X_{t+dt} = X_t + f(X_t, t) * dt
        '''
        return x + self(jnp.hstack([x, t]), deterministic=True) * dt
    
    def generate(self, z, key, dt=0.01):
        '''
        Generate samples from the learned ODE model starting from source_gen samples.

        Args:
            z: Initial samples from the source distribution.
            dt: Time step for the ODE numerical integration.
        Returns:
            A JAX array of shape (n_samples, 2) representing generated samples.
        '''
        # Integrate forward in time: t from 0 -> 1. The model was trained
        # with linear interpolation x_t = (1 - t) x0 + t x1, whose velocity
        # field is dx/dt = x1 - x0. Starting from source samples (t=0), we
        # should step forward in t and add the predicted velocity.
        num_steps = int(1 / dt)
        for step in range(num_steps):
            # Keep t within the training support [0, 1). Avoid exactly t=1.
            t_scalar = min(step * dt, 0.9999)
            t = jnp.ones((z.shape[0], 1)) * t_scalar
            z = self.euler_step(z, t, dt)

        return z