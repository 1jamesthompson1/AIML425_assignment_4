import jax.numpy as jnp
from jax import random
import jax

def create_dogs(n_samples, key):
    '''
    Generate synthetic dog images as random noise for demonstration purposes.
    They are represented as points in a uniform 2d space with corners at (-1,1) and (-2,2)
    
    Args:
        n_samples: Number of dog images to generate.
        key: JAX random key for generating random numbers.

    Returns:
        A JAX array of shape (n_samples, 2) representing dog images.
    '''
    
    return random.uniform(key, shape=(n_samples, 2), minval=jnp.array([-2, 1]), maxval=jnp.array([-1, 2]))

def create_cats(n_samples, key):
    '''
    Generate synthetic cat images as random noise for demonstration purposes.
    It is represented as points in a uniform 2d space with corners at (2,-2) and (3,-3)

    Args:
        n_samples: Number of cats images to generate.
        key: JAX random key for generating random numbers.

    Returns:
        A JAX array of shape (n_samples, 2) representing cat images.
    '''
    
    return random.uniform(key, shape=(n_samples, 2), minval=jnp.array([2, -3]), maxval=jnp.array([3, -2]))

def create_gaussian(n_samples, key, scale=1):
    '''
    Generate synthetic gaussian images as random noise for demonstration purposes.

    Args:
        n_samples: Number of gaussian images to generate.
        key: JAX random key for generating random numbers.

    Returns:
        A JAX array of shape (n_samples, 2) representing gaussian images.
    '''

    return random.multivariate_normal(key, mean=jnp.zeros(2), cov=jnp.eye(2) * scale, shape=(n_samples,))

def create_database(x_gen, y_gen, n_samples, key, technique):
    '''
    Create a dataset by generating samples using the provided generator functions.

    Args:
        x_gen: Function to generate input data.
        y_gen: Function to generate target data.
        n_samples: Number of samples to generate.
        key: JAX random key for generating random numbers.

    Returns:
        A tuple of JAX arrays (x_data, y_data), x_data of shape (n_samples, 3), y_data of shape (n_samples, 2).
    '''

    if technique == "linear_interpolation":
        key_x, key_y, key_t = random.split(key, 3)
        x_data = x_gen(n_samples, key_x)
        y_data = y_gen(n_samples, key_y)
        t = random.uniform(key_t, shape=(n_samples, 1), minval=0.0, maxval=1.0)
        x_data = (1 - t) * x_data + t * y_data

        x_data = jnp.hstack([x_data, t])
        
        return x_data, x_data - y_data

    elif technique == "score_matching":
        key_x, key_y, key_t, key_eps = random.split(key, 4)
        y_data = y_gen(n_samples, key_y)
        t = random.uniform(key_t, shape=(n_samples, 1), minval=0.01, maxval=0.99)  # Avoid t=0,1

        # Improved noise schedule - Variance Preserving (VP)
        beta_min = 0.1
        beta_max = 10.0
        beta_t = beta_min + t * (beta_max - beta_min)
        sigma_t = jnp.sqrt(1 - jnp.exp(-beta_t))
        
        # Generate proper Gaussian noise
        epsilon = x_gen(n_samples, key_eps)
        x_data = y_data + sigma_t * epsilon
        x_data = jnp.hstack([x_data, t])
        return x_data, -epsilon / (sigma_t**2)
    
    else:
        raise ValueError(f"Unknown technique: {technique}")

def create_batches(x_data, y_data, minibatch_size, key):
    '''
    Create mini-batches from the input data. It yields a generator that produces batches of the specified size.
    
    Args:
        x_data: JAX array of shape (num_samples, features).
        minibatch_size: Size of each mini-batch.
        y_data: JAX array of shape (num_samples, features).
        key: JAX random key for shuffling the data.
    
    Yields:
        A dictionary with keys 'input' and 'target' containing mini-batches of the data.
    '''
    x_data = jax.device_put(x_data)
    y_data = jax.device_put(y_data)

    n_samples = x_data.shape[0]
    if key is not None:
        indices = jnp.arange(n_samples)
        shuffled_indices = random.permutation(key, indices)
        x_data = x_data[shuffled_indices]
        y_data = y_data[shuffled_indices]

    if x_data.ndim == 1 or y_data.ndim == 1:
        raise ValueError(f"Input data must be at least 2D arrays.\nRecieved shapes: x_data {x_data.shape}, y_data {y_data.shape}")

    n_batches = n_samples // minibatch_size
    for i in range(n_batches):
        start_idx = i * minibatch_size
        end_idx = start_idx + minibatch_size
        batch = {
            "input": x_data[start_idx:end_idx],
            "target": y_data[start_idx:end_idx],
        }
        yield batch
    if n_samples % minibatch_size != 0:
        start_idx = n_batches * minibatch_size
        batch = {
            "input": x_data[start_idx:],
            "target": y_data[start_idx:],
        }
        yield batch