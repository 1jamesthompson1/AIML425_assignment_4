import jax.numpy as jnp
from jax import random
import jax
import time
import math
from itertools import product


def create_dogs(n_samples, key):
    '''
    Generate synthetic dog images as random noise for demonstration purposes.
    
    Args:
        n_samples: Number of dog images to generate.
        key: JAX random key for generating random numbers.

    Returns:
        A JAX array of shape (n_samples, 2) representing dog images.
    '''
    
    raise NotImplementedError("This function is a placeholder. Implement dog image generation logic here.")

def create_cats(n_samples, key):
    '''
    Generate synthetic cat images as random noise for demonstration purposes.

    Args:
        n_samples: Number of cats images to generate.
        key: JAX random key for generating random numbers.

    Returns:
        A JAX array of shape (n_samples, 2) representing cat images.
    '''
    
    raise NotImplementedError("This function is a placeholder. Implement dog image generation logic here.")

def create_gaussian(n_samples, key):
    '''
    Generate synthetic gaussian images as random noise for demonstration purposes.

    Args:
        n_samples: Number of gaussian images to generate.
        key: JAX random key for generating random numbers.

    Returns:
        A JAX array of shape (n_samples, 2) representing gaussian images.
    '''

    raise NotImplementedError("This function is a placeholder. Implement dog image generation logic here.")

def create_database(x_gen, y_gen, n_samples, key):
    '''
    Create a dataset by generating samples using the provided generator functions.

    Args:
        x_gen: Function to generate input data.
        y_gen: Function to generate target data.
        n_samples: Number of samples to generate.
        key: JAX random key for generating random numbers.

    Returns:
        A tuple of JAX arrays (x_data, y_data) each of shape (n_samples, features).
    '''
    key_x, key_y = random.split(key)
    x_data = x_gen(n_samples, key_x)
    y_data = y_gen(n_samples, key_y)
    return x_data, y_data

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