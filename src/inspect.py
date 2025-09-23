# This code will be to generate some graphs as well as performance test the model.
import matplotlib.pyplot as plt
from jax import random, numpy as jnp
from flax import nnx
import optax
import pandas as pd
import seaborn as sns
import math

output_path = "./figures/"

def plot_training_history(history, name):
    '''
    Plot the train and val loss over epochs.
    Will make three plots one for regular loss, and one each for the loss_recon and loss_reg components.    Will make three plots one for regular loss, and one each for the loss_recon and loss_reg components.
    
    Args:
        history: A dictionary with keys 'train_loss' and 'val_loss', each containing a list of loss values per epoch.

    Returns:
        None (displays plots)
    '''

    plt.figure(figsize=(12, 8))

    # Plot training losses
    plt.plot(history['train_epochs'], history['train_loss'], label='Train Total Loss', color='blue')

    # Plot validation losses
    plt.plot(history['val_epochs'], history['val_loss'], label='Val Total Loss', color='cyan')

    plt.title('Training and Validation Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    if name is not None:
        plt.savefig(output_path + "/" + name + ".png")

    plt.show()

def inspect_images(image_generators, key, n_samples=500, name=None):
    '''
    Each iamge geneartor will produce a array of shape (n_samples, 2).
    These will be plotted as a scatter plot.
    '''

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    for i, (gen_fn, label) in enumerate(image_generators):
        key, subkey = random.split(key)
        samples = gen_fn(n_samples, subkey)
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label=label, s=10, color=colors[i % len(colors)])
        
    ax.set_title('Generated Samples from Different Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if name is not None:
        plt.savefig(output_path + "/" + name + ".png")

    plt.show()


def visualize_interpolation(source_gen, target_gen, n_samples=500, n_trajectories=20, key=None, name=None):
    '''
    Visualize how linear interpolation looks between two distributions.
    Shows individual trajectories and the interpolated points at different time steps.
    
    Args:
        source_gen: Function to generate source distribution (e.g., create_gaussian)
        target_gen: Function to generate target distribution (e.g., create_dogs)
        n_samples: Total number of samples to generate
        n_trajectories: Number of individual trajectories to show
        key: JAX random key
        name: Name for saving the plot
    '''
    if key is None:
        key = random.PRNGKey(42)
    
    key_source, key_target, key_t = random.split(key, 3)
    
    # Generate source and target points
    source_points = source_gen(n_samples, key_source)
    target_points = target_gen(n_samples, key_target)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Original distributions
    axes[0, 0].scatter(source_points[:, 0], source_points[:, 1], alpha=0.6, c='blue', label='Source', s=20)
    axes[0, 0].scatter(target_points[:, 0], target_points[:, 1], alpha=0.6, c='red', label='Target', s=20)
    axes[0, 0].set_title('Original Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Individual trajectories
    indices = jnp.arange(n_trajectories)
    t_vals = jnp.linspace(0, 1, 50)
    
    for i in indices:
        source_pt = source_points[i]
        target_pt = target_points[i]
        trajectory = jnp.array([(1-t) * source_pt + t * target_pt for t in t_vals])
        axes[0, 1].plot(trajectory[:, 0], trajectory[:, 1], alpha=0.7, linewidth=1)
    
    axes[0, 1].scatter(source_points[indices, 0], source_points[indices, 1], c='blue', s=50, label='Source', zorder=5)
    axes[0, 1].scatter(target_points[indices, 0], target_points[indices, 1], c='red', s=50, label='Target', zorder=5)
    axes[0, 1].set_title(f'Interpolation Trajectories (n={n_trajectories})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Interpolated points at different time steps
    t_steps = [0.2, 0.5, 0.8, 0.95]
    colors = ['purple', 'orange', 'green', 'blue']
    
    for t_val, color in zip(t_steps, colors):
        interpolated = (1 - t_val) * source_points + t_val * target_points
        axes[1, 0].scatter(interpolated[:, 0], interpolated[:, 1], alpha=0.6, c=color, 
                          label=f't={t_val}', s=15)
    
    axes[1, 0].set_title('Interpolated Points at Different Times')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Velocity field visualization
    # Sample random t values for training data
    t_random = random.uniform(key_t, shape=(n_samples, 1), minval=0.0, maxval=1.0)
    interpolated_random = (1 - t_random) * source_points + t_random * target_points
    velocities = target_points - source_points
    
    # Create a regular grid for cleaner visualization
    x_min, x_max = jnp.min(jnp.concatenate([source_points[:, 0], target_points[:, 0]])), jnp.max(jnp.concatenate([source_points[:, 0], target_points[:, 0]]))
    y_min, y_max = jnp.min(jnp.concatenate([source_points[:, 1], target_points[:, 1]])), jnp.max(jnp.concatenate([source_points[:, 1], target_points[:, 1]]))
    
    # Create grid points
    grid_resolution = 15  # Adjust this for density
    x_grid = jnp.linspace(x_min, x_max, grid_resolution)
    y_grid = jnp.linspace(y_min, y_max, grid_resolution)
    X, Y = jnp.meshgrid(x_grid, y_grid)
    grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    
    # For each grid point, find average velocity from nearby training points
    grid_velocities = []
    for grid_pt in grid_points:
        # Find distances to all training points
        dists = jnp.linalg.norm(interpolated_random - grid_pt, axis=1)
        # Use points within a certain radius for averaging
        radius = 0.5  # Adjust this for smoothness
        nearby_mask = dists < radius
        
        if jnp.sum(nearby_mask) > 0:
            avg_velocity = jnp.mean(velocities[nearby_mask], axis=0)
        else:
            # If no nearby points, use closest point
            closest_idx = jnp.argmin(dists)
            avg_velocity = velocities[closest_idx]
        
        grid_velocities.append(avg_velocity)
    
    grid_velocities = jnp.array(grid_velocities)
    
    # Normalize velocities for better visualization
    velocity_magnitudes = jnp.linalg.norm(grid_velocities, axis=1)
    max_mag = jnp.max(velocity_magnitudes)
    normalized_velocities = grid_velocities / max_mag * 0.3  # Scale factor for arrow size
    
    # Plot the velocity field
    axes[1, 1].quiver(grid_points[:, 0], grid_points[:, 1], 
                     normalized_velocities[:, 0], normalized_velocities[:, 1], 
                     velocity_magnitudes, cmap='viridis', alpha=0.8, 
                     scale_units='xy', scale=1, width=0.004)
    
    # Add some sample points for context
    sample_indices = jnp.arange(0, n_samples, n_samples // 100)
    axes[1, 1].scatter(interpolated_random[sample_indices, 0], interpolated_random[sample_indices, 1], 
                      c='lightgray', s=10, alpha=0.5, zorder=1)
    
    axes[1, 1].set_title('Velocity Field (Direction to Target)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if name is not None:
        plt.savefig(output_path + "/" + name + ".png", dpi=150, bbox_inches='tight')
    plt.show()

def visualize_noise_process(x_data, scores, name=None):
    '''
    Visualize how the noise process looks for score matching.
    Shows the noisy samples at different noise levels and the corresponding score vectors.
    
    Args:
        x_data: Array of shape (n_samples, 3) where columns are [x1, x2, t]
        scores: Array of shape (n_samples, 2) containing the true score vectors
        name: Name for saving the plot
    '''
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Noisy samples at different noise levels
    t_steps = [0.2, 0.5, 0.8, 0.95]
    colors = ['purple', 'orange', 'green', 'blue']
    
    for t_val, color in zip(t_steps, colors):
        mask = jnp.abs(x_data[:, 2].flatten() - t_val) < 0.05  # Select points near t_val
        if jnp.sum(mask) > 0:  # Only plot if we have points at this time
            axes[0].scatter(x_data[mask, 0], x_data[mask, 1], alpha=0.4, c=color, 
                            label=f't={t_val}', s=15)
    
    axes[0].set_title('Noisy Samples at Different Noise Levels')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Score field visualization
    # Create a regular grid for cleaner visualization
    x_min, x_max = jnp.min(x_data[:, 0]), jnp.max(x_data[:, 0])
    y_min, y_max = jnp.min(x_data[:, 1]), jnp.max(x_data[:, 1])

    # Create grid points
    grid_resolution = 20  # Adjust this for density
    x_grid = jnp.linspace(x_min, x_max, grid_resolution)
    y_grid = jnp.linspace(y_min, y_max, grid_resolution)
    X, Y = jnp.meshgrid(x_grid, y_grid)
    grid_points = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    grid_scores = []
    
    for grid_pt in grid_points:
        # Find distances to all noisy points
        dists = jnp.linalg.norm(x_data[:, :2] - grid_pt, axis=1)
        # Use points within a certain radius for averaging
        radius = 0.8  # Adjust this for smoothness
        nearby_mask = dists < radius
        
        if jnp.sum(nearby_mask) > 0:
            avg_score = jnp.mean(scores[nearby_mask], axis=0)
        else:
            # If no nearby points, use closest point
            closest_idx = jnp.argmin(dists)
            avg_score = scores[closest_idx]
        
        grid_scores.append(avg_score)
        
    grid_scores = jnp.array(grid_scores)
    
    # Normalize scores for better visualization
    score_magnitudes = jnp.linalg.norm(grid_scores, axis=1)
    max_mag = jnp.max(score_magnitudes)
    if max_mag > 0:  # Avoid division by zero
        normalized_scores = grid_scores / max_mag * 0.4  # Scale factor for arrow size
    else:
        normalized_scores = grid_scores
    
    # Plot the score field
    axes[1].quiver(grid_points[:, 0], grid_points[:, 1], 
                   normalized_scores[:, 0], normalized_scores[:, 1], 
                   score_magnitudes, cmap='viridis', alpha=0.8, 
                   scale_units='xy', scale=1, width=0.004)
    
    # Add some sample points for context
    sample_indices = jnp.arange(0, x_data.shape[0], max(1, x_data.shape[0] // 100))
    axes[1].scatter(x_data[sample_indices, 0], x_data[sample_indices, 1], 
                    c='lightgray', s=10, alpha=0.5, zorder=1)
    
    axes[1].set_title('Score Field (Direction to De-noise)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if name is not None:
        plt.savefig(output_path + "/" + name + ".png", dpi=150, bbox_inches='tight')
    plt.show()

def visualize_model_generation(model, source_gen, target_gen, n_samples, key, name=None):
    '''
    Visualize samples generated by the model starting from source_gen distribution.
    
    Args:
        model: Trained model with a .generate method
        source_gen: Function to generate initial samples (e.g., create_gaussian)
        n_samples: Number of samples to generate
        name: Name for saving the plot
    '''
    noise_samples = source_gen(n_samples, key)
    generated_samples = model.generate(noise_samples, key=key, dt=0.001)

    target_samples = target_gen(n_samples, key)

    plt.figure(figsize=(8, 8))
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, c='green', s=10)
    plt.scatter(target_samples[:, 0], target_samples[:, 1], alpha=0.1, c='yellow', s=10)
    plt.scatter(noise_samples[:, 0], noise_samples[:, 1], alpha=0.1, c='blue', s=10)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(['Generated Samples', 'Target Samples', 'Noise Samples'])
    plt.title('Samples Generated by the Model')
    plt.grid(True, alpha=0.3)

    if name is not None:
        plt.savefig(output_path + "/" + name + ".png")

    plt.show()