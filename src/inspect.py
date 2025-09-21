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

