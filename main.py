# %% [markdown]
# # AIML 425 - Assignment 3
# ## Problem 2: Variational Auto Encoders and Auto Encoders

# This notebook is provided for ease of use for marking. However it sohuld be noted that the development was conducted with the notebook as script percent format. The assignment repository can be found at [my gitea instance](https://gitea.james-server.duckdns.org/james/AIML425_assignment_4)  
# 
# Most of the interesting good stuff is in the `src/` directory this file just runs the experiments and call the implementations.


# ## Startup
# %% 
import os
# Only run if in Google Colab environment
if 'google.colab' in str(get_ipython()):
    # Clone your repository (only needed once per session)
    if not os.path.isdir('src') and len(os.listdir('src')) > 0:
        raise RuntimeError("The src directory already exists and is not empty. Please remove or rename it before running this script.")

    
    !git clone https://github.com/1jamesthompson1/AIML425_assignment_4.git
    
    # Copy the src directory to the current working directory
    !cp -r AIML425_assignment_4/src ./
    
    # Clean up the cloned repository
    !rm -rf AIML425_assignment_4

    print("Repository downloaded and src directory extracted to project root")
else:
    print("Not running in Google Colab - skipping repository clone")

# %% 

from jax import random
from jax import numpy as jnp
from importlib import reload
from functools import partial

from src import model, train, data, inspect

# This is the main key used for all random operations.
key = random.key(42)

reload(model)
reload(train)
reload(data)
reload(inspect)

################################################################################
################################################################################

# ------------ Data generation -----------------
# %% [markdown]
################################################################################
# # Generate data

# %%
reload(data)
reload(inspect)


# %%

################################################################################
################################################################################

# ------------ SDE Gaussian to dogs -----------------
# %% [markdown]
################################################################################
# # Train a SDE model to convert Gaussian noise to dog images

# %%
reload(train)
reload(data)
reload(model)
reload(inspect)

sde_trained_model, sde_history = train.do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    model_class=model.SDE,
    loss_fn=partial(
        train.sde_loss_fn,
        kl_beta=0.65
    ),
    learning_rate=0.001,
    minibatch_size=512,
    latent_dim=32,
    encoder_arch=[1000, 1000, 1000, 500 ],
    decoder_arch=[500, 1000, 1000, 1000],
    num_epochs=1000,
    eval_every=20,
    dropout=0.1, 
)

inspect.plot_training_history(sde_history, 'sde-training-history')
# %% [markdown]
# ## Understand the performance of the model

# %%
# Visualizing the mapping from Gaussian to dog images
reload(inspect)


################################################################################
################################################################################

# ------------ ODE from Gaussian to dogs -----------------
# %% [markdown]
# # Train an ODE model
################################################################################
################################################################################

# %%
reload(train)
reload(data)
reload(model)
reload(inspect)

ode_gaussdogs_trained_model, ode_gaussdogs_history = train.do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    model_class=model.ODE,
    model_kwargs={
        "latent_noise_scale": 0.1
    },
    loss_fn=partial(
        train.ode_loss_fn,
    ),
    learning_rate=0.001,
    minibatch_size=64,
    latent_dim=10,
    encoder_arch=[1000, 1000, 500],
    decoder_arch=[500, 1000, 1000],
    num_epochs=500,
    eval_every=10,
    dropout=0.1,
)

inspect.plot_training_history(ode_gaussdogs_history, name='ode-gaussdogs-training-history')

# %% [markdown]
# ## Understand the performance of the model

# %%
# Visualizing the mapping from Gaussian to dog images
reload(inspect)

################################################################################
################################################################################

# ------------ Compare ODE to SDE -----------------

# %% [markdown]
# # Compare ODE to SDE



################################################################################
################################################################################

# ------------ ODE from cats to dogs -----------------
# %% [markdown]
# # Train an ODE model
################################################################################
################################################################################

# %%
reload(train)
reload(data)
reload(model)
reload(inspect)

ode_catsdogs_trained_model, ode_catsdogs_history = train.do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    model_class=model.ODE,
    model_kwargs={
        "latent_noise_scale": 0.1
    },
    loss_fn=partial(
        train.ode_loss_fn,
    ),
    learning_rate=0.001,
    minibatch_size=64,
    latent_dim=10,
    encoder_arch=[1000, 1000, 500],
    decoder_arch=[500, 1000, 1000],
    num_epochs=500,
    eval_every=10,
    dropout=0.1,
)

inspect.plot_training_history(ode_catsdogs_history, name='ode-catsdogs-training-history')

# %% [markdown]
# ## Understand the performance of the model

# %%
# Visualizing the mapping from Gaussian to dog images
reload(inspect)

