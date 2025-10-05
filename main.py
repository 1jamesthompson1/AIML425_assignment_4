# %% [markdown]
# # AIML 425 - Assignment 3
# ## Problem 2: Variational Auto Encoders and Auto Encoders

# This notebook is provided for ease of use for marking. However it should be noted that the development was conducted with the notebook as script percent format. The assignment repository can be found at [my gitea instance](https://gitea.james-server.duckdns.org/james/AIML425_assignment_4)  
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

# My own code split into modules in the src directory.
# model has the model definitions
# train has the training loop and loss functions
# data has the data generation functions
# inspect has the visualization functions
from src import model, train, data, inspect

# This is the main key used for all random operations.
key = random.key(42)

reload(model)
reload(train)
reload(data)
reload(inspect)

# %%
################################################################################

# ------------ Data generation -----------------

################################################################################

reload(inspect)
reload(data)

inspect.inspect_images(
    [
        (data.create_dogs, "Dogs"),
        (data.create_cats, "Cats"),
        (data.create_gaussian, "Gaussian"),
    ],
    key,
    name ="data-samples"
)

inspect.inspect_ind_images(
    [
        (data.create_dogs, "Dogs"),
        (data.create_cats, "Cats"),
        (data.create_gaussian, "Gaussian"),
    ],
    key,
    n=5,
    name="individual-data-samples"
)

# %%
reload(data)
reload(inspect)

inspect.visualize_interpolation(
    source_gen=data.create_gaussian,
    target_gen=data.create_dogs,
    n_samples=500,
    n_trajectories=30,
    key=random.split(key)[0],
    name="interpolation-gaussians-to-dogs"
)

inspect.visualize_interpolation(
    source_gen=data.create_cats,
    target_gen=data.create_dogs,
    n_samples=500,
    n_trajectories=30,
    key=random.split(key)[0],
    name="interpolation-cats-to-dogs"
)

# %%
reload(data)
inspect.visualize_noise_process(
    *data.create_database(
        x_gen=data.create_gaussian,
        y_gen=data.create_dogs,
        n_samples=5000,
        key=random.split(key)[0],
        technique="score_matching",
    ),
    name="noise-process-dogs"
)



# %%

# %% [markdown]
################################################################################

# ------------ SDE Gaussian to dogs -----------------

################################################################################
# # Train SDE model to go from Gaussian to dogs
# This is trained using the score matching loss function.

# %%
reload(train)
reload(data)
reload(model)
reload(inspect)
train_batches = partial(
    data.create_batches,
    *data.create_database(
        x_gen=data.create_gaussian,
        y_gen=data.create_dogs,
        n_samples=100000,
        key=random.split(key)[0],
        technique="score_matching",
    ),
)

valid_batches = partial(
    data.create_batches,
    *data.create_database(
        x_gen=data.create_gaussian,
        y_gen=data.create_dogs,
        n_samples=20000,
        key=random.split(key)[1],
        technique="score_matching",
    ),
)

sde_trained_model, sde_history = train.do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    model_class=model.SDE,
    loss_fn=train.mse_loss,
    output_dim=2,
    learning_rate=0.0001,
    minibatch_size=512,
    hidden_dims=[512] * 4,
    num_epochs=500,
)

inspect.plot_training_history(sde_history, 'sde-training-history')

# %%
reload(inspect)
inspect.visualize_model_generation(
    sde_trained_model,
    source_gen=data.create_gaussian,
    target_gen=data.create_dogs,
    n_samples=500,
    key=random.split(key)[1],
    name="sde-generation",
    dt=0.0001
)

inspect.visualize_score_field_sde(
    sde_trained_model,
    name="sde-score-field",
    key=random.split(key)[1],
)


# %% [markdown]
################################################################################

# ------------ ODE from Gaussian to dogs -----------------

################################################################################

# # Train an ODE model
# %%
reload(train)
reload(data)
reload(model)
reload(inspect)
train_batches = partial(
    data.create_batches,
    *data.create_database(
        x_gen=data.create_gaussian,
        y_gen=data.create_dogs,
        n_samples=10000,
        key=random.split(key)[0],
        technique="linear_interpolation"
    ),
)

valid_batches = partial(
    data.create_batches,
    *data.create_database(
        x_gen=data.create_gaussian,
        y_gen=data.create_dogs,
        n_samples=2000,
        key=random.split(key)[1],
        technique="linear_interpolation"
    ),
)

ode_gaussdog_trained_model, ode_gaussdog_history = train.do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    model_class=model.ODE,
    loss_fn=train.mse_loss,
    output_dim=2,
    learning_rate=0.0001,
    num_epochs=400,
)

inspect.plot_training_history(ode_gaussdog_history, 'ode-gaussdog-training-history')

# %%
reload(inspect)
inspect.visualize_model_generation(
    ode_gaussdog_trained_model,
    source_gen=data.create_gaussian,
    target_gen=data.create_dogs,
    n_samples=500,
    key=random.split(key)[1],
    name="ode-gaussdog-generation"
)

inspect.visualize_velocity_field_ode(
    ode_gaussdog_trained_model,
    name="ode-gaussdog-velocity-field",
    key=random.split(key)[1],
)

# %% [markdown]
################################################################################

# ------------ ODE from cats to dogs -----------------
################################################################################

# # Train an ODE model
# %%
reload(train)
reload(data)
reload(model)
reload(inspect)
train_batches = partial(
    data.create_batches,
    *data.create_database(
        x_gen=data.create_cats,
        y_gen=data.create_dogs,
        n_samples=10000,
        key=random.split(key)[0],
        technique="linear_interpolation"
    ),
)

valid_batches = partial(
    data.create_batches,
    *data.create_database(
        x_gen=data.create_cats,
        y_gen=data.create_dogs,
        n_samples=2000,
        key=random.split(key)[1],
        technique="linear_interpolation"
    ),
)

ode_catdog_trained_model, ode_catdog_history = train.do_complete_experiment(
    key,
    train_batches,
    valid_batches,
    model_class=model.ODE,
    loss_fn=train.mse_loss,
    output_dim=2,
    learning_rate=0.0001,
    num_epochs=200,
)

inspect.plot_training_history(ode_catdog_history, 'ode-catdog-training-history')

# %%
reload(inspect)
inspect.visualize_model_generation(
    ode_catdog_trained_model,
    source_gen=data.create_cats,
    target_gen=data.create_dogs,
    n_samples=500,
    key=random.split(key)[1],
    name="ode-catdog-generation"
)

inspect.visualize_velocity_field_ode(
    ode_catdog_trained_model,
    name="ode-catdog-velocity-field",
    key=random.split(key)[1],
)

# %% [markdown]
################################################################################

# ------------ Compare ODE to SDE for Gaussian to dog -----------------

################################################################################

# # Compare ODE to SDE

reload(inspect)

mmd_sde = inspect.generative_performance(
    model=sde_trained_model,
    source_dist=data.create_gaussian,
    target_dist=data.create_dogs,
    num_samples=10000,
    rng_key=random.split(key)[0],
)

mmd_ode = inspect.generative_performance(
    model=ode_gaussdog_trained_model,
    source_dist=data.create_gaussian,
    target_dist=data.create_dogs,
    num_samples=10000,
    rng_key=random.split(key)[0],
)

mmd_ode_catdog = inspect.generative_performance(
    model=ode_catdog_trained_model,
    source_dist=data.create_cats,
    target_dist=data.create_dogs,
    num_samples=10000,
    rng_key=random.split(key)[0],
)

print(f"MMD between SDE generated samples and dogs: {mmd_sde:.6f}")
print(f"MMD between ODE generated samples and dogs: {mmd_ode:.6f}")
print(f"MMD between ODE (cats to dogs) generated samples and dogs: {mmd_ode_catdog:.6f}")
