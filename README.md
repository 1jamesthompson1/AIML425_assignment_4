# Instructions

Consider three types of animals: dogs, cats, and gaussians. We will map from
gaussians to dogs and we will map cats to dogs. Unfortunately we have to limit the
problem to two-pixel images. Dogs have a uniform distribution in the square with corners (-1,1) and (-2,2). Cats have a uniform distribution in the square with corners (2,-2) and (3,-3). Gaussians have an identity covariance matrix scaled by a factor s.
▶ Your solutions must use the first-order Euler or Euler-Murayama approach that you program yourself.
▶ You will need one neural network for each approach, with inputs x ∈ R2 and t (straight or embedded). The output is in R2.
▶ Use the score function approach to obtain an SDE that maps gaussians into dogs. Use either f (x, t) = 0 (simple, fast, good) or the variance preserving case). Fine to end up with any scaling s.
▶ Train an ODE that maps Gaussians into dogs using the linear interpolant approach (slide 32). For this case use s = 1.
▶ Compare performance of the two approaches with an appropriate measure.
▶ Train an ODE that maps cats into dogs using your interpolant approach.
▶ Illustrate your work with suitable figures for distributions obtained with your SDE and ODEs and possibly their velocity fields.
▶ To keep track of what all this is, plot some example two-pixel images of dogs and cats and gaussians (amplify and offset by reasonable factor).

# Setting up environment

After git cloning simply run

```bash
uv sync
```

In project environment. Then the main.py can either be ran as a notebook or done from start to finish (in a few minutes on a GPU) with
```bash
uv run main.py
```

Some figures will be made and put in the `figures/` dir.

## Syncing of library

My workflow involves working on my laptop for most tasks but access vscode on a remote machine (on VUW campus) which is more powerful to do my coding and report writing. Therefore I need to sync my Zotero .bib export from my laptop to this University machine. Therefore to do this I run this command:
```bash
rsync -avz /home/james/code/VUWMAI/bibliographies/AIML425.bib vuw-lab:/home/thompsjame1/code/AIML425_assignment_4/references.bib
```