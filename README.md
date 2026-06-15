# PINN Bubble Profiler (NEEDS AN UPADATE)

Brief physics‑informed neural network (PINN) experiments for spherically symmetric field/bubble profiles. The repository contains small PINN examples (polynomial toy model and a 2-field singlet/Higgs-like potential) used to learn radial field profiles that satisfy the underlying Euler–Lagrange / field equations. Finding the bubble profile is equivalent to solving a nonlinear boundary value problem; PINNs replace a numerical ODE solver by encoding the residual of the field equation directly into the loss function. Related methods: Raissi et al., "Physics-informed neural networks" (2019).

**Quick Start**

Prerequisites:
- Python 3.8+
- PyTorch (CPU or CUDA build depending on your machine)

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch
```

Run an experiment (examples: `singlet`, `polynomial`):

```bash
python main.py --exp singlet
python main.py --exp polynomial
```

This will run a pretraining stage and a finetuning stage and save the model to the path configured in the experiment `Config` (example: `saved_models/polynomial.pth`). Create the directory if it does not exist:

```bash
mkdir -p saved_models
```

Project structure
- `main.py` — CLI entry point. Choose experiment with `--exp`.
- `train.py` — training loop with `pretrain` and `finetune` helpers.
- `src/model.py` — simple MLP model `Bouncer` used by experiments.
- `src/physics/polynomial.py` — toy polynomial potential, `Config` and `Loss`.
- `src/physics/singlet.py` — two-field potential (singlet + Higgs-like), `Config` and `Loss`.

How to tweak experiments
- Hyperparameters and physical parameters live in each experiment `Config` class (`src/physics/*.py`). Edit `pretrain_epochs`, `finetune_epochs`, `r_max`, etc., there.
- Model architecture is defined in `src/model.py` — `Bouncer` layers and activations.

Notes and references
- PINNs: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics.
- The physics implemented here targets radial (spherically symmetric) field equations; for background on bubble/radial profile physics see literature on the Rayleigh–Plesset / field-theory bounce solutions.
