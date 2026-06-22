import os
import argparse
import torch
from train import pretrain, finetune
from compare_benchmark import PolynomialProfileComparison, SingletProfileComparison
from model import Bouncer
from physics import polynomial, singlet

# Registry of available experiments.
# Each experiment defines:
#   (1) a Config class (hyperparameters, physics setup, paths)
#   (2) a comparison/evaluation class (PINN vs benchmark plotting)
EXPERIMENTS = {
    "singlet": (singlet.Config, SingletProfileComparison),
    "polynomial": (polynomial.Config, PolynomialProfileComparison)
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=EXPERIMENTS.keys())
    args = parser.parse_args()

    # Ensure output directories exist for models and plots
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Instantiate experiment-specific configuration (physics + training setup)
    config = EXPERIMENTS[args.exp][0]()

    # Initialize PINN model and move to correct device (CPU/GPU)
    model = Bouncer(output_dim=config.output_dim).to(config.device)

    print("=== PRETRAIN ===")
    # Pretraining phase: enforces general structure / stabilizes solution by simplifying boundary conditions (BCs)
    pretrain(model, config)

    print("=== FINETUNE ===")
    # Finetuning phase: enforces physics constraints more strongly; uses correct BCs
    finetune(model, config)

    # Save trained model parameters for later evaluation or reuse
    torch.save(model.state_dict(), config.saveModelPath)

    # Run experiment-specific evaluation:
    # compares PINN solution against the benchmark ("the shooting method")
    benchmark_comparison = EXPERIMENTS[args.exp][1](config, model)
    benchmark_comparison.plot_comparison()


if __name__ == "__main__":
    main()
