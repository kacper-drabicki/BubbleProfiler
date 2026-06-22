import os
import argparse
import torch
from train import pretrain, finetune
from compare_profile_with_benchmark import PolynomialProfileComparison, SingletProfileComparison
from model import Bouncer
from physics import polynomial, singlet

EXPERIMENTS = {
    "singlet": (singlet.Config, SingletProfileComparison),
    "polynomial": (polynomial.Config, PolynomialProfileComparison)
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=EXPERIMENTS.keys())
    args = parser.parse_args()

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    config = EXPERIMENTS[args.exp][0]()
    
    model = Bouncer(output_dim=config.output_dim).to(config.device)

    print("=== PRETRAIN ===")
    pretrain(model, config)

    print("=== FINETUNE ===")
    finetune(model, config)

    torch.save(model.state_dict(), config.saveModelPath)
        
    comparison_with_benchmark = EXPERIMENTS[args.exp][1](config, model)
    comparison_with_benchmark.plot_and_save()

if __name__ == "__main__":
    main()
