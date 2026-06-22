import torch
import numpy as np
import matplotlib.pyplot as plt

from cosmoTransitions import pathDeformation
from cosmoTransitions.tunneling1D import SingleFieldInstanton
from functools import partial
from abc import ABC, abstractmethod


class BubbleProfileComparison(ABC):
    """
    Abstract base class for comparing PINN-generated bubble profiles
    against benchmark solvers (analytic / numerical / CosmoTransitions).

    Each subclass defines:
    - how to compute benchmark solution
    - how to evaluate PINN solution
    - how to visualize comparison
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    @abstractmethod
    def _get_benchmark_profile(self):
        """Compute reference solution using external solver."""
        pass

    @abstractmethod
    def _get_pinn_profile(self):
        """Evaluate trained PINN on radial grid."""
        pass

    @abstractmethod
    def plot_comparison(self):
        """Generate comparison plot between PINN and benchmark."""
        pass


class PolynomialProfileComparison(BubbleProfileComparison):
    """
    Comparison for single-field polynomial potential.
    """

    def __init__(self, config, model):
        super().__init__(config, model)

    def _get_benchmark_profile(self):
        # Construct scalar potential and its derivative at vacuum configuration c
        potential = self.config.potential(self.config.c)
        V, dV = potential.V, potential.dV

        # Solve bounce equation using shooting method (CosmoTransitions)
        bubble_profile = SingleFieldInstanton(1, 0, V, dV).findProfile()
        return bubble_profile

    def _get_pinn_profile(self):
        # Radial grid for evaluating learned field configuration
        # start at r=0.01 to avoid singularity in spherical Laplacian
        test_points = torch.linspace(
            0.01,
            self.config.r_max,
            1000,
            device=self.config.device
        ).view(-1, 1)

        self.model.eval()

        # Disable gradient tracking during inference
        with torch.no_grad():
            bubble_profile = self.model(test_points).cpu().numpy()

        return bubble_profile

    def plot_comparison(self):
        # Compute both solutions
        benchmark_bubble_profile = self._get_benchmark_profile()
        pinn_bubble_profile = self._get_pinn_profile()

        # radial grid for PINN evaluation
        grid = np.linspace(0.01, self.config.r_max, 1000).reshape(-1, 1)

        plt.figure(figsize=(8, 5))

        # PINN solution (learned approximation of bounce solution)
        plt.plot(grid, pinn_bubble_profile,
                 label='PINN Solution', c="blue", linewidth=2)

        # Benchmark solution from CosmoTransitions (numerical ground truth)
        plt.plot(benchmark_bubble_profile.R,
                 benchmark_bubble_profile.Phi,
                 "-.",
                 label="CosmoTransitions Solution",
                 c="red",
                 linewidth=2)

        plt.xlabel(r"$r$", fontsize=16)
        plt.ylabel(r"$\phi(r)$", fontsize=16)

        # Match plot domain to benchmark solution domain
        plt.xlim(right=benchmark_bubble_profile.R[-1])

        plt.legend(fontsize=14, frameon=False)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save figure for later analysis / paper plots
        plt.savefig(f"{self.config.saveComparisonPath}", dpi=300)
        plt.show()


class SingletProfileComparison(BubbleProfileComparison):
    """
    Comparison for two-field (h, s) singlet extension model.
    """

    def __init__(self, config, model):
        super().__init__(config, model)

    def _get_benchmark_profile(self):
        # Construct finite-temperature scalar potential
        potential = self.config.potential(
            self.config.Tc,
            self.config.lambda_m,
            self.config.lambda_s
        )

        v, w = potential.v, potential.w

        # Temperature-dependent potential and derivatives
        V = partial(potential.V, T=self.config.T)
        dV = partial(potential.dV, T=self.config.T)

        # Solve multi-field tunneling trajectory using path deformation
        bubble_profile = pathDeformation.fullTunneling(
            [[v(self.config.T), 0],
             [0, w(self.config.T)]],
            V,
            dV
        )

        return bubble_profile

    def _get_pinn_profile(self):
        # Radial grid for PINN evaluation
        test_points = torch.linspace(
            0.01,
            self.config.r_max,
            1000,
            device=self.config.device
        ).view(-1, 1)

        self.model.eval()

        with torch.no_grad():
            bubble_profile = self.model(test_points).cpu().numpy()

        return bubble_profile

    def plot_comparison(self):
        # Compute both solutions
        benchmark_bubble_profile = self._get_benchmark_profile()
        pinn_bubble_profile = self._get_pinn_profile()

        # Radial grid for PINN evaluation
        grid = np.linspace(0.01, self.config.r_max, 1000).reshape(-1, 1)

        plt.figure(figsize=(8, 5))

        # PINN solutions for both fields
        plt.plot(grid, pinn_bubble_profile[:, 0],
                 label='$h$ PINN Solution', c="blue", linewidth=2)

        plt.plot(grid, pinn_bubble_profile[:, 1],
                 label='$s$ PINN Solution', c="green", linewidth=2)

        # Benchmark solutions (CosmoTransitions)
        plt.plot(benchmark_bubble_profile.profile1D.R,
                 benchmark_bubble_profile.Phi[:, 0],
                 "-.",
                 label="$h$ CosmoTransitions Solution",
                 c="red",
                 linewidth=2)

        plt.plot(benchmark_bubble_profile.profile1D.R,
                 benchmark_bubble_profile.Phi[:, 1],
                 "-.",
                 label="$s$ CosmoTransitions Solution",
                 c="purple",
                 linewidth=2)

        plt.xlabel(r"$r$ $\times$ $v_{EW}^2$", fontsize=16)
        plt.ylabel(r"$\phi(r) / v_{EW}$", fontsize=16)

        # Align x-range with benchmark solution
        plt.xlim(right=benchmark_bubble_profile.profile1D.R[-1])

        plt.legend(fontsize=14, frameon=False)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save for analysis / publication
        plt.savefig(f"{self.config.saveComparisonPath}", dpi=300)
        plt.show()



