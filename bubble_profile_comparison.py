import torch
import numpy as np
import matplotlib.pyplot as plt
from cosmoTransitions import pathDeformation
from functools import partial

class BubbleProfileComparison:

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def _get_benchmark_profile(self):
        potential = self.config.potential(self.config.Tc, self.config.lambda_m, self.config.lambda_s)
        v, w = potential.v, potential.w
        V, dV = partial(potential.V, T=self.config.T), partial(potential.dV, T=self.config.T)

        bubble_profile = pathDeformation.fullTunneling([[v(self.config.T),0],[0,w(self.config.T)]], V, dV)
        return bubble_profile

    def _get_pinn_profile(self):
        test_points = torch.linspace(0.01, self.config.r_max, 1000, device=self.config.device).view(-1, 1)
        
        self.model.eval()
        with torch.no_grad():
            bubble_profile = self.model(test_points).cpu().numpy()
        return bubble_profile

    def plot_and_save(self):
        benchmark_bubble_profile = self._get_benchmark_profile()
        pinn_bubble_profile = self._get_pinn_profile()

        grid = np.linspace(0.01, self.config.r_max, 1000).reshape(-1,1)

        plt.figure(figsize=(8,5))
        plt.plot(grid, pinn_bubble_profile[:,0],label='$h$ PINN Solution', c="blue", linewidth=2)
        plt.plot(grid, pinn_bubble_profile[:,1], label='$s$ PINN Solution', c="green", linewidth=2)
        plt.plot(benchmark_bubble_profile.profile1D.R, benchmark_bubble_profile.Phi[:,0], "-.",label="$h$ CosmoTransitions Solution", c="red", linewidth=2)
        plt.plot(benchmark_bubble_profile.profile1D.R, benchmark_bubble_profile.Phi[:,1], "-.",label="$s$ CosmoTransitions Solution", c="purple", linewidth=2)
        plt.xlabel(r"$r$ $\times$ $v_{EW}^2$", fontsize=16)
        plt.ylabel(r"$\phi(r) / v_{EW}$", fontsize=16)
        plt.xlim(xmax=benchmark_bubble_profile.profile1D.R[-1])
        plt.legend(fontsize=14, frameon=False)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.config.saveComparisonPath}",dpi=300)
        plt.show()


