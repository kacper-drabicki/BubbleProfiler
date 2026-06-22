import torch.nn as nn

class Bouncer(nn.Module):
    """
    Simple fully-connected neural network used as a PINN ansatz
    for approximating scalar field configurations.
    """
    
    def __init__(
            self,
            input_dim: int = 1,
            hidden_dim: int = 10,
            output_dim: int = 1,
            activation: nn.Module = nn.Tanh()
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim)
            )
        
    def forward(self, x):
        """
        Forward pass of the PINN.

        Args:
            x (Tensor): radial coordinate r, shape (batch_size, 1)

        Returns:
            Tensor: field prediction, shape (batch_size, output_dim)
        """
        
        return self.network(x) 