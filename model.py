import torch.nn as nn

class Bouncer(nn.Module):

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
        return self.network(x) # returns (batch_size, output_dim)