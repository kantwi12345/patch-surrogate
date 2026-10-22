import torch
import torch.nn as nn


class PatchSurrogate(nn.Module):
    """
    MLP: geometry (5 features) -> [S11_db curve, gain_dbi curve] concatenated.
    """

    def __init__(self, input_dim=5, output_dim=602, hidden_dim=256, n_hidden_layers=5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)