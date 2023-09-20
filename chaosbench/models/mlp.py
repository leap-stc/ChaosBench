import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple single-layer MLP for benchmarking purposes
    """
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        
        layers = []

        # Loop through the hidden sizes and add linear layers to the list
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.15))
            input_size = hidden_size

        # Add the final linear layer for the output
        layers.append(nn.Linear(input_size, output_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        B, P, L, H, W = x.shape
        
        x = x.permute((0, 3, 4, 1, 2)) # to shape (B, H, W, P, L)
        
        x = self.model(x.view(B, H, W, -1)) # to shape (B, H, W, P*L)
        
        x = x.permute((0, 3, 1, 2)) # to shape (B, P*L, H, W)
        x = x.reshape((B, P, L, H, W))
        
        return x
