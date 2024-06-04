import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-layer Perceptron
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
        # To handle legacy code where the inputs are separated by pressure level
        try:
            B, P, L, H, W = x.shape
            x = x.permute((0, 3, 4, 1, 2)) # to shape (B, H, W, P, L)
            x = self.model(x.view(B, H, W, -1)) # to shape (B, H, W, P*L)
            x = x.permute((0, 3, 1, 2)) # to shape (B, P*L, H, W)
            x = x.reshape((B, P, L, H, W))
            
        except:
            B, P, H, W = x.shape
            x = x.permute((0, 2, 3, 1)) # to shape (B, H, W, P)
            x = self.model(x) # to shape (B, H, W, P)
            x = x.permute((0, 3, 1, 2)) # to shape (B, P, H, W)
            
        return x
