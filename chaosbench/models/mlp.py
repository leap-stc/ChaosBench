import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple single-layer MLP for benchmarking purposes
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        
        return x
