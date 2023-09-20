import torch
import torch.nn as nn

class RMSE(nn.Module):
    """
    Compute root mean squared error (RMSE)
    """
    
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, predictions, targets):
        
        # Calculate the squared differences between predictions and targets
        squared_diff = (predictions - targets) ** 2
        
        # Calculate the mean squared error
        mean_squared_error = torch.nanmean(squared_diff)
        
        # Take the square root to get the RMSE
        rmse = torch.sqrt(mean_squared_error)
        
        return rmse
    
class MSE(nn.Module):
    """
    Compute mean squared error (MSE)
    """
    
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, predictions, targets):
        
        # Calculate the squared differences between predictions and targets
        squared_diff = (predictions - targets) ** 2
        
        # Calculate the mean squared error
        mean_squared_error = torch.nanmean(squared_diff)
        
        return mean_squared_error
