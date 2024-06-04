import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d_fast(nn.Module):
    def __init__(self, input_size, output_size, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.input_size = input_size
        self.output_size = output_size
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (input_size * output_size))
        self.weights1 = nn.Parameter(self.scale * torch.view_as_real(
            torch.rand(input_size, output_size, self.modes1, self.modes2, dtype=torch.cfloat)
        ))
        self.weights2 = nn.Parameter(self.scale * torch.view_as_real(
            torch.rand(input_size, output_size, self.modes1, self.modes2, dtype=torch.cfloat)
        ))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, input_size, x, y), (input_size, output_size, x, y) -> (batch, output_size, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, torch.view_as_complex(weights))

    def forward(self, x):
        B = x.shape[0]
        
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, self.output_size,  x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, input_size=60, modes1=4, modes2=4, width=[64, 128, 256, 512, 1024], initial_step=1):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2
        
        input: the solution of the previous K timesteps
        input shape: (B, H, W, P*L)
        output: the solution of the next timestep
        output shape: (B, H, W, P*L)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(initial_step*input_size, self.width[0])

        self.conv0 = SpectralConv2d_fast(self.width[0], self.width[1], self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width[1], self.width[2], self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width[2], self.width[3], self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width[3], self.width[4], self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width[0], self.width[1], 1)
        self.w1 = nn.Conv2d(self.width[1], self.width[2], 1)
        self.w2 = nn.Conv2d(self.width[2], self.width[3], 1)
        self.w3 = nn.Conv2d(self.width[3], self.width[4], 1)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.width[4], self.width[3], 1),
            nn.BatchNorm2d(self.width[3]),
            nn.GELU(),
            nn.Conv2d(self.width[3], self.width[2], 1),
            nn.BatchNorm2d(self.width[2]),
            nn.GELU(),
            nn.Conv2d(self.width[2], self.width[1], 1),
            nn.BatchNorm2d(self.width[1]),
            nn.GELU(),
            nn.Conv2d(self.width[1], self.width[0], 1),
            nn.BatchNorm2d(self.width[0]),
            nn.GELU(),
            nn.Conv2d(self.width[0], input_size, 1)
        )
        
    def forward(self, x):
        IS_MERGED = False # To handle legacy code where the inputs are separated by pressure level
        
        try:
            B, P, L, H, W = x.shape
            x = x.permute((0, 3, 4, 1, 2)) # to shape (B, H, W, P, L)
            
        except:
            B, P, H, W = x.shape
            x = x.permute((0, 2, 3, 1)) # to shape (B, H, W, P)
            IS_MERGED = True
        

        x = self.fc0((x.view(B, H, W, -1))) # to shape (B, H, W, P*L)
        x = x.permute(0, 3, 1, 2) # to shape (B, width, H, W)
        
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        ##### Main convolutional steps (in the FT + non-FT) space #####
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        ###############################################################

        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor
        x = self.decoder(x)
        x = x.permute((0, 3, 1, 2)) # to shape (B, P*L, H, W)
        x = x.reshape((B, P, H, W)) if IS_MERGED else x.reshape((B, P, L, H, W))
        
        return x
