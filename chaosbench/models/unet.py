import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    Simple U-Net architecture
    """
    def __init__(
        self, 
        input_size, 
        output_size
    ):
        super(UNet, self).__init__()

        # Contracting path
        self.enc1 = self.conv_block(input_size, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        self.pool = nn.MaxPool2d(2, 2)

        # Expansive path
        self.up4 = self.upconv_block(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, output_size, 1)

    def conv_block(self, input_size, output_size):
        block = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(negative_slope=0.15),
            nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(negative_slope=0.15)
        )
        return block

    def upconv_block(self, input_size, output_size):
        block = nn.Sequential(
            nn.ConvTranspose2d(input_size, output_size, kernel_size=2, stride=2),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(negative_slope=0.15)
        )
        return block

    def forward(self, x):
        B, P, L, H, W = x.shape
        
        # Contracting path
        enc1 = self.enc1(x.view(B, -1, H, W))
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # Expansive path
        ## Note interpolation/padding in the first and last layer to ensure the sizes match
        ## ie. this is necessary since our inputs are not nicely divisible by 2
        
        up4 = self.up4(enc5)
        up4 = F.interpolate(up4, size=(enc4.shape[-2], enc4.shape[-1]), mode='bilinear', align_corners=True)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        up1 = F.interpolate(up1, size=(H, W), mode='bilinear', align_corners=True)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        out = self.out_conv(dec1)
        out = out.reshape((B, P, L, H, W))
        

        return out
