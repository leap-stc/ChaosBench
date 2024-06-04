import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_size),
        nn.LeakyReLU(negative_slope=0.15),
        nn.Conv2d(output_size, output_size, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_size),
        nn.LeakyReLU(negative_slope=0.15)
    )
    return block

def upconv_block(input_size, output_size, kernel_size=2, stride=2):
    block = nn.Sequential(
        nn.ConvTranspose2d(input_size, output_size, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(output_size),
        nn.LeakyReLU(negative_slope=0.15)
    )
    return block

class UNet(nn.Module):
    """
    U-Net architecture (5 blocks)
    """
    
    def __init__(
        self, 
        input_size, 
        output_size
    ):
        super(UNet, self).__init__()

        # Contracting path
        self.enc1 = conv_block(input_size, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.enc5 = conv_block(512, 1024)

        self.pool = nn.MaxPool2d(2, 2)

        # Expansive path
        self.up4 = upconv_block(1024, 512)
        self.dec4 = conv_block(1024, 512)
        self.up3 = upconv_block(512, 256)
        self.dec3 = conv_block(512, 256)
        self.up2 = upconv_block(256, 128)
        self.dec2 = conv_block(256, 128)
        self.up1 = upconv_block(128, 64)
        self.dec1 = conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, output_size, 1)

    def forward(self, x):
        IS_MERGED = False # To handle legacy code where the inputs are separated by pressure level
        
        try:
            B, P, L, H, W = x.shape
            
        except:
            B, P, H, W = x.shape
            IS_MERGED = True
        
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
        out = out.reshape((B, P, H, W)) if IS_MERGED else out.reshape((B, P, L, H, W))
        
        return out
    
class ResNet(nn.Module):
    """
    ResNet backbone architecture + deconv layers similar to UNet without the skip connection
    """
    
    def __init__(
        self, 
        input_size, 
        output_size
    ):
        super(ResNet, self).__init__()

        self.enc = timm.create_model('resnet50', in_chans=input_size, features_only=True)

        # Expansive path
        self.up5 = upconv_block(2048, 1024)
        self.up4 = upconv_block(1024, 512)
        self.up3 = upconv_block(512, 256)
        self.up2 = upconv_block(256, 128)
        self.up1 = upconv_block(128, 64)
        
        self.out_conv = nn.Conv2d(64, output_size, 1)



    def forward(self, x):
        IS_MERGED = False # To handle legacy code where the inputs are separated by pressure level
        
        try:
            B, P, L, H, W = x.shape
            
        except:
            B, P, H, W = x.shape
            IS_MERGED = True
        
        # Contracting path
        enc = self.enc(x.view(B, -1, H, W))

        # Expansive path
        ## Note interpolation/padding in the first and last layer to ensure the sizes match
        ## ie. this is necessary since our inputs are not nicely divisible by 2
        
        up5 = self.up5(enc[-1])
        up4 = self.up4(up5)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        up1 = F.interpolate(up1, size=(H, W), mode='bilinear', align_corners=True)
        
        out = self.out_conv(up1)
        out = out.reshape((B, P, H, W)) if IS_MERGED else out.reshape((B, P, L, H, W))
        
        return out