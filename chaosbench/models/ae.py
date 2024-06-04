import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_bn_relu(input_size, output_size, kernel_size, stride=1, padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride, padding=padding),
        torch.nn.BatchNorm2d(output_size),
        torch.nn.ReLU()
    )
    return convlayer

def deconv_sigmoid(input_size, output_size, kernel_size, stride=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(input_size, output_size, kernel_size=kernel_size, stride=stride),
        torch.nn.BatchNorm2d(output_size),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_relu(input_size, output_size, kernel_size, stride=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(input_size, output_size, kernel_size=kernel_size, stride=stride),
        torch.nn.BatchNorm2d(output_size),
        torch.nn.ReLU()
    )
    return convlayer


class EncoderDecoder(torch.nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size
    ):
        
        super(EncoderDecoder, self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(input_size, 64, 3),
            conv2d_bn_relu(64, 64, 3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(64, 128, 3),
            conv2d_bn_relu(128, 128, 3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(128, 256, 3),
            conv2d_bn_relu(256, 256, 3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(256, 512, 3),
            conv2d_bn_relu(512, 512, 3),
        )
        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(512, 1024, 3),
            conv2d_bn_relu(1024, 1024, 3),
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dec4 = deconv_relu(1024, 512, 2, 2)
        self.dec3 = deconv_relu(515, 256, 2, 2)
        self.dec2 = deconv_relu(259, 128, 2, 2)
        self.dec1 = deconv_relu(131, 64, 2, 2)

        self.predict4 = torch.nn.Conv2d(1024, 3, 1)
        self.predict3 = torch.nn.Conv2d(515, 3, 1)
        self.predict2 = torch.nn.Conv2d(259, 3, 1)
        self.predict1 = torch.nn.Conv2d(131, 3, 1)

        self.up4 = deconv_sigmoid(3, 3, 2, 2)
        self.up3 = deconv_sigmoid(3, 3, 2, 2)
        self.up2 = deconv_sigmoid(3, 3, 2, 2)
        self.up1 = deconv_sigmoid(3, 3, 2, 2)
        
        self.out_conv = nn.Conv2d(67, output_size, 1)


    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(self.pool(conv1_out))
        conv3_out = self.conv_stack3(self.pool(conv2_out))
        conv4_out = self.conv_stack4(self.pool(conv3_out))
        conv5_out = self.conv_stack5(self.pool(conv4_out))
        return conv5_out

    def decoder(self, x):
        dec4 = self.dec4(x)
        up4 = self.up4(self.predict4(x))
        concat4 = torch.cat([dec4, up4],dim=1)
        
        dec3 = self.dec3(concat4)
        up3 = self.up3(self.predict3(concat4))
        concat3 = torch.cat([dec3, up3],dim=1)

        dec2 = self.dec2(concat3)
        up2 = self.up2(self.predict2(concat3))
        concat2 = torch.cat([dec2, up2],dim=1)

        dec1 = self.dec1(concat2)
        up1 = self.up1(self.predict1(concat2))
        concat1 = torch.cat([dec1, up1],dim=1)

        predict_out = self.out_conv(concat1)
        predict_out = F.interpolate(predict_out, size=(121, 240), mode='bilinear', align_corners=True)
        
        return predict_out
        

    def forward(self, x):
        IS_MERGED = False # To handle legacy code where the inputs are separated by pressure level
        
        try:
            B, P, L, H, W = x.shape
            
        except:
            B, P, H, W = x.shape
            IS_MERGED = True
        
        # encoder
        x = self.encoder(x.view(B, -1, H, W))
        
        # decoder
        out = self.decoder(x)
        out = out.reshape((B, P, H, W)) if IS_MERGED else out.reshape((B, P, L, H, W))
        
        return out
    
    

class VAE(torch.nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size,
        latent_size
    ):
        
        super(VAE,self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(input_size, 64, 3),
            conv2d_bn_relu(64, 64, 3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(64, 128, 3),
            conv2d_bn_relu(128, 128, 3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(128, 256, 3),
            conv2d_bn_relu(256, 256, 3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(256, 512, 3),
            conv2d_bn_relu(512, 512, 3),
        )
        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(512, 1024, 3),
            conv2d_bn_relu(1024, 1024, 3),
        )
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc_mu = torch.nn.Sequential(nn.Linear(1024 * 7 * 15, latent_size), nn.Tanh())
        self.fc_logvar = torch.nn.Sequential(nn.Linear(1024 * 7 * 15, latent_size), nn.ReLU())
        self.fc_dec = nn.Linear(latent_size, 1024 * 7 * 15)
        
        self.dec4 = deconv_relu(1024, 512, 2, 2)
        self.dec3 = deconv_relu(515, 256, 2, 2)
        self.dec2 = deconv_relu(259, 128, 2, 2)
        self.dec1 = deconv_relu(131, 64, 2, 2)

        self.predict4 = torch.nn.Conv2d(1024, 3, 1)
        self.predict3 = torch.nn.Conv2d(515, 3, 1)
        self.predict2 = torch.nn.Conv2d(259, 3, 1)
        self.predict1 = torch.nn.Conv2d(131, 3, 1)

        self.up4 = deconv_sigmoid(3, 3, 2, 2)
        self.up3 = deconv_sigmoid(3, 3, 2, 2)
        self.up2 = deconv_sigmoid(3, 3, 2, 2)
        self.up1 = deconv_sigmoid(3, 3, 2, 2)
        
        self.out_conv = nn.Conv2d(67, output_size, 1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(self.pool(conv1_out))
        conv3_out = self.conv_stack3(self.pool(conv2_out))
        conv4_out = self.conv_stack4(self.pool(conv3_out))
        conv5_out = self.conv_stack5(self.pool(conv4_out))
        return conv5_out

    def decoder(self, x):
        dec4 = self.dec4(x)
        up4 = self.up4(self.predict4(x))
        concat4 = torch.cat([dec4, up4],dim=1)
        
        dec3 = self.dec3(concat4)
        up3 = self.up3(self.predict3(concat4))
        concat3 = torch.cat([dec3, up3],dim=1)

        dec2 = self.dec2(concat3)
        up2 = self.up2(self.predict2(concat3))
        concat2 = torch.cat([dec2, up2],dim=1)

        dec1 = self.dec1(concat2)
        up1 = self.up1(self.predict1(concat2))
        concat1 = torch.cat([dec1, up1],dim=1)

        predict_out = self.out_conv(concat1)
        predict_out = F.interpolate(predict_out, size=(121, 240), mode='bilinear', align_corners=True)
        
        return predict_out
        

    def forward(self, x):
        IS_MERGED = False # To handle legacy code where the inputs are separated by pressure level
        
        try:
            B, P, L, H, W = x.shape
            
        except:
            B, P, H, W = x.shape
            IS_MERGED = True
        
        # encoder
        x = self.encoder(x.view(B, -1, H, W))
        
        # variational layer
        mu = self.fc_mu(x.view(B, -1))
        logvar = self.fc_logvar(x.view(B, -1))
        z = self.reparameterize(mu, logvar)
        
        # decoder
        x = self.fc_dec(z)
        out = self.decoder(x.view(B, 1024, 7, 15))
        out = out.reshape((B, P, H, W)) if IS_MERGED else out.reshape((B, P, L, H, W))
        
        return out, mu, logvar