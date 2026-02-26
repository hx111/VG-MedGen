import torch.nn as nn
import torch
def normalize(x, norm_type):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(x)
    else:
        return nn.BatchNorm2d(x) #temp

def deconv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
    )

def conv_lrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv_bn_lrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True)
    )

def conv_no_activ(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

def conv_id_unet(in_channels, out_channels, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True)
    )

def upconv(in_channels, out_channels, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        normalize(out_channels, norm)
    )

def conv_block_unet(in_channels, out_channels, kernel_size, stride=1, padding=0, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True),
    )

def conv_block_unet_last(in_channels, out_channels, kernel_size, stride=1, padding=0, norm='batchnorm'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
    )

def conv_preactivation_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, norm='batchnorm'):
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        normalize(out_channels, norm)
    )


class ResConv(nn.Module):
    def __init__(self, ndf, norm):
        super(ResConv, self).__init__()
        """
        Args:
            ndf: constant number from channels
        """
        self.ndf = ndf
        self.norm = norm
        self.conv1 = conv_preactivation_relu(self.ndf, self.ndf * 2, 3, 1, 1, self.norm)
        self.conv2 = conv_preactivation_relu(self.ndf * 2 , self.ndf * 2, 3, 1, 1, self.norm)
        self.resconv = conv_preactivation_relu(self.ndf , self.ndf * 2, 1, 1, 0, self.norm)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.resconv(residual)

        return out + residual


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        """
        Args:
            size: expected size after interpolation
            mode: interpolation type (e.g. bilinear, nearest)
        """
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        out = self.interp(x, size=self.size, mode=self.mode)
        
        return out


class AdaIN(nn.Module):
    def __init__(self, style_dim, content_dim):
        super().__init__()
        self.style_mlp = nn.Linear(style_dim, content_dim * 2)

    def forward(self, content_feat, style_feat):
        eps = 1e-5
        content_mean = torch.mean(content_feat, dim=[2, 3], keepdim=True)
        content_var = torch.var(content_feat, dim=[2, 3], keepdim=True) + eps
        content_std = torch.sqrt(content_var)
        normalized_feat = (content_feat - content_mean) / content_std

        modulation_params = self.style_mlp(style_feat)
        B, C = content_feat.shape[:2]
        gamma, beta = torch.chunk(modulation_params, 2, dim=1)

        output = normalized_feat * gamma.view(B, C, 1, 1) + beta.view(B, C, 1, 1)

        return output