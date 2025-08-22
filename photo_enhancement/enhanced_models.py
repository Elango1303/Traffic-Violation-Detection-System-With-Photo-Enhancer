import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import vgg19
import math

class SpectralNorm(nn.Module):
    """Spectral normalization for stable training"""
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = F.normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data)
        v.data = F.normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class EnhancedFeatureExtractor(nn.Module):
    """Enhanced feature extractor with multiple VGG layers"""
    def __init__(self):
        super(EnhancedFeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        
        # Extract multiple feature levels
        self.vgg19_conv1 = nn.Sequential(*list(vgg19_model.features.children())[:8])   # conv1_2
        self.vgg19_conv2 = nn.Sequential(*list(vgg19_model.features.children())[8:17]) # conv2_2  
        self.vgg19_conv3 = nn.Sequential(*list(vgg19_model.features.children())[17:26]) # conv3_2
        self.vgg19_conv4 = nn.Sequential(*list(vgg19_model.features.children())[26:35]) # conv4_2
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, img):
        conv1 = self.vgg19_conv1(img)
        conv2 = self.vgg19_conv2(conv1)
        conv3 = self.vgg19_conv3(conv2)
        conv4 = self.vgg19_conv4(conv3)
        return {'conv1': conv1, 'conv2': conv2, 'conv3': conv3, 'conv4': conv4}

# Keep the original FeatureExtractor for backward compatibility
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, img):
        return self.vgg19_54(img)

class EnhancedDenseResidualBlock(nn.Module):
    """Enhanced Dense Residual Block with attention"""
    def __init__(self, filters, res_scale=0.2, use_attention=True):
        super(EnhancedDenseResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.use_attention = use_attention

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        
        if self.use_attention:
            self.attention = CBAM(filters)

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        
        out = out.mul(self.res_scale)
        if self.use_attention:
            out = self.attention(out)
        
        return out + x

class DenseResidualBlock(nn.Module):
    """Original Dense Residual Block for backward compatibility"""
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x

class EnhancedResidualInResidualDenseBlock(nn.Module):
    """Enhanced RRDB with attention mechanism"""
    def __init__(self, filters, res_scale=0.2, use_attention=True):
        super(EnhancedResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            EnhancedDenseResidualBlock(filters, use_attention=use_attention),
            EnhancedDenseResidualBlock(filters, use_attention=use_attention),
            EnhancedDenseResidualBlock(filters, use_attention=use_attention)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class ResidualInResidualDenseBlock(nn.Module):
    """Original RRDB for backward compatibility"""
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
            DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class EnhancedGeneratorRRDB(nn.Module):
    """Enhanced Generator with attention and improved upsampling"""
    def __init__(self, channels, filters=64, num_res_blocks=23, num_upsample=2, 
                 use_attention=True, use_spectral_norm=False):
        super(EnhancedGeneratorRRDB, self).__init__()
        self.use_attention = use_attention

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        
        # Residual blocks with optional attention
        if use_attention:
            res_blocks = [EnhancedResidualInResidualDenseBlock(filters, use_attention=True) 
                         for _ in range(num_res_blocks)]
        else:
            res_blocks = [ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)]
        
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        
        # Enhanced upsampling layers with sub-pixel convolution
        upsample_layers = []
        for i in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2),
            ]
            # Add attention after each upsampling
            if use_attention:
                upsample_layers.append(CBAM(filters))
        
        self.upsampling = nn.Sequential(*upsample_layers)
        
        # Final output block with better activation
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Better for normalized outputs
        )
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            self.apply_spectral_norm()

    def apply_spectral_norm(self):
        """Apply spectral normalization to all conv layers"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module = SpectralNorm(module)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

# Keep original generator for backward compatibility
class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class EnhancedDiscriminator(nn.Module):
    """Enhanced Discriminator with spectral normalization and self-attention"""
    def __init__(self, input_shape, use_spectral_norm=True, use_self_attention=True):
        super(EnhancedDiscriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)
        self.use_self_attention = use_self_attention

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)
            if use_spectral_norm:
                conv = SpectralNorm(conv)
            layers.append(conv)
            
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)
            if use_spectral_norm:
                conv2 = SpectralNorm(conv2)
            layers.append(conv2)
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            return layers

        layers = []
        in_filters = in_channels
        filter_sizes = [64, 128, 256, 512]
        
        for i, out_filters in enumerate(filter_sizes):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            
            # Add self-attention after the second block
            if use_self_attention and i == 1:
                layers.append(SelfAttention(out_filters))
            
            in_filters = out_filters

        # Final conv layer
        final_conv = nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = SpectralNorm(final_conv)
        layers.append(final_conv)

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

class SelfAttention(nn.Module):
    """Self-attention module for discriminator"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Attention
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        out = self.gamma * out + x
        return out

# Keep original discriminator for backward compatibility
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

# Loss functions
class PerceptualLoss(nn.Module):
    """Enhanced perceptual loss using multiple VGG layers"""
    def __init__(self, feature_extractor, layer_weights=None):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights or {'conv1': 1.0, 'conv2': 1.0, 'conv3': 1.0, 'conv4': 1.0}
        self.criterion = nn.L1Loss()

    def forward(self, sr_images, hr_images):
        sr_features = self.feature_extractor(sr_images)
        hr_features = self.feature_extractor(hr_images)
        
        loss = 0
        for layer_name, weight in self.layer_weights.items():
            if layer_name in sr_features and layer_name in hr_features:
                loss += weight * self.criterion(sr_features[layer_name], hr_features[layer_name])
        
        return loss

class TVLoss(nn.Module):
    """Total Variation Loss for smoothness"""
    def __init__(self, weight=1e-6):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
