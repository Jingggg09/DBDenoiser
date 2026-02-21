import torch
import torch.nn as nn
import torch.nn.functional as F
class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=1):
        super(FrequencyBranch, self).__init__()

        self.channel_expand = nn.Conv2d(in_channels, 32, kernel_size=1)
        
        self.freq_conv = nn.Sequential(
            ResBlock(32, 3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            ResBlock(64, 3),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):

        x_expanded = self.channel_expand(x)

        freq_result = self.freq_conv(x_expanded) + x
        
        return freq_result

class MultiScaleSpatialBranch(nn.Module):
    def __init__(self, in_channels=1, scales=3):
        super(MultiScaleSpatialBranch, self).__init__()
        
        self.scales = scales

        self.scale_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 32 * (2**i), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32 * (2**i)),
                nn.LeakyReLU(0.1)
            ) for i in range(scales)
        ])

        self.mask_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32 * (2**i), 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            ) for i in range(scales)
        ])
        
        total_channels = sum(32 * (2**i) for i in range(scales))
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(total_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )  

        self.trunk_network = UNet_n2n_un(in_channels, in_channels)
    
    def forward(self, x, mask_ratio=0.5):

        multi_scale_features = []
        multi_scale_masks = []
        
        current_input = x
        for scale_module, mask_generator in zip(self.scale_modules, self.mask_generators):

            feature = scale_module(current_input)
            multi_scale_features.append(feature)

            mask = mask_generator(feature)
            multi_scale_masks.append(mask)
            
            current_input = F.interpolate(current_input, scale_factor=0.5, mode='bilinear', align_corners=False)

        max_size = multi_scale_features[0].shape[-2:]
        aligned_features = []
        aligned_masks = []
        for feature, mask in zip(multi_scale_features, multi_scale_masks):
            aligned_feature = F.interpolate(feature, size=max_size, mode='bilinear', align_corners=False)
            aligned_mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
            aligned_masks.append(aligned_mask)
            aligned_features.append(aligned_feature)
        
        fused_feature = torch.cat(aligned_features, dim=1)
        fused_feature = self.feature_fusion(fused_feature)

        combined_mask = torch.mean(torch.stack(aligned_masks), dim=0)
        
        if combined_mask.shape != x.shape:
            combined_mask = F.interpolate(combined_mask, size=x.shape[-2:], mode='bilinear', align_corners=False)

        masked_input = x * combined_mask
        denoised_output = self.trunk_network(masked_input)
        
        return denoised_output, multi_scale_masks

class DualBranchDenoiser(nn.Module):
    def __init__(self, in_channels=1):
        super(DualBranchDenoiser, self).__init__()
        
        self.frequency_branch = FrequencyBranch(in_channels)
        self.spatial_branch = MultiScaleSpatialBranch(in_channels)
        
        # Feature fusion module
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, mask_ratio=0.5):
        # Frequency domain processing
        freq_result = self.frequency_branch(x)
        
        # Spatial domain processing
        spatial_result, multi_scale_masks = self.spatial_branch(x, mask_ratio)
        
        # Fusion
        fused_result = torch.cat([freq_result, spatial_result], dim=1)
        final_output = self.fusion_conv(fused_result)
        
        return final_output, multi_scale_masks

class UNet_n2n_un(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet_n2n_un, self).__init__()

        self.en_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2))

        self.en_block5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 48, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block3 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block4 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.de_block5 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(64, 32, 3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            nn.Conv2d(32, out_channels, 3, padding=1, bias=True))

        # Initialize weights
        # self._init_weights()

    # def _init_weights(self):
    #     """Initializes weights using He et al. (2015)."""
    #     for m in self.modules():
    #         if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data)
    #             m.bias.data.zero_()

    def forward(self, x):
        pool1 = self.en_block1(x)
        pool2 = self.en_block2(pool1)
        pool3 = self.en_block3(pool2)
        pool4 = self.en_block4(pool3)
        upsample5 = self.en_block5(pool4)

        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.de_block1(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self.de_block2(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self.de_block3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self.de_block4(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        out = self.de_block5(concat1)

        return out

class ResBlock(nn.Module):
    def __init__(self, nf, ksize, norm=nn.BatchNorm2d, act=nn.LeakyReLU):
        super().__init__()
        
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, ksize, 1, ksize//2),
            norm(nf),
            act(0.1),
            nn.Conv2d(nf, nf, ksize, 1, ksize//2),
            norm(nf),
            act(0.1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(nf, nf, 1),
            norm(nf)
        )
    
    def forward(self, x):
        return F.leaky_relu(x + self.body(x) + self.shortcut(x), 0.1)

class MultiScaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
    
    def forward(self, pred, target):

        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return l1 + 0.5 * (1 - ssim)

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img1, img2):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        
        sigma1_sq = F.avg_pool2d(img1 ** 2, 3, 1, 1) - mu1 ** 2
        sigma2_sq = F.avg_pool2d(img2 ** 2, 3, 1, 1) - mu2 ** 2
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1 * mu2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim.mean()

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim_loss = SSIMLoss()
    
    def forward(self, pred, target):

        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        ssim_val = self.ssim_loss(pred, target)

        l1_loss = F.l1_loss(pred, target)
        
        return l1_loss + 0.5 * ssim_val  
