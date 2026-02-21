import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarDWT(nn.Module):
    def __init__(self, in_channels=3):
        super(HaarDWT, self).__init__()
        self.in_channels = in_channels

        kernel = torch.tensor([
            [[1, 1], [1, 1]],  # LL 
            [[1, -1], [1, -1]], # LH 
            [[1, 1], [-1, -1]], # HL 
            [[1, -1], [-1, 1]]  # HH 
        ], dtype=torch.float32).view(4, 1, 2, 2) / 2.0
        
        weight = kernel.repeat(in_channels, 1, 1, 1)     # [4*in_channels, 1, 2, 2]
        self.register_buffer('weight', weight)

    def forward(self, x):
        out = F.conv2d(x, self.weight, stride=2, groups=self.in_channels)
        B, C4, H, W = out.shape
        out = out.view(B, self.in_channels, 4, H, W)
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

class HaarIDWT(nn.Module):
    def __init__(self, in_channels=3):
        super(HaarIDWT, self).__init__()
        self.in_channels = in_channels
        kernel = torch.tensor([
            [[1, 1], [1, 1]], 
            [[1, -1], [1, -1]], 
            [[1, 1], [-1, -1]], 
            [[1, -1], [-1, 1]]
        ], dtype=torch.float32).view(4, 1, 2, 2) / 2.0
        self.register_buffer('weight', kernel.repeat(in_channels, 1, 1, 1))#.unsqueeze(1))

    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        x = torch.stack([LL, LH, HL, HH], dim=2).view(B, C*4, H, W)
        
        out = F.conv_transpose2d(
            x, 
            self.weight, 
            stride=2, 
            groups=self.in_channels,
            output_padding=0
        )     
        return out

class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=3):
        super(FrequencyBranch, self).__init__()
        self.dwt = HaarDWT(in_channels)
        self.idwt = HaarIDWT(in_channels)
        
        self.alpha = nn.Parameter(torch.full((1,), 0.05))
        
        self.refine_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1), 
            ResBlock(32, 3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            ResBlock(64, 3),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        )

    def adaptive_threshold_process(self, S):
        S_abs = torch.abs(S)
        std_S = torch.std(S, dim=(2,3), keepdim=True)
        B, C, H, W = S_abs.shape
        S_abs_flat = S_abs.view(B, C, H * W)          # [B, C, H*W]
        median_abs, _ = torch.median(S_abs_flat, dim=2, keepdim=True)  # [B, C, 1]
        median_abs = median_abs.unsqueeze(-1)          # [B, C, 1, 1] 

        mean_abs = torch.mean(S_abs, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        T_S = median_abs + self.alpha * std_S
        T_H = mean_abs + std_S

        M_hard = torch.where(S_abs > T_H,
                             torch.tensor(0.1, device=S.device, dtype=S.dtype),
                             torch.tensor(1.0, device=S.device, dtype=S.dtype))

        soft_weight = torch.sigmoid((S_abs - T_S) / (std_S + 1e-6))
        S_prime = S * soft_weight * M_hard

        return S_prime

    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)
        
        LH_p = self.adaptive_threshold_process(LH)
        HL_p = self.adaptive_threshold_process(HL)
        HH_p = self.adaptive_threshold_process(HH)
        
        I_f = self.idwt(LL, LH_p, HL_p, HH_p)
        
        return I_f + self.refine_cnn(I_f)

class MultiScaleSpatialBranch(nn.Module):
    def __init__(self, in_channels=3, scales=3):
        super(MultiScaleSpatialBranch, self).__init__()
        self.scales = scales
        
        self.scale_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32 * (2**i), kernel_size=3, padding=1),
                nn.BatchNorm2d(32 * (2**i)),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2) 
            ) for i in range(scales)
        ])
        
        self.mask_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32 * (2**i), 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            ) for i in range(scales)
        ])
        
        self.trunk_network = UNet_n2n_un(in_channels, in_channels)

    def forward(self, x):
        multi_scale_masks = []
        current_input = x
        
        for i in range(self.scales):
            feat = self.scale_modules[i](current_input)
            mask = self.mask_generators[i](feat)
            aligned_mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
            multi_scale_masks.append(aligned_mask)
            current_input = F.interpolate(x, scale_factor=1/(2**(i+1)), mode='bilinear', align_corners=False)

        combined_mask = torch.mean(torch.stack(multi_scale_masks), dim=0)
        
        return self.trunk_network(x * combined_mask), multi_scale_masks

class DualBranchDenoiser(nn.Module):
    def __init__(self, in_channels=3):
        super(DualBranchDenoiser, self).__init__()
        self.frequency_branch = FrequencyBranch(in_channels)
        self.spatial_branch = MultiScaleSpatialBranch(in_channels)
        
        self.fusion_module = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        f_freq = self.frequency_branch(x)
        f_spatial, masks = self.spatial_branch(x)
        
        return self.fusion_module(torch.cat([f_freq, f_spatial], dim=1)), masks

class ResBlock(nn.Module):
    def __init__(self, nf, ksize):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, ksize, 1, ksize//2),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, ksize, 1, ksize//2),
            nn.BatchNorm2d(nf)
        )
    def forward(self, x):
        return F.leaky_relu(x + self.body(x), 0.1)
        

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

        self._init_weights()  #  

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

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
    def __init__(self, lambda_ssim=0.5):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.ssim_loss = SSIMLoss()
    def forward(self, pred, target):
        return F.l1_loss(pred, target) + self.lambda_ssim * self.ssim_loss(pred, target)
