import torch
from PIL import Image
import numpy as np
from torch import nn
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import yaml
import random
import os
from model import DualBranchDenoiser, MultiScaleLoss, PerceptualLoss
from utils import reproduce, calculate_sliding_std, shuffle_input, get_shuffling_mask, generate_random_permutation, dwt_torch, idwt_torch, noise_suppress_freq_domain
import cv2
import pandas as pd
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "5"   

# 设置随机种子
reproduce(42)

if not os.path.exists('output-fmdd'):
    os.makedirs('output-fmdd')
    
results_fmdd = []

paths = os.listdir('data/FMDD/raw/')
for path in paths:
    # 模型和优化器设置
    torch.cuda.empty_cache()
    model = DualBranchDenoiser(1).cuda()
    model.train()      
    
    with open('configs/fmdd.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 加载图像   
    noisy_orig_np = np.float32(Image.open('data/FMDD/raw/' + path))
    clean_orig_np = np.float32(Image.open('data/FMDD/gt/' + path))

    noisy_original_torch = torch.from_numpy(noisy_orig_np / 255.).unsqueeze(0).unsqueeze(0).cuda()
    clean_original_torch = torch.from_numpy(clean_orig_np / 255.).unsqueeze(0).unsqueeze(0).cuda()


    def train_model(model, noisy_original_torch, clean_orig_torch, config):
        # 改进的损失函数
        criterion = PerceptualLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=config['lr'], 
                                    weight_decay=1e-4, 
                                    betas=(0.9, 0.999))
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['num_iterations'], 
            eta_min=1e-6
        )
        
        best_loss = float('inf')
        patience = 50
        trigger_times = 0
        
        for iter in range(config['num_iterations']):
            # Wavelet Transform
            LL, LH, HL, HH = dwt_torch(noisy_original_torch)
            
            # Noise suppression in frequency domain
            LL_denoised, LH_denoised, HL_denoised, HH_denoised = noise_suppress_freq_domain(
                (LL, LH, HL, HH)
            )
            
            # Reconstruct image
            freq_denoised = idwt_torch(LL_denoised, LH_denoised, HL_denoised, HH_denoised)
            
            # Model forward pass
            output, multi_scale_masks = model(noisy_original_torch)
            
            # 损失计算
            loss = criterion(output, clean_original_torch)
            
            # Mask正则化
            for mask in multi_scale_masks:
                mask_loss = torch.mean(torch.abs(mask - 0.5))
                loss += 0.1 * mask_loss
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Early stopping at iteration {iter}")
                    break
        
        return model

    model = train_model(model, noisy_original_torch, clean_original_torch, config)

    # 推理阶段
    model.eval()
    avg = 0.
    with torch.no_grad():
        for _ in range(config['num_predictions']):
            output, _ = model(noisy_original_torch)
            to_img = output.detach().cpu().squeeze().numpy()
            avg += to_img

    # 后处理
    denoised_img = np.clip((avg/float(config['num_predictions']))*255., 0., 255.).astype(np.uint8)

    cv2.imwrite(f'output-fmdd/denoised_{path}', denoised_img)
        
    # 评估指标
    psnr = peak_signal_noise_ratio(clean_orig_np, denoised_img, data_range=255)
    ssim1 = ssim(clean_orig_np, denoised_img, data_range=255, win_size=3, channel_axis=None)
    print(psnr, ssim1)   

    results_fmdd.append({
        'image_name': path,
        'psnr': psnr,
        'ssim': ssim1
    })

# 结果保存
results_df = pd.DataFrame(results_fmdd)
results_df.to_csv('output-fmdd/results.csv', index=False)
print(f"Average PSNR: {results_df['psnr'].mean():.2f}")
print(f"Average SSIM: {results_df['ssim'].mean():.2f}")