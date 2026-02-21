import torch
from PIL import Image
import numpy as np
from torch import nn
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import yaml
import random
from model import DualBranchDenoiser, PerceptualLoss
from utils import reproduce, calculate_sliding_std, shuffle_input, get_shuffling_mask, generate_random_permutation, dwt_torch, idwt_torch, noise_suppress_freq_domain
import os
import cv2
import pandas as pd

if not os.path.exists('output-sidd'):
    os.makedirs('output-sidd')    

results_sidd = []

mat = sio.loadmat('data/ValidationNoisyBlocksSrgb.mat')
noisy_block = mat['ValidationNoisyBlocksSrgb']
mat = sio.loadmat('data/ValidationGtBlocksSrgb.mat')
gt_block = mat['ValidationGtBlocksSrgb']


def save_model(model, path, iteration, loss):
 
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)

def train_model(model, noisy_original_torch, clean_original_torch, config):

    criterion = PerceptualLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config['lr'], 
                                weight_decay=1e-4, 
                                betas=(0.9, 0.999))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['num_iterations'], 
        eta_min=1e-6
    )
    
    best_loss = float('inf')
    patience = 50
    trigger_times = 0

    os.makedirs(config.get('model_save_dir', 'checkpoints'), exist_ok=True)
    
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
        
        loss = criterion(output, clean_original_torch)
        
        for mask in multi_scale_masks:
            mask_loss = torch.mean(torch.abs(mask - 0.5))
            loss += 0.1 * mask_loss
        
        # 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            trigger_times = 0
            
            model_save_path = os.path.join(
                config.get('model_save_dir', 'checkpoints'), 
                f'best_model_block_{index_n}_{index_k}.pth'
            )
            save_model(model, model_save_path, iter, loss.item())
            print(f"Saved best model at iteration {iter} with loss {loss.item()}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at iteration {iter}")
                break
    
    return model



for index_n in range(noisy_block.shape[0]):
    for index_k in range(noisy_block.shape[1]):
        with open('configs/sidd.yaml', 'r') as f:
            config = yaml.safe_load(f)

        model = DualBranchDenoiser(in_channels=3).cuda()
        model.train()       

        noisy_orig_np = np.float32(noisy_block[index_n, index_k])
        clean_orig_np = np.float32(gt_block[index_n, index_k])

        noisy_original_torch = torch.from_numpy(np.transpose(noisy_orig_np, (2, 0, 1)) / 255.).unsqueeze(0).cuda()
        clean_original_torch = torch.from_numpy(np.transpose(clean_orig_np, (2, 0, 1)) / 255.).unsqueeze(0).cuda()

        model = train_model(model, noisy_original_torch, clean_original_torch, config)

        model.eval()
        avg = 0.
        with torch.no_grad():
            for _ in range(config['num_predictions']):
                output, _ = model(noisy_original_torch)
                to_img = output.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                avg += to_img

        # 
        denoised_img = np.clip((avg/float(config['num_predictions']))*255., 0., 255.).astype(np.uint8)

        # 
        cv2.imwrite(f'output-sidd/denoised_{index_n}_{index_k}.png', cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR))
             
        # 
        psnr = peak_signal_noise_ratio(clean_orig_np, denoised_img, data_range=255)
        ssim1 = ssim(clean_orig_np, denoised_img, channel_axis=2, data_range=255)
        print(psnr, ssim1)      
       
        # 
        results_sidd.append({
            'block_n': index_n,
            'block_k': index_k,
            'psnr': psnr,
            'ssim': ssim1
        })

# 
results_df = pd.DataFrame(results_sidd)
results_df.to_csv('output-sidd/results.csv', index=False)
print(f"Average PSNR: {results_df['psnr'].mean():.2f}")
print(f"Average SSIM: {results_df['ssim'].mean():.2f}")
