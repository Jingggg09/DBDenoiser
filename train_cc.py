import torch
from PIL import Image
import numpy as np
from torch import nn
import os
import cv2
import pandas as pd
import yaml
from model import DualBranchDenoiser, PerceptualLoss
from utils import reproduce, calculate_sliding_std, shuffle_input, get_shuffling_mask, generate_random_permutation, dwt_torch, idwt_torch, noise_suppress_freq_domain, torch_to_numpy
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 创建输出目录
if not os.path.exists('output-cc'):
    os.makedirs('output-cc')
if not os.path.exists('output-cc/subbands'):
    os.makedirs('output-cc/subbands')
if not os.path.exists('output-cc/denoised_subbands'):
    os.makedirs('output-cc/denoised_subbands')

# 存储结果的列表
results_cc = []

# 设置随机种子
reproduce(42)

def save_model(model, path, iteration, loss):
    """
    保存模型检查点
    
    Args:
        model (nn.Module): 要保存的模型
        path (str): 保存路径
        iteration (int): 当前迭代次数
        loss (float): 当前损失值
    """
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)

def save_subbands(LL, LH, HL, HH, filename_prefix, output_dir):
    """
    保存分解子带图像
    """
    for name, subband in zip(['LL', 'LH', 'HL', 'HH'], [LL, LH, HL, HH]):
        subband_np = torch_to_numpy(subband) * 255.
        subband_np = np.clip(subband_np, 0, 255).astype(np.uint8)
        if subband_np.shape[0] == 3:  # RGB图像
            subband_np = np.transpose(subband_np, (1, 2, 0))
            cv2.imwrite(f'{output_dir}/{filename_prefix}_{name}.png', cv2.cvtColor(subband_np, cv2.COLOR_RGB2BGR))
        else:  # 单通道图像
            cv2.imwrite(f'{output_dir}/{filename_prefix}_{name}.png', subband_np)

def train_model(model, noisy_original_torch, clean_original_torch, config, filename):
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
    
    # 进行DWT分解并保存原始子带
    LL, LH, HL, HH = dwt_torch(noisy_original_torch)
    save_subbands(LL, LH, HL, HH, f'noisy_{filename}', 'output-cc/subbands')
    
    # 频率域噪声抑制
    LL_denoised, LH_denoised, HL_denoised, HH_denoised = noise_suppress_freq_domain(
        (LL, LH, HL, HH)
    )
    
    # 保存去噪后的子带
    save_subbands(LL_denoised, LH_denoised, HL_denoised, HH_denoised, 
                  f'denoised_{filename}', 'output-cc/denoised_subbands')
    
    # 重建图像
    freq_denoised = idwt_torch(LL_denoised, LH_denoised, HL_denoised, HH_denoised)
    
    for iter in range(config['num_iterations']):
        output, multi_scale_masks = model(noisy_original_torch)
        
        loss = criterion(output, clean_original_torch)
        
        for mask in multi_scale_masks:
            mask_loss = torch.mean(torch.abs(mask - 0.5))
            loss += 0.1 * mask_loss
        
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
                f'best_model_{filename}.pth'
            )
            save_model(model, model_save_path, iter, loss.item())
            print(f"Saved best model at iteration {iter} with loss {loss.item()}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at iteration {iter}")
                break
    
    return model

# 加载CC数据集
noisy_dir = 'data/CC/NOISE'
clean_dir = 'data/CC/GT'
noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith('.png')]

# 加载配置
with open('configs/sidd.yaml', 'r') as f:
    config = yaml.safe_load(f)

for noisy_file in noisy_files:
    # 获取对应的干净图像文件名
    clean_file = noisy_file.replace('_real.png', '_mean.png')
    
    # 加载噪声和干净图像
    noisy_img = np.array(Image.open(os.path.join(noisy_dir, noisy_file)).convert('RGB')).astype(np.float32)
    clean_img = np.array(Image.open(os.path.join(clean_dir, clean_file)).convert('RGB')).astype(np.float32)
    
    # 转换为 torch 张量并确保为float类型
    noisy_original_torch = torch.from_numpy(np.transpose(noisy_img, (2, 0, 1)) / 255.).float().unsqueeze(0).cuda()
    clean_original_torch = torch.from_numpy(np.transpose(clean_img, (2, 0, 1)) / 255.).float().unsqueeze(0).cuda()
    
    # 获取基础名（去掉'_real.png'）
    base_name = noisy_file.replace('_real.png', '')
    
    # 初始化模型
    model = DualBranchDenoiser(in_channels=3).cuda()
    model.train()
    
    # 训练模型
    model = train_model(model, noisy_original_torch, clean_original_torch, config, base_name)
    
    # 推理阶段
    model.eval()
    avg = 0.
    with torch.no_grad():
        for _ in range(config['num_predictions']):
            output, _ = model(noisy_original_torch)
            to_img = output.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            avg += to_img
    
    # 后处理
    denoised_img = np.clip((avg/float(config['num_predictions']))*255., 0., 255.).astype(np.uint8)
    
    # 保存去噪图像
    output_path = f'output-cc/denoised_{base_name}.png'
    cv2.imwrite(output_path, cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR))
    
    # 计算评估指标
    psnr = peak_signal_noise_ratio(clean_img, denoised_img, data_range=255)
    ssim_val = ssim(clean_img, denoised_img, channel_axis=2, data_range=255)
    print(f"Image: {noisy_file}, PSNR: {psnr:.2f}, SSIM: {ssim_val:.4f}")
    
    # 存储结果
    results_cc.append({
        'image': noisy_file,
        'psnr': psnr,
        'ssim': ssim_val
    })

# 保存结果到 CSV
results_df = pd.DataFrame(results_cc)
results_df.to_csv('output-cc/results.csv', index=False)
print(f"Average PSNR: {results_df['psnr'].mean():.2f}")
print(f"Average SSIM: {results_df['ssim'].mean():.4f}")