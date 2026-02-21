import torch
from torch import nn
from einops import rearrange
import numpy as np
import random
import torch.nn.functional as F
import pywt

def get_conv(ratio=5, stride=1):
    conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=ratio, stride=stride, padding=0, bias=False)
    conv.weight = torch.nn.Parameter((torch.ones((1,1,ratio, ratio))/(1.0 * ratio * ratio)).cuda())
    return conv

def smooth(noisy, ratio=13, stride=1):
    conv = get_conv(ratio, stride)
    b, c, h, w = noisy.shape
    smoothed = conv(noisy.view(-1, 1, h, w))
    _, _, new_h, new_w = smoothed.shape     
    return smoothed.view(1, c, new_h, new_w).detach()

def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def calculate_sliding_std(img, upsampler, kernel_size, stride):   
    slided_mean = smooth(img, kernel_size, stride=stride)
    mean_upsampled = upsampler(slided_mean)
    variance = smooth( (img - mean_upsampled)**2, kernel_size, stride=stride)
    upsampled_variance = upsampler(variance)
    return upsampled_variance.sqrt()


def shuffle_input(img, indices, mask, c, size, k):
    if c == 1:
        img_torch = torch.from_numpy(img).unsqueeze(0)
    else:
        img_torch = torch.from_numpy(img)
    mask_torch = torch.from_numpy(mask).unsqueeze(0).repeat(c, 1, 1)
    img_torch_rearranged = rearrange(img_torch.unsqueeze(1), 'c 1 (h1 h) (w1 w) -> c (h1 w1) h w ', h1=size//k, w1=size//k) # (c H//k W//k k k)
    mask_torch_rearranged = rearrange(mask_torch.unsqueeze(1), 'c 1 (h1 h) (w1 w) -> c (h1 w1) h w ', h1=size//k, w1=size//k)
    img_torch_rearranged = img_torch_rearranged.view(c, -1, k*k)# (c H//k*W//k k*k)
    mask_torch_rearranged, _ = torch.max(mask_torch_rearranged.view(c, -1, k*k), 2, keepdim=True)
    img_torch_reordered = torch.gather(img_torch_rearranged.clone(), dim=-1, index=indices).clone()
    img_torch_reordered_v2 = img_torch_reordered.view(c, -1, k*k)
    # Shuffle the image only at the flat regions (where mask = 0)
    img_torch_final = mask_torch_rearranged * img_torch_rearranged + (1 - mask_torch_rearranged) * img_torch_reordered_v2
    img_torch_final = img_torch_final.view(c, -1, k, k)    
    img_torch_final_v2 = rearrange(img_torch_final, 'c (h1 w1) h w -> c 1 (h1 h) (w1 w) ', h1=size//k, w1=size//k)
    return img_torch_final_v2.squeeze().cpu().numpy() 

def get_shuffling_mask(std_map_torch, threshold=0.5):
    std_map = std_map_torch.cpu().numpy().squeeze()
    normalized = std_map/std_map.max()
    thresholded = np.zeros_like(normalized)
    thresholded[normalized >= threshold] = 1.
    return thresholded


def generate_random_permutation(img_size, c, factor):
    d1, d2, d3 = c, (img_size//factor)*(img_size//factor), factor*factor
    permutaion_indices = torch.argsort(torch.rand(1, d2, d3), dim=-1)
    permutaion_indices = permutaion_indices.repeat(d1, 1, 1)
    reverse_permutation_indices = torch.argsort(permutaion_indices, dim=-1)

    return permutaion_indices, reverse_permutation_indices


def numpy_to_torch(x):
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)

def torch_to_numpy(x):
    """Convert torch tensor to numpy array."""
    return x.squeeze().cpu().numpy()

def dwt_torch(x):
    """
    Perform 2D Discrete Wavelet Transform using PyWavelets on torch tensor
    Returns LL, LH, HL, HH subbands
    """
    x_np = torch_to_numpy(x)
    coeffs = pywt.dwt2(x_np, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    return (numpy_to_torch(coeff) for coeff in [LL, LH, HL, HH])

def idwt_torch(LL, LH, HL, HH):
    """
    Perform 2D Inverse Discrete Wavelet Transform using PyWavelets
    """
    LL_np = torch_to_numpy(LL)
    LH_np = torch_to_numpy(LH)
    HL_np = torch_to_numpy(HL)
    HH_np = torch_to_numpy(HH)
    
    reconstructed = pywt.idwt2((LL_np, (LH_np, HL_np, HH_np)), 'haar')
    
    return numpy_to_torch(reconstructed)

def noise_suppress_freq_domain(subbands, noise_threshold=0.05, adaptive=True):
    LL, LH, HL, HH = subbands
    
    if adaptive:
        for i, subband in enumerate([LH, HL, HH]):
            subband_abs = torch.abs(subband)
            median = torch.median(subband_abs)
            mean = torch.mean(subband_abs)
            std = torch.std(subband_abs)

            threshold_soft = median + noise_threshold * std
            threshold_hard = mean + std
      
            subband_processed = subband * torch.sigmoid((subband_abs - threshold_soft) / std)

            mask_extreme = subband_abs > threshold_hard
            subband_processed[mask_extreme] *= 0.1
            
            if i == 0:
                LH = subband_processed
            elif i == 1:
                HL = subband_processed
            else:
                HH = subband_processed
    
    return LL, LH, HL, HH
