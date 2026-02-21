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

    b, c, h, w = noisy.shape
    conv = get_conv(ratio, stride)
    smoothed = conv(noisy.view(b * c, 1, h, w))  
    _, _, new_h, new_w = smoothed.shape
    smoothed = smoothed.view(b, c, new_h, new_w)        
    
    return smoothed.detach()

def reproduce(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def calculate_sliding_std(img, kernel_size=2):
    # 
    upsampler = nn.Upsample(size=(img.shape[-2], img.shape[-1]), mode='bilinear', align_corners=True)
    slided_mean = smooth(img, kernel_size, stride=1)
    mean_upsampled = upsampler(slided_mean)
    variance = smooth((img - mean_upsampled)**2, kernel_size, stride=1)
    upsampled_variance = upsampler(variance)
    return upsampled_variance.sqrt()

def get_shuffling_mask(std_map_torch, threshold=0.5):
    #  
    std_map = std_map_torch.mean(dim=1, keepdim=True) #  
    max_val = std_map.view(std_map.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    normalized = std_map / (max_val + 1e-8)
    binary_mask = (normalized >= threshold).float()
    return binary_mask

def shuffle_input(img_torch, indices, binary_mask, k=4):
    #  
    b, c, h, w = img_torch.shape
    #  
    img_rearranged = rearrange(img_torch, 'b c (h1 h) (w1 w) -> b (h1 w1) c (h w)', h=k, w=k)
    #  
    img_reordered = torch.gather(img_rearranged, dim=-1, index=indices.expand(-1, -1, c, -1))
    #  
    img_reordered = rearrange(img_reordered, 'b (h1 w1) c (h w) -> b c (h1 h) (w1 w)', h1=h//k, w1=w//k, h=k, w=k)
    
    img_final = binary_mask * img_torch + (1 - binary_mask) * img_reordered
    return img_final

def generate_random_permutation(b, h, w, k, device):
   
    num_blocks = (h // k) * (w // k)
    num_pixels = k * k
    indices = torch.stack([torch.randperm(num_pixels) for _ in range(b * num_blocks)]).to(device)
    indices = indices.view(b, num_blocks, 1, num_pixels)
    return indices

def numpy_to_torch(x):
    if len(x.shape) == 3: # HWC
        x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).float().unsqueeze(0).cuda()

def torch_to_numpy(x):
    return x.detach().cpu().squeeze().numpy()

def dwt_torch(x):  
    x_np = torch_to_numpy(x)
    #
    if len(x_np.shape) == 3:
        ll_list, lh_list, hl_list, hh_list = [], [], [], []
        for c in range(x_np.shape[0]):
            coeffs = pywt.dwt2(x_np[c], 'haar')
            ll, (lh, hl, hh) = coeffs
            ll_list.append(ll); lh_list.append(lh); hl_list.append(hl); hh_list.append(hh)
        return (numpy_to_torch(np.array(l)) for l in [ll_list, lh_list, hl_list, hh_list])
    else:
        coeffs = pywt.dwt2(x_np, 'haar')
        ll, (lh, hl, hh) = coeffs
        return (numpy_to_torch(coeff) for coeff in [ll, lh, hl, hh])

def idwt_torch(LL, LH, HL, HH):   
    ll_np, lh_np, hl_np, hh_np = (torch_to_numpy(x) for x in [LL, LH, HL, HH])
    if len(ll_np.shape) == 3:
        recon_list = []
        for c in range(ll_np.shape[0]):
            recon = pywt.idwt2((ll_np[c], (lh_np[c], hl_np[c], hh_np[c])), 'haar')
            recon_list.append(recon)
        return numpy_to_torch(np.array(recon_list))
    else:
        recon = pywt.idwt2((ll_np, (lh_np, hl_np, hh_np)), 'haar')
        return numpy_to_torch(recon)

def noise_suppress_freq_domain(subbands, noise_threshold=0.05):   
    LL, LH, HL, HH = subbands
    processed_list = []
    
    for subband in [LH, HL, HH]:
        subband_abs = torch.abs(subband)
        std_val = torch.std(subband)
        median_val = torch.median(subband_abs)
        T_S = median_val + noise_threshold * std_val
                
        T_H = torch.mean(subband_abs) + std_val
        
        soft_mask = torch.sigmoid((subband_abs - T_S) / (std_val + 1e-8))
        hard_mask = torch.ones_like(subband)
        hard_mask[subband_abs > T_H] = 0.1
        
        processed_list.append(subband * soft_mask * hard_mask)
        
    return LL, processed_list[0], processed_list[1], processed_list[2]
