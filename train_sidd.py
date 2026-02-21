import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader, random_split
from model import DualBranchDenoiser, PerceptualLoss
from utils import reproduce, calculate_sliding_std, shuffle_input, get_shuffling_mask, generate_random_permutation
import os
import cv2
import pandas as pd
import yaml
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

if not os.path.exists('output-sidd'):
    os.makedirs('output-sidd')
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')


mat_noisy = sio.loadmat('data/ValidationNoisyBlocksSrgb.mat')
mat_gt    = sio.loadmat('data/ValidationGtBlocksSrgb.mat')

noisy_5d = mat_noisy['ValidationNoisyBlocksSrgb'].astype(np.float32)   
gt_5d    = mat_gt['ValidationGtBlocksSrgb'].astype(np.float32)


noisy_blocks = noisy_5d.reshape(-1, *noisy_5d.shape[2:])   # [1280, 256, 256, 3]
gt_blocks    = gt_5d.reshape(-1, *gt_5d.shape[2:])

total_blocks = noisy_blocks.shape[0]  # 1280
print(f"Loaded and flattened: {total_blocks} blocks, shape={noisy_blocks.shape}")


class SIDDValDataset(Dataset):
    def __init__(self, noisy_blocks, gt_blocks, indices):
        self.noisy_blocks = noisy_blocks[indices]
        self.gt_blocks    = gt_blocks[indices]
        self.indices      = indices
        
        print(f"Dataset init: noisy_blocks shape = {self.noisy_blocks.shape}")  
        print(f"First block shape: {self.noisy_blocks[0].shape if len(self.noisy_blocks)>0 else 'empty'}")
    def __len__(self):
        return len(self.indices) 
    def __getitem__(self, idx):
        noisy_np = self.noisy_blocks[idx] / 255.0
        gt_np    = self.gt_blocks[idx] / 255.0
        
        assert noisy_np.shape == (256, 256, 3), f"Unexpected noisy shape: {noisy_np.shape}"
        assert gt_np.shape    == (256, 256, 3), f"Unexpected gt shape: {gt_np.shape}"
        
        noisy = torch.from_numpy(noisy_np.transpose(2, 0, 1)).float()
        gt    = torch.from_numpy(gt_np.transpose(2, 0, 1)).float()
        
        return noisy, gt, self.indices[idx]


# reproduce(42)  # 
indices = list(range(total_blocks))
random.shuffle(indices)

train_size = 1000    
train_indices = indices[:train_size]
val_indices   = indices[train_size:]

train_dataset = SIDDValDataset(noisy_blocks, gt_blocks, train_indices)
val_dataset   = SIDDValDataset(noisy_blocks, gt_blocks, val_indices)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False,  num_workers=4)

 
def train_model(model, train_loader, val_loader, config):
    model.train()
    criterion = PerceptualLoss(lambda_ssim=config['lambda_ssim']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_iterations'], eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    while global_step < config['num_iterations'] * 5:   
        for noisy, gt, _ in train_loader:
            noisy = noisy.cuda()  # [B,C,H,W]
            gt    = gt.cuda()

           
            std_map = calculate_sliding_std(noisy, kernel_size=config['std_kernel_size'])
            binary_mask = get_shuffling_mask(std_map, threshold=config['masking_threshold'])
            shuffle_indices = generate_random_permutation(
                noisy.shape[0], noisy.shape[2], noisy.shape[3],
                config['shuffling_tile_size'], noisy.device
            )
            pseudo_target = shuffle_input(noisy, shuffle_indices, binary_mask, k=config['shuffling_tile_size'])

            output, _ = model(noisy)
            loss = criterion(output, pseudo_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            scheduler.step()

            global_step += 1

            
            if global_step % 200 == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for v_noisy, v_gt, _ in val_loader:
                        v_noisy = v_noisy.cuda()
                        v_gt    = v_gt.cuda()
                        v_output, _ = model(v_noisy)
                        val_loss += criterion(v_output, v_gt).item()   
                val_loss /= len(val_loader)
                print(f"Step {global_step} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), 'checkpoints/best_model_all_val.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= config['early_stopping_patience']:
                        print("Early stopping triggered")
                        return model

                model.train()

            if global_step >= config['num_iterations'] * 5:
                break

    return model

 
def main():
    with open('configs/sidd.yaml', 'r') as f:
        config = yaml.safe_load(f)

    reproduce(config['seed'])

    model = DualBranchDenoiser(in_channels=config['input_channels']).cuda()
    print(f"Training on {len(train_indices)} blocks, validating on {len(val_indices)} blocks")

    model = train_model(model, train_loader, val_loader, config)

    
    model.load_state_dict(torch.load('checkpoints/best_model_all_val.pth'))
    model.eval()

    results = []
    with torch.no_grad():
        for batch_idx, (noisy, gt, orig_idx) in enumerate(val_loader):
            noisy = noisy.cuda()
            accumulated = 0.0
            
            for _ in range(config['num_predictions']):  #  
                out, _ = model(noisy)
                accumulated += out.detach().cpu()
            
            avg_out = (accumulated / config['num_predictions'])  # [B, C, H, W]
            
            for i in range(avg_out.shape[0]):
                single_denoised = avg_out[i].permute(1, 2, 0).numpy()
                single_denoised = np.clip(single_denoised * 255., 0, 255).astype(np.uint8)
                
                single_gt = gt[i].permute(1, 2, 0).numpy() * 255.
                single_gt = single_gt.astype(np.uint8)
                
                psnr_val = peak_signal_noise_ratio(single_gt, single_denoised, data_range=255)
                ssim_val = ssim(single_gt, single_denoised, channel_axis=2, data_range=255)
                
                block_id = orig_idx[i].item()
                results.append({'block_idx': block_id, 'psnr': psnr_val, 'ssim': ssim_val})
                print(f"Block {block_id} - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
                
                cv2.imwrite(f'output-sidd/denoised_val_{block_id}.png', cv2.cvtColor(single_denoised, cv2.COLOR_RGB2BGR))

    df = pd.DataFrame(results)
    df.to_csv('output-sidd/final_results_one_model.csv', index=False)
    print(f"Mean PSNR on validation set: {df['psnr'].mean():.2f}")
    print(f"Mean SSIM on validation set: {df['ssim'].mean():.4f}")

if __name__ == "__main__":
    main()
