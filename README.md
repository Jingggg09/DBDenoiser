# DBDenoiser

## 🚀 Key Features
Dual-Branch Architecture: Combines a FrequencyBranch for global noise suppression and a MultiScaleSpatialBranch for local detail preservation.

Wavelet Integration: Uses 2D Haar Discrete Wavelet Transform (DWT) to process subbands (LL, LH, HL, HH) for adaptive noise suppression.

Multi-Scale Masking: Generates learned masks at different scales to focus the denoising process on relevant image features.

Perceptual Loss: Utilizes a combination of L1 loss and SSIM (Structural Similarity) to ensure high-quality visual results.

## 📂 Project Structure
├── configs/

│   └── sidd.yaml           # Hyperparameters (lr, iterations, mask thresholds)

├── data/                   # Directory for SIDD .mat validation files

├── model.py                # Core architecture: DualBranchDenoiser, UNet, and Losses

├── train_sidd.py           # Training and evaluation logic for SIDD blocks

├── utils.py                # Wavelet transforms and shuffling utilities

├── output-sidd/            # Directory for denoised images and results.csv

└── README.md


## ⚙️ Configuration
The training behavior is managed via configs/sidd.yaml:

Learning Rate: $0.0004$.

Iterations: $800$ (with early stopping).

Masking Threshold: $0.5$.

Inference: Uses $10$ predictions averaged for final output quality.

## 🏋️ Training & Inference
To train the model on the SIDD validation set:

python train_sidd.py

Training Workflow:

Block-wise Training: The script iterates through each block in ValidationNoisyBlocksSrgb.mat.

Wavelet Processing: Input is decomposed into LL, LH, HL, and HH subbands for frequency-domain suppression.

Optimization: The DualBranchDenoiser is optimized using the Adam optimizer and a CosineAnnealingLR scheduler.

Evaluation: Post-training, the model generates denoised images and calculates PSNR and SSIM against the ground truth.

## 📊 Results
The script outputs evaluation metrics for every block to output-sidd/results.csv.
