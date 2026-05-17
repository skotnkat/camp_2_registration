import torch
import itk
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from unigradicon import get_unigradicon
import torch.nn as nn

torch.set_num_threads(2)

DATA = "D:/camp 2 project/dataset/LungCT/LungCT"

def load_landmarks(path):
    return np.loadtxt(path, delimiter=',')

def load_image_torch(path):
    img = itk.imread(path, itk.F)
    arr = itk.array_from_image(img)
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(175, 175, 175), mode='trilinear', align_corners=False)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor

def corrupt_noise(tensor, std=0.2):
    noise = torch.randn_like(tensor) * std
    corrupted = tensor + noise
    return corrupted.clamp(0, 1)

def fix_gaussian_smooth(tensor, kernel_size=5, sigma=1.5):
    """Apply Gaussian smoothing to reduce noise"""
    # create 3D Gaussian kernel
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing='ij')
    kernel = torch.exp(-(grid_x**2 + grid_y**2 + grid_z**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    padding = kernel_size // 2
    smoothed = F.conv3d(tensor, kernel, padding=padding)
    return smoothed.clamp(0, 1)

model = get_unigradicon()
model.eval()

for pair in ["0001", "0002", "0003"]:
    fixed_lm = load_landmarks(f"{DATA}/landmarksTr/LungCT_{pair}_0000.csv")
    moving_lm = load_landmarks(f"{DATA}/landmarksTr/LungCT_{pair}_0001.csv")

    dists_before = [np.sqrt(np.sum((fixed_lm[i] - moving_lm[i])**2)) for i in range(len(fixed_lm))]
    tre_before = np.mean(dists_before)

    fixed_t = load_image_torch(f"{DATA}/imagesTr/LungCT_{pair}_0000.nii.gz")
    moving_t = load_image_torch(f"{DATA}/imagesTr/LungCT_{pair}_0001.nii.gz")

    moving_c = corrupt_noise(moving_t, std=0.2)
    moving_fixed = fix_gaussian_smooth(moving_c)

    with torch.no_grad():
        phi = model(moving_fixed, fixed_t, moving_fixed, fixed_t)

    warped = model.warped_image_A

    mid = 87
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(fixed_t[0,0,mid].numpy(), cmap='gray')
    axes[0].set_title('Fixed')
    axes[1].imshow(moving_fixed[0,0,mid].numpy(), cmap='gray')
    axes[1].set_title('Moving (denoised)')
    axes[2].imshow(warped[0,0,mid].detach().numpy(), cmap='gray')
    axes[2].set_title('Warped Moving')
    axes[3].imshow((fixed_t[0,0,mid] - warped[0,0,mid]).detach().numpy(), cmap='bwr')
    axes[3].set_title('Difference (after reg)')

    plt.suptitle(f'Pair {pair} — Fixed (Gaussian Smoothing)')
    plt.savefig(f"results/noise_fixed_pair{pair}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Pair {pair}: TRE before={tre_before:.2f}mm | sim={phi.similarity_loss.item():.4f} | flips={phi.flips.item():.0f}")