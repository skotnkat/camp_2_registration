import torch
import itk
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from unigradicon import get_unigradicon

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

def corrupt_fov_middle(tensor, start=60, end=130):
    corrupted = tensor.clone()
    corrupted[:, :, start:end, :, :] = 0
    return corrupted

def fix_interpolate(tensor, start=60, end=130):
    """Fill truncated region by linear interpolation between boundaries"""
    fixed = tensor.clone()
    top_slice = tensor[:, :, start-1, :, :]
    bottom_slice = tensor[:, :, end, :, :]
    n_slices = end - start
    for i in range(n_slices):
        alpha = i / n_slices
        fixed[:, :, start+i, :, :] = (1-alpha) * top_slice + alpha * bottom_slice
    return fixed

model = get_unigradicon()
model.eval()

for pair in ["0001", "0002", "0003"]:
    fixed_lm = load_landmarks(f"{DATA}/landmarksTr/LungCT_{pair}_0000.csv")
    moving_lm = load_landmarks(f"{DATA}/landmarksTr/LungCT_{pair}_0001.csv")

    dists_before = [np.sqrt(np.sum((fixed_lm[i] - moving_lm[i])**2)) for i in range(len(fixed_lm))]
    tre_before = np.mean(dists_before)

    fixed_t = load_image_torch(f"{DATA}/imagesTr/LungCT_{pair}_0000.nii.gz")
    moving_t = load_image_torch(f"{DATA}/imagesTr/LungCT_{pair}_0001.nii.gz")

    fixed_c = corrupt_fov_middle(fixed_t)
    moving_c = corrupt_fov_middle(moving_t)

    fixed_fixed = fix_interpolate(fixed_c)
    moving_fixed = fix_interpolate(moving_c)

    with torch.no_grad():
        phi = model(moving_fixed, fixed_fixed, moving_fixed, fixed_fixed)

    warped = model.warped_image_A

    mid = 87
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(fixed_fixed[0,0,mid].numpy(), cmap='gray')
    axes[0].set_title('Fixed (repaired)')
    axes[1].imshow(moving_fixed[0,0,mid].numpy(), cmap='gray')
    axes[1].set_title('Moving (repaired)')
    axes[2].imshow(warped[0,0,mid].detach().numpy(), cmap='gray')
    axes[2].set_title('Warped Moving')
    axes[3].imshow((fixed_fixed[0,0,mid] - warped[0,0,mid]).detach().numpy(), cmap='bwr')
    axes[3].set_title('Difference (after reg)')

    plt.suptitle(f'Pair {pair} — Fixed (Linear Interpolation)')
    plt.savefig(f"results/fov_mid_fixed_pair{pair}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Pair {pair}: TRE before={tre_before:.2f}mm | sim={phi.similarity_loss.item():.4f} | flips={phi.flips.item():.0f}")