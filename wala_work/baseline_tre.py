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

def compute_tre(fixed_lm, moving_lm, phi_field, orig_shape):
    scale = np.array([175/orig_shape[2], 175/orig_shape[1], 175/orig_shape[0]])
    fixed_scaled = fixed_lm * scale
    moving_scaled = moving_lm * scale
    phi_np = phi_field.detach().numpy()[0]
    warped_lm = []
    for lm in moving_scaled:
        x, y, z = int(lm[0]), int(lm[1]), int(lm[2])
        x = np.clip(x, 0, 174)
        y = np.clip(y, 0, 174)
        z = np.clip(z, 0, 174)
        new_x = phi_np[2, z, y, x] * 174
        new_y = phi_np[1, z, y, x] * 174
        new_z = phi_np[0, z, y, x] * 174
        warped_lm.append([new_x, new_y, new_z])
    warped_lm = np.array(warped_lm)
    errors = np.linalg.norm(fixed_scaled - warped_lm, axis=1)
    return np.mean(errors)

model = get_unigradicon()
model.eval()

for pair in ["0001", "0002", "0003"]:
    orig_shape = itk.array_from_image(itk.imread(f"{DATA}/imagesTr/LungCT_{pair}_0000.nii.gz", itk.F)).shape

    fixed_lm = load_landmarks(f"{DATA}/landmarksTr/LungCT_{pair}_0000.csv")
    moving_lm = load_landmarks(f"{DATA}/landmarksTr/LungCT_{pair}_0001.csv")

    dists_before = [np.sqrt(np.sum((fixed_lm[i] - moving_lm[i])**2)) for i in range(len(fixed_lm))]
    tre_before = np.mean(dists_before)

    fixed_t = load_image_torch(f"{DATA}/imagesTr/LungCT_{pair}_0000.nii.gz")
    moving_t = load_image_torch(f"{DATA}/imagesTr/LungCT_{pair}_0001.nii.gz")

    with torch.no_grad():
        phi = model(moving_t, fixed_t, moving_t, fixed_t)

    phi_field = model.phi_AB_vectorfield
    tre_after = compute_tre(fixed_lm, moving_lm, phi_field, orig_shape)
    warped = model.warped_image_A

    mid = 87
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(fixed_t[0,0,mid].numpy(), cmap='gray')
    axes[0].set_title('Fixed')
    axes[1].imshow(moving_t[0,0,mid].numpy(), cmap='gray')
    axes[1].set_title('Moving')
    axes[2].imshow(warped[0,0,mid].detach().numpy(), cmap='gray')
    axes[2].set_title('Warped Moving')
    axes[3].imshow((fixed_t[0,0,mid] - warped[0,0,mid]).detach().numpy(), cmap='bwr')
    axes[3].set_title('Difference (after reg)')

    plt.suptitle(f'Pair {pair} — Baseline')
    plt.savefig(f"results/baseline_pair{pair}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Pair {pair}: TRE before={tre_before:.2f}mm | TRE after={tre_after:.2f}mm | sim={phi.similarity_loss.item():.4f} | flips={phi.flips.item():.0f}")