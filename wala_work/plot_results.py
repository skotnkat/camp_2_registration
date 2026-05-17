import matplotlib.pyplot as plt
import numpy as np

pairs = ['0001', '0002', '0003']

baseline_sim = [0.9114, 0.6528, 0.8049]
corrupted_sim = [1.6336, 1.1559, 1.2551]
fixed_sim = [0.9031, 0.5833, 0.7637]

baseline_flips = [5267, 1325, 1686]
corrupted_flips = [30284, 39452, 9503]
fixed_flips = [7097, 1273, 2123]

x = np.arange(len(pairs))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Similarity Loss
axes[0].bar(x - width, baseline_sim, width, label='Baseline', color='steelblue')
axes[0].bar(x, corrupted_sim, width, label='Corrupted (FOV truncation)', color='tomato')
axes[0].bar(x + width, fixed_sim, width, label='Fixed (Reference Inpainting)', color='seagreen')
axes[0].set_xlabel('Pair', fontsize=12)
axes[0].set_ylabel('Similarity Loss (LNCC)', fontsize=12)
axes[0].set_title('Similarity Loss by Condition', fontsize=13)
axes[0].set_xticks(x)
axes[0].set_xticklabels(pairs)
axes[0].legend(fontsize=9)

# Folded Voxels
axes[1].bar(x - width, baseline_flips, width, label='Baseline', color='steelblue')
axes[1].bar(x, corrupted_flips, width, label='Corrupted (FOV truncation)', color='tomato')
axes[1].bar(x + width, fixed_flips, width, label='Fixed (Reference Inpainting)', color='seagreen')
axes[1].set_xlabel('Pair', fontsize=12)
axes[1].set_ylabel('Folded Voxels (Flips)', fontsize=12)
axes[1].set_title('Folded Voxels by Condition', fontsize=13)
axes[1].set_xticks(x)
axes[1].set_xticklabels(pairs)
axes[1].legend(fontsize=9)

plt.suptitle('LungCT Registration: Baseline vs Corrupted vs Fixed\n(FOV Truncation + Reference-based Inpainting Fix)', fontsize=13)
plt.tight_layout()
plt.savefig('results/final_comparison_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/final_comparison_chart.png")

# Print summary table
print("\n=== RESULTS SUMMARY ===")
print(f"{'Condition':<12} {'Avg Sim':>10} {'Avg Flips':>12}")
print("-" * 36)
print(f"{'Baseline':<12} {np.mean(baseline_sim):>10.4f} {np.mean(baseline_flips):>12.0f}")
print(f"{'Corrupted':<12} {np.mean(corrupted_sim):>10.4f} {np.mean(corrupted_flips):>12.0f}")
print(f"{'Fixed':<12} {np.mean(fixed_sim):>10.4f} {np.mean(fixed_flips):>12.0f}")