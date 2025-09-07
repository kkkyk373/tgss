import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

base_dir = "outputs"
area_ids_path = os.path.join(base_dir, "fgw_area_ids.npy")
area_ids = np.load(area_ids_path)
dist_mats = {}
alphas = ['00', '50', '100']
for alpha in alphas:
    dist_mat_path = os.path.join(base_dir, f"fgw_dist_{alpha}.dat")
    dist_mat = np.memmap(dist_mat_path, dtype=np.float32, mode="r", shape=(len(area_ids), len(area_ids)))
    dist_mats[alpha] = dist_mat


fig, axes = plt.subplots(2, 4, figsize=(16, 11), gridspec_kw={'width_ratios': [1, 1, 1, 0.08]})
fig.suptitle('Comparison of FGW Distance Matrices with Different Normalizations', fontsize=20)


all_values = []
for alpha in alphas:
    mat = dist_mats[alpha][:100, :100]
    all_values.append(mat[mat > 0])
concatenated_values = np.concatenate(all_values)
vmin = concatenated_values.min()
vmax = concatenated_values.max()


for i, alpha in enumerate(alphas):
    ax = axes[0, i]
    submatrix = dist_mats[alpha][:100, :100]
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    im1 = ax.imshow(submatrix, cmap='viridis', norm=norm)
    ax.set_title(f"alpha={int(alpha)}", fontsize=15)
    if i == 0:
        ax.set_ylabel("Global Log Norm\n\nIndex", fontsize=15)

fig.colorbar(im1, cax=axes[0, 3], label="Common Normalized Distance")


dummy_im = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap='viridis')
for i, alpha in enumerate(alphas):
    ax = axes[1, i]
    submatrix = dist_mats[alpha][:100, :100]
    local_min = submatrix.min()
    local_max = submatrix.max()
    normalized_matrix = (submatrix - local_min) / (local_max - local_min + 1e-9)
    im2 = ax.imshow(normalized_matrix, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f"alpha={int(alpha)}", fontsize=15)
    ax.set_xlabel("Index", fontsize=15)
    if i == 0:
        ax.set_ylabel("Local Linear Norm\n\nIndex", fontsize=15)

fig.colorbar(dummy_im, cax=axes[1, 3], label="Individual Normalized Distance")


fig.subplots_adjust(
    top=0.95,      # Spacing to the overall title
    left=0.05,    # Left margin
    right=0.95,   # Right margin
    bottom=0.05,  # Bottom margin
    wspace=0.1,   # Horizontal spacing between plots
    hspace=0.1,    # Vertical spacing between plots
)
plt.savefig("figs/fgw_dist_comparison.png", dpi=300)
plt.savefig("figs/fgw_dist_comparison.svg", dpi=300)  # Also save as SVG format
plt.show()