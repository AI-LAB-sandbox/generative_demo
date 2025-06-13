import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DomainGen_Graph")

datasets = ['ONP', 'Moons', 'MNIST', 'Elec2', 'Portrait', 'Yearbook']
parser.add_argument("--dataset", default="Moons", type=str,
                    help="one of: {}".format(", ".join(sorted(datasets))))
parser.add_argument("--portrait_pkl", default="portraits_original.pkl", type=str,
                    help="Path to the portrait dataset .pkl file.")
parser.add_argument("--yearbook_pkl", default="yearbook.pkl", type=str,
                    help="Path to the yearbook dataset .pkl file.")

# Hyper-parameters
parser.add_argument("--hidden_dim", default=64, type=float,
                    help="the hidden dimension of predictor.")

parser.add_argument("--num_workers", default=0, type=int,
                    help="the number of threads for loading data.")

parser.add_argument("--epoches", default=100, type=int,
                    help="the number of epoches for each task.")

parser.add_argument("--st_epoches", default=50, type=int,
                    help="the number of epoches for each task.")

parser.add_argument("--batch_size", default=16, type=int,
                    help="the number of epoches for each task.")

parser.add_argument("--learning_rate", default=1e-4*5, type=float,
                    help="the unified learning rate for each single task.")

parser.add_argument("--is_test", default=True, type=bool,
                    help="if this is a testing period.")

parser.add_argument("--gpu", default=0, type=int,
                    help="GPU device id to use (default: 0)")

args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
dataset = "Shuttle"
generated_dir = "generated"
model_dir = "./checkpoint"
step = 5   # 10번마다 한 개 태스크씩 시각화

def plot_torch_decision_boundary(model, X, y, device='cpu', ax=None):
    model.eval()
    X_plot = X[:, :2]
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if X.shape[1] > 2:
        grid_full = np.tile(np.median(X, axis=0), (grid.shape[0], 1))
        grid_full[:, :2] = grid
    else:
        grid_full = grid

    with torch.no_grad():
        inputs = torch.tensor(grid_full, dtype=torch.float32).to(model.fc1.weight.device)
        probs, _ = model(inputs)
        if probs.shape[1] == 1:
            Z = (probs.cpu().numpy() > 0.5).astype(int).reshape(xx.shape)
        else:
            Z = torch.argmax(probs, dim=1).cpu().numpy().reshape(xx.shape)

    if ax is None:
        ax = plt.gca()

    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
if not os.path.isdir(generated_dir):
    raise RuntimeError(f"Directory '{generated_dir}' not found.")

# -----------------------------------------------------------------------------
# Find all task IDs for Moons
# -----------------------------------------------------------------------------
files = os.listdir(generated_dir)
task_ids = sorted({
    int(f.split(f"{dataset}_x_task_")[1].split(".npy")[0])
    for f in files
    if f.startswith(f"{dataset}_x_task_")
})
if not task_ids:
    raise RuntimeError(f"No files matching '{dataset}_x_task_*.npy' in '{generated_dir}'.")

# -----------------------------------------------------------------------------
# Select every `step`-th task to plot
# -----------------------------------------------------------------------------
selected_ids = task_ids[::step]
nrows = len(selected_ids)

# -----------------------------------------------------------------------------
# Load data for each selected task
# -----------------------------------------------------------------------------
x_all = [np.load(f"{generated_dir}/{dataset}_x_task_{tid}.npy") for tid in selected_ids]
xhat_all = [np.load(f"{generated_dir}/{dataset}_xhat_task_{tid}.npy") for tid in selected_ids]
y_all = [np.load(f"{generated_dir}/{dataset}_y_task_{tid}.npy") for tid in selected_ids]
yhat_all = [np.load(f"{generated_dir}/{dataset}_yhat_task_{tid}.npy") for tid in selected_ids]

x_eval = np.load(f"{generated_dir}/{dataset}_x_eval.npy")
xhat_eval = np.load(f"{generated_dir}/{dataset}_xhat_eval.npy")
y_eval = np.load(f"{generated_dir}/{dataset}_y_eval.npy")
yhat_eval = np.load(f"{generated_dir}/{dataset}_yhat_eval.npy")

# -----------------------------------------------------------------------------
# Prepare figure: one row per selected task, two columns (orig / recon)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=nrows + 1, ncols=2,
    figsize=(2*3, (nrows + 1)*3),
    squeeze=False
)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 12))
gs = gridspec.GridSpec(4, 4, figure=fig)  # 총 4행 4열 그리드 (왼쪽 위 4x2, 오른쪽 아래 3x2)

# -------------------------------
# 위쪽 4x2 (8개 subplot: 0~3 task)
# -------------------------------
for row_idx, tid in enumerate(selected_ids[:4]):
    ax_orig = fig.add_subplot(gs[row_idx, 0])
    ax_recon = fig.add_subplot(gs[row_idx, 1])

    ax_orig.scatter(
        x_all[row_idx][:, 0], x_all[row_idx][:, 1],
        c=y_all[row_idx], cmap='coolwarm',
        edgecolor='k', s=20
    )
    ax_orig.set_title(f"[Original] Domain {tid}", fontsize=8)
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])

    ax_recon.scatter(
        xhat_all[row_idx][:, 0], xhat_all[row_idx][:, 1],
        c=yhat_all[row_idx], cmap='coolwarm',
        edgecolor='k', s=20
    )
    ax_recon.set_title(f"[Reconstruction] Domain {tid}", fontsize=8)
    ax_recon.set_xticks([])
    ax_recon.set_yticks([])

# -------------------------------
# 오른쪽 아래쪽 3x2 (tasks 4~6 or rest)
# -------------------------------
for i, row_idx in enumerate(range(4, len(selected_ids))):
    tid = selected_ids[row_idx]

    ax_orig = fig.add_subplot(gs[i, 2])
    ax_recon = fig.add_subplot(gs[i, 3])

    ax_orig.scatter(
        x_all[row_idx][:, 0], x_all[row_idx][:, 1],
        c=y_all[row_idx], cmap='coolwarm',
        edgecolor='k', s=20
    )
    ax_orig.set_title(f"[Original] Domain {tid}", fontsize=8)
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])

    ax_recon.scatter(
        xhat_all[row_idx][:, 0], xhat_all[row_idx][:, 1],
        c=yhat_all[row_idx], cmap='coolwarm',
        edgecolor='k', s=20
    )
    ax_recon.set_title(f"[Reconstruction] Domain {tid}", fontsize=8)
    ax_recon.set_xticks([])
    ax_recon.set_yticks([])

# 마지막 180도 Target domain은 오른쪽 아래 맨 마지막에 추가
ax_orig = fig.add_subplot(gs[2, 2])
ax_recon = fig.add_subplot(gs[2, 3])

ax_orig.scatter(
    x_eval[:, 0], x_eval[:, 1],
    c=y_eval, cmap='coolwarm',
    edgecolor='k', s=20
)
ax_orig.set_title("[Original] Target Domain", fontsize=8)
ax_orig.set_xticks([])
ax_orig.set_yticks([])

ax_recon.scatter(
    xhat_eval[:, 0], xhat_eval[:, 1],
    c=yhat_eval, cmap='coolwarm',
    edgecolor='k', s=20
)
ax_recon.set_title("[Reconstruction] Target Domain", fontsize=8)
ax_recon.set_xticks([])
ax_recon.set_yticks([])

plt.tight_layout()
save_path = f"./visual/{dataset}_reconstruction_label.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Saved reconstruction plot for '{dataset}' to {save_path}")

# -----------------------------------------------------------------------------
# Save the figure
# -----------------------------------------------------------------------------
os.makedirs('./visual', exist_ok=True)
save_path = f"./visual/{dataset}_reconstruction_label.png"
plt.savefig(save_path, dpi=300)
print(f"✅ Saved reconstruction plot for '{dataset}' to {save_path}")

# -----------------------------------------------------------------------------
# Decision boundary visualization grid
# -----------------------------------------------------------------------------
from model import Predictor
angles_to_plot = [1, 2, 3, 4, 5, 6]  # Example angles to plot
fig, axes = plt.subplots(
    nrows=1, ncols=len(angles_to_plot),
    figsize=(len(angles_to_plot)*4, 4),
    squeeze=False
)

for col_idx, tid in enumerate(angles_to_plot):
    if tid == 6:
        model_path = os.path.join(model_dir, f"{dataset}_last_classifier.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ No model found for task {tid*5}, skipping.")
            continue
        X, Y = xhat_eval, yhat_eval

    else:
        model_path = os.path.join(model_dir, f"{dataset}_classifier_task_{tid*5}.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ No model found for task {tid*5}, skipping.")
            continue
        X, Y = xhat_all[tid], yhat_all[tid]

    model = Predictor(data_size=2, args=argparse.Namespace(hidden_dim=64)).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    plot_torch_decision_boundary(model, X, Y, device=device, ax=axes[0, col_idx])

    if tid == 6:
        axes[0, col_idx].set_title(f"Target Domain w/ decision boundary")
    else:
        axes[0, col_idx].set_title(f" Domain {tid*5} w/ decision boundary")

    axes[0, col_idx].set_xticks([])
    axes[0, col_idx].set_yticks([])

plt.tight_layout()
os.makedirs('./visual', exist_ok=True)
decision_path = f"./visual/{dataset}_decision_boundaries.png"
plt.savefig(decision_path, dpi=300)
print(f"✅ Saved selected decision boundaries for '{dataset}' to {decision_path}")
