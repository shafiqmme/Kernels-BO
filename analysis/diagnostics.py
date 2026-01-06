# analysis/diagnostics.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────────────────────
# Numeric diagnostics
# ─────────────────────────────────────────────────────────────

def mds_reconstruction_error(D2: torch.Tensor, X: torch.Tensor):
    """
    Mean absolute reconstruction error between original squared distances
    and distances reconstructed from MDS coordinates.
    """
    with torch.no_grad():
        D2_rec = torch.cdist(X, X, p=2) ** 2
        err = torch.mean(torch.abs(D2_rec - D2))
        rel = err / (torch.mean(torch.abs(D2)) + 1e-12)
    return err.item(), rel.item()


def parity_metrics(D2: torch.Tensor, X: torch.Tensor):
    """
    R², RMSE, MAE between original distances and MDS-reconstructed distances.
    """
    D_orig = torch.sqrt(torch.clamp(D2, min=0.0))
    D_mds = torch.cdist(X, X, p=2)

    mask = torch.triu(torch.ones_like(D_orig), diagonal=1).bool()
    y_true = D_orig[mask].cpu().numpy()
    y_pred = D_mds[mask].cpu().numpy()

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return r2, rmse, mae


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_mds_error_vs_k(
    D2: torch.Tensor,
    k_max: int = 25,
    save_path: str | None = None,
):
    """
    Mean absolute reconstruction error vs embedding dimension k.
    """
    ks = range(1, min(k_max, D2.shape[0]) + 1)
    errors = []

    from embedding.mds import mds_from_squared_distance

    for k in ks:
        X, _ = mds_from_squared_distance(D2, k=k)
        err, _ = mds_reconstruction_error(D2, X)
        errors.append(err)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(ks, errors, marker="o", linewidth=1.5)
    plt.xlabel("Number of MDS Coordinates")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_mds_parity(
    D2: torch.Tensor,
    X: torch.Tensor,
    save_path: str | None = None,
):
    """
    Parity plot: original √JS distance vs MDS-reconstructed distance.
    """
    D_orig = torch.sqrt(torch.clamp(D2, min=0.0))
    D_mds = torch.cdist(X, X, p=2)

    mask = torch.triu(torch.ones_like(D_orig), diagonal=1).bool()
    x = D_orig[mask].cpu().numpy()
    y = D_mds[mask].cpu().numpy()

    plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(x, y, s=20, alpha=0.6, edgecolors="k")
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5)
    plt.xlabel("Original Squared Distances between Kernels")
    plt.ylabel("Squared Distances Reconstructed from MDS Coordinates")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
