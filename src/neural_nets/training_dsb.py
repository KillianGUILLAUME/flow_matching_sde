"""
Entraînement du Diffusion Schrödinger Bridge Matching (DSBM).

Implémente l'Algorithme IMF (Iterative Markovian Fitting) du papier :
  Shi, De Bortoli, Campbell, Doucet — NeurIPS 2023.

Structure :
  Boucle externe  : n_imf itérations de Markovian Fitting
  Boucle interne  : Bridge Matching — apprend le drift u_θ via régression
  Mise à jour     : le couplage (x₀, x₁) est mis à jour par intégration SDE

Usage :
    python -m src.neural_nets.training_dsb
"""

import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv

from sklearn.datasets import make_moons

from src.neural_nets.models import DriftNetwork
from src.algorithms.losses import BridgeMatchingLoss
from src.algorithms.sde_utils import euler_maruyama
from src.visualization.plots import (
    visualize_dsb_samples,
    visualize_drift_quiver,
    visualize_trajectories_spaghetti,
)
from src.utils.maths import wasserstein_2, mmd_gaussian, kinetic_energy

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    # Données
    "n_samples":     4096,
    "data_noise":    0.05,
    # SDE
    "sigma":         0.5,    # bruit du processus de référence (Brownian Bridge)
    "n_sde_steps":   100,    # pas Euler-Maruyama pour la mise à jour du couplage
    # IMF
    "n_imf":         5,      # nombre d'itérations Markovian Fitting
    "n_inner":       2000,   # epochs d'entraînement inner par itération IMF
    # Optimiseur
    "lr":            2e-3,
    "min_lr":        1e-6,
    "weight_decay":  1e-5,
    # Infrastructure
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir":      "checkpoints",
    "plot_dir":      "figures/dsb",
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
os.makedirs(CONFIG["plot_dir"], exist_ok=True)


# ─── Données ──────────────────────────────────────────────────────────────────

def get_target_data(n_samples: int, device: str) -> torch.Tensor:
    """Tire n_samples points de la distribution two-moons (normalisée)."""
    x, _ = make_moons(n_samples=n_samples, noise=CONFIG["data_noise"])
    x = (x - 0.5) / 0.6
    return torch.from_numpy(x).float().to(device)


def get_source_data(n_samples: int, device: str) -> torch.Tensor:
    """Tire n_samples points du bruit blanc gaussien."""
    return torch.randn(n_samples, 2, device=device)


# ─── Boucle IMF ───────────────────────────────────────────────────────────────

def train_dsb():
    """
    Boucle principale IMF :

      k = 0 : couplage indépendant  π₀ = p₀ ⊗ p₁
      Pour k = 1…N_imf :
        1.  Bridge Matching inner : minimise BridgeMatchingLoss avec couplage π_{k-1}
        2.  Mise à jour du couplage : générer x̂₁ = EM(u_θ_k, x₀) et former π_k = (x₀, x̂₁)
    """
    device = CONFIG["device"]
    sigma  = CONFIG["sigma"]
    print(f"🌀 DSBM — Iterative Markovian Fitting sur {device}")
    print(f"   σ={sigma} | {CONFIG['n_imf']} itérations IMF | {CONFIG['n_inner']} inner epochs each")

    # ── Initialisation du modèle ──────────────────────────────────────────────
    model = DriftNetwork(data_dim=2, hidden_dim=256, time_dim=64).to(device)
    loss_fn = BridgeMatchingLoss(model, sigma=sigma)

    # ── Couplage initial : indépendant (π₀ = p₀ ⊗ p₁) ───────────────────────
    x0_coupled = get_source_data(CONFIG["n_samples"], device)
    x1_coupled = get_target_data(CONFIG["n_samples"], device)

    all_losses = []   # historique global des losses
    metrics_log = []  # W2 / MMD / KE par itération IMF

    # ── Échantillons de référence fixes pour l'évaluation ─────────────────────
    # On fixe la seed pour avoir des comparaisons cohérentes entre itérations.
    eval_n = 1000
    eval_x0 = torch.randn(eval_n, 2, device=device)
    rng = np.random.default_rng(42)
    x_ref_np, _ = make_moons(n_samples=eval_n, noise=CONFIG["data_noise"], random_state=42)
    x_ref_np = (x_ref_np - 0.5) / 0.6  # même normalisation que l'entraînement

    for imf_iter in range(CONFIG["n_imf"]):
        print(f"\n{'='*60}")
        print(f"  IMF Iteration {imf_iter + 1}/{CONFIG['n_imf']}")
        print(f"{'='*60}")

        # ── Réinitialise l'optimiseur à chaque itération IMF ─────────────────
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG["n_inner"],
            eta_min=CONFIG["min_lr"]
        )

        iter_losses = []

        # ── Inner loop : Bridge Matching ──────────────────────────────────────
        model.train()
        for epoch in range(1, CONFIG["n_inner"] + 1):
            # On re-tire des paires fraîches du couplage courant à chaque epoch
            # Le couplage est stocké comme un grand batch ; on sous-échantillonne
            idx = torch.randperm(x0_coupled.shape[0])[:CONFIG["n_samples"]]
            x0_batch = x0_coupled[idx]
            x1_batch = x1_coupled[idx]

            optimizer.zero_grad()
            loss = loss_fn(x0_batch, x1_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            iter_losses.append(loss.item())

            if epoch % 500 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:04d}/{CONFIG['n_inner']} | Loss: {loss.item():.6f} | LR: {lr_now:.1e}")

        all_losses.extend(iter_losses)

        # ── Visualisation après chaque itération IMF ──────────────────────────
        model.eval()
        x0_vis = get_source_data(1000, device)
        x1_ref  = get_target_data(1000, device)
        visualize_dsb_samples(
            drift_model=model,
            epoch=imf_iter + 1,
            sigma=sigma,
            n_sde_steps=CONFIG["n_sde_steps"],
            device=device,
            data_sample=x1_ref.cpu().numpy(),
            plot_dir=CONFIG["plot_dir"],
            x0_samples=x0_vis,
        )

        # Quiver : champ de dérive à t=0.2 / 0.5 / 0.8
        visualize_drift_quiver(
            drift_model=model,
            imf_iter=imf_iter + 1,
            device=device,
            data_sample=x1_ref.cpu().numpy(),
            plot_dir=CONFIG["plot_dir"],
        )

        # Spaghetti : 50 trajectoires individuelles
        visualize_trajectories_spaghetti(
            drift_model=model,
            imf_iter=imf_iter + 1,
            sigma=sigma,
            n_sde_steps=CONFIG["n_sde_steps"],
            device=device,
            data_sample=x1_ref.cpu().numpy(),
            plot_dir=CONFIG["plot_dir"],
        )

        # ── Mise à jour du couplage π_{k+1} ──────────────────────────────────
        #    On génère de nouveaux x₀ (bruit), on intègre le drift appris,
        #    et le nouveau couplage associe (x₀, x̂₁).
        print(f"  Mise à jour du couplage π_{imf_iter + 1}…")
        n_couple = CONFIG["n_samples"]
        x0_new = get_source_data(n_couple, device)
        x1_new = euler_maruyama(model, x0_new, sigma, CONFIG["n_sde_steps"], device)

        x0_coupled = x0_new.detach()
        x1_coupled = x1_new.detach()
        print(f"  Nouveau couplage : {n_couple} paires (x₀, x̂₁)")

        # ── Évaluation des métriques ──────────────────────────────────────────
        print(f"  Calcul des métriques…")
        model.eval()

        # Générer des échantillons fixes pour comparaison reproductible
        x_gen_eval = euler_maruyama(
            model, eval_x0, sigma, CONFIG["n_sde_steps"], device
        ).cpu().numpy()

        w2  = wasserstein_2(x_gen_eval, x_ref_np, max_samples=500)
        mmd = mmd_gaussian(x_gen_eval, x_ref_np, max_samples=eval_n)
        ke  = kinetic_energy(
            model, eval_x0, sigma, CONFIG["n_sde_steps"], device
        )

        metrics_log.append({
            "imf_iter": imf_iter + 1,
            "w2": w2,
            "mmd": mmd,
            "kinetic_energy": ke,
            "loss_final": iter_losses[-1],
        })

        print(f"  ┌─ Métriques IMF iter {imf_iter + 1} ─────────────────────┐")
        print(f"  │  W₂ (Wasserstein-2)      : {w2:.4f}")
        print(f"  │  MMD (Gaussian kernel)   : {mmd:.4f}")
        print(f"  │  Énergie cinétique (KE)  : {ke:.4f}")
        print(f"  └────────────────────────────────────────────────────┘")

        # ── Sauvegarde du checkpoint de l'itération ───────────────────────────
        ckpt_path = f"{CONFIG['save_dir']}/dsb_drift_imf{imf_iter + 1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  ✅ Checkpoint sauvegardé : {ckpt_path}")

    # ── Sauvegarde finale ─────────────────────────────────────────────────────
    final_path = f"{CONFIG['save_dir']}/dsb_drift_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Modèle final sauvegardé : {final_path}")

    # ── Courbe de loss globale ────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(all_losses, linewidth=0.8, color="steelblue")
    plt.yscale("log")
    plt.xlabel("Iteration (inner steps cumulés)")
    plt.ylabel("BridgeMatchingLoss")
    plt.title("DSBM — Loss d'entraînement (toutes itérations IMF)")
    for k in range(1, CONFIG["n_imf"]):
        plt.axvline(k * CONFIG["n_inner"], color="orange", linestyle="--", linewidth=0.8, label=f"IMF iter {k}" if k == 1 else "")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CONFIG['plot_dir']}/loss_global.png", dpi=150)
    plt.close()
    print(f"📊 Courbe de loss sauvegardée : {CONFIG['plot_dir']}/loss_global.png")

    # ── Sauvegarde des métriques en CSV ───────────────────────────────────────
    csv_path = f"{CONFIG['plot_dir']}/metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["imf_iter", "w2", "mmd", "kinetic_energy", "loss_final"])
        writer.writeheader()
        writer.writerows(metrics_log)
    print(f"📋 Métriques CSV sauvegardées : {csv_path}")

    # ── Figure métriques par itération IMF ───────────────────────────────────
    iters = [m["imf_iter"] for m in metrics_log]
    w2s   = [m["w2"]           for m in metrics_log]
    mmds  = [m["mmd"]          for m in metrics_log]
    kes   = [m["kinetic_energy"] for m in metrics_log]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("DSBM — Métriques par itération IMF", fontsize=13)

    axes[0].plot(iters, w2s, marker="o", color="steelblue")
    axes[0].set_title("W₂ (Wasserstein-2)")
    axes[0].set_xlabel("IMF Iteration")
    axes[0].set_ylabel("W₂ ↓")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, mmds, marker="s", color="darkorange")
    axes[1].set_title("MMD (Gaussian kernel)")
    axes[1].set_xlabel("IMF Iteration")
    axes[1].set_ylabel("MMD ↓")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(iters, kes, marker="^", color="forestgreen")
    axes[2].set_title("Énergie Cinétique (∫‖u_θ‖² dt)")
    axes[2].set_xlabel("IMF Iteration")
    axes[2].set_ylabel("KE ↓")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    metrics_fig_path = f"{CONFIG['plot_dir']}/metrics_per_imf.png"
    plt.savefig(metrics_fig_path, dpi=150)
    plt.close()
    print(f"📊 Figure métriques sauvegardée : {metrics_fig_path}")

    # ── Tableau récapitulatif final ───────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  RÉCAPITULATIF DES MÉTRIQUES PAR ITÉRATION IMF")
    print(f"{'='*62}")
    print(f"  {'Iter':>4}  {'W₂':>10}  {'MMD':>10}  {'KE':>12}  {'Loss':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")
    for m in metrics_log:
        print(f"  {m['imf_iter']:>4}  {m['w2']:>10.4f}  {m['mmd']:>10.4f}  {m['kinetic_energy']:>12.4f}  {m['loss_final']:>10.6f}")
    print(f"{'='*62}")


if __name__ == "__main__":
    train_dsb()
