import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from src.neural_nets.models import DriftNetwork
from src.algorithms.losses import BridgeMatchingLoss
from src.utils.maths import kinetic_energy
from src.algorithms.sde_utils import euler_maruyama, euler_maruyama_trajectory

# Configuration réduite pour tester de nombreux sigmas rapidement
CONFIG = {
    "n_samples": 2048,
    "data_noise": 0.05,
    "n_sde_steps": 100,
    "n_imf": 2,          # 2 itérations pour s'approcher du bridge
    "n_inner": 1500,     # suffisant sur un petit MLP
    "lr": 2e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "plot_dir": "figures/dsb_phase_study",
}

os.makedirs(CONFIG["plot_dir"], exist_ok=True)


def get_target_data(n, d):
    x, _ = make_moons(n_samples=n, noise=CONFIG["data_noise"])
    return torch.from_numpy((x - 0.5) / 0.6).float().to(d)


def get_source_data(n, d):
    return torch.randn(n, 2, device=d)


def trajectory_variance(traj_numpy):
    """
    Calcule la 'tremulance' stochastique : la variance moyenne des positions X, Y
    autour de la trajectoire lissée moyenne pour une particule.
    traj_numpy : (T, N, 2)
    Retourne un scalaire (variance globale spatiale).
    """
    # Différences d'ordre 2 pour estimer la rugosité (le tremblement brownien)
    # v_t = x_{t+1} - x_t  => acc_t = v_{t+1} - v_t
    # La variance de acc_t est approximativement 2*sigma² dt (roughness)
    diffs = np.diff(traj_numpy, axis=0)
    second_diffs = np.diff(diffs, axis=0)
    
    # Énergie des hautes fréquences = "tremblement" moyen par pas
    tremor_variance = np.mean(np.sum(second_diffs**2, axis=-1))
    return float(tremor_variance)


def train_and_measure(sigma):
    """
    Entraîne un modèle DSBM pour un sigma donné et mesure :
     1. L'énergie cinétique (efficacité OT)
     2. La variance de trajectoire (tremblement SDE)
    """
    print(f"\n==========================================")
    print(f" Phase Study: Formation σ = {sigma}")
    print(f"==========================================")
    
    device = CONFIG["device"]
    model = DriftNetwork(data_dim=2, hidden_dim=256, time_dim=64).to(device)
    loss_fn = BridgeMatchingLoss(model, sigma=sigma)
    
    x0_coupled = get_source_data(CONFIG["n_samples"], device)
    x1_coupled = get_target_data(CONFIG["n_samples"], device)
    
    for imf_iter in range(CONFIG["n_imf"]):
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["n_inner"])
        
        model.train()
        for epoch in range(1, CONFIG["n_inner"] + 1):
            idx = torch.randperm(CONFIG["n_samples"])
            x0_batch = x0_coupled[idx]
            x1_batch = x1_coupled[idx]
            
            optimizer.zero_grad()
            loss = loss_fn(x0_batch, x1_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
        print(f"  > IMF {imf_iter+1} final loss: {loss.item():.4f}")
        
        if imf_iter < CONFIG["n_imf"] - 1:
            x0_new = get_source_data(CONFIG["n_samples"], device)
            x1_new = euler_maruyama(model, x0_new, sigma, CONFIG["n_sde_steps"], device)
            x0_coupled = x0_new.detach()
            x1_coupled = x1_new.detach()

    # ─── Mesures ──────────────────────────────────────────────
    model.eval()
    
    # 1. Énergie Cinétique (∫‖u_θ‖² dt) moyennée sur 1000 particules
    eval_x0 = get_source_data(1000, device)
    ke = kinetic_energy(model, eval_x0, sigma, CONFIG["n_sde_steps"], device)
    
    # 2. Variance du tremblement (Rugosité des trajectoires)
    traj, _ = euler_maruyama_trajectory(model, eval_x0, sigma, CONFIG["n_sde_steps"], device)
    tremor = trajectory_variance(traj.cpu().numpy())
    
    print(f"  ✔️ Terminé ! KE: {ke:.4f} | Tremor: {tremor:.6f}")
    return ke, tremor


if __name__ == "__main__":
    # Plage logarithmique de sigma pour bien voir la limite zéro
    sigmas = [1.0, 0.8, 0.5, 0.3, 0.15, 0.08, 0.03]
    
    ke_results = []
    tremor_results = []
    
    for s in sigmas:
        ke, trem = train_and_measure(s)
        ke_results.append(ke)
        tremor_results.append(trem)
        
    print("\n✅ Expériences terminées. Génération du graphique...")
    
    # ─── Graphique Double Axe ──────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    
    # La limite Zero-Noise implique que le modèle "gèle" en ODE (le tremblement -> 0)
    # L'énergie cinétique se stabilise vers la formulation de Monge-Kantorovich (OT)
    
    ax1.set_xlabel("Niveau de bruit intrinsèque $\sigma$ (Échelle Log)", fontsize=12, fontweight='bold')
    ax1.set_xlim(min(sigmas)*0.8, max(sigmas)*1.2)
    ax1.set_xscale("log")
    
    # Axe Gauche : Tremblement (Variance)
    color1 = 'tab:red'
    ax1.set_ylabel("Variance des trajectoires (Rugosité SDE)", color=color1, fontsize=12, fontweight='bold')
    ax1.plot(sigmas, tremor_results, marker='s', markersize=8, color=color1, linewidth=2.5, label="Tremblement SDE")
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Axe Droit : Énergie Cinétique
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel("Énergie Cinétique $E[|u_θ|^2 dt]$", color=color2, fontsize=12, fontweight='bold')
    ax2.plot(sigmas, ke_results, marker='o', markersize=8, color=color2, linewidth=2.5, linestyle='--', label="Énergie Cinétique")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Titre et grid
    plt.title("L'Étude de Phase : Bifurcation Continue vers la Zero-Noise Limit\n(Flow Matching ODE vs Schrödinger Bridge SDE)", fontsize=14, fontweight='bold', pad=15)
    
    # Flèches d'asymptote
    ax1.annotate("Limite Déterministe\n(Conditional Flow Matching)", 
                 xy=(0.04, min(tremor_results)), xytext=(0.05, max(tremor_results)*0.4),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
                 
    ax2.annotate("Limite Stochastique \n(Haute Entropie SB)", 
                 xy=(0.8, max(ke_results)), xytext=(0.4, max(ke_results)*0.8),
                 arrowprops=dict(facecolor='gray', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    ax1.grid(True, which="both", ls="-", alpha=0.3)
    fig.tight_layout()
    
    save_path = os.path.join(CONFIG["plot_dir"], "phase_study_zero_noise_limit.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Graphique d'étude de phase sauvegardé dans : {save_path}")
