import os
import torch
import torch.optim as optim
import numpy as np

from sklearn.datasets import make_moons
from src.neural_nets.models import DriftNetwork
from src.algorithms.losses import BridgeMatchingLoss
from src.algorithms.sde_utils import euler_maruyama
from src.visualization.plots import visualize_trajectories_spaghetti

CONFIG = {
    "n_samples": 4096,
    "data_noise": 0.05,
    "n_sde_steps": 100,
    "n_imf": 2,      # 2 itérations suffisent pour voir les traj droites
    "n_inner": 1500, # Moins d'époques pour aller plus vite en CPU
    "lr": 2e-3,
    "min_lr": 1e-6,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "plot_dir": "figures/dsb_sigma_limit",
}

os.makedirs(CONFIG["plot_dir"], exist_ok=True)

def get_target_data(n, d):
    x, _ = make_moons(n_samples=n, noise=CONFIG["data_noise"])
    return torch.from_numpy((x - 0.5) / 0.6).float().to(d)

def get_source_data(n, d):
    return torch.randn(n, 2, device=d)

def run_experiment(sigma):
    print(f"\n{'='*50}")
    print(f" Lancement DSBM avec sigma = {sigma}")
    print(f"{'='*50}")
    device = CONFIG["device"]
    model = DriftNetwork(data_dim=2, hidden_dim=256, time_dim=64).to(device)
    loss_fn = BridgeMatchingLoss(model, sigma=sigma)
    
    x0_coupled = get_source_data(CONFIG["n_samples"], device)
    x1_coupled = get_target_data(CONFIG["n_samples"], device)
    
    for imf_iter in range(CONFIG["n_imf"]):
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["n_inner"], eta_min=CONFIG["min_lr"])
        
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
            
            if epoch % 500 == 0:
                print(f"  [sigma={sigma}] Iter {imf_iter+1} | Epoch {epoch:04d}/{CONFIG['n_inner']} | Loss: {loss.item():.4f}")
            
        model.eval()
        x1_ref_np = get_target_data(1000, device).cpu().numpy()
        
        # Spaghetti plot
        visualize_trajectories_spaghetti(
            drift_model=model,
            imf_iter=imf_iter + 1,
            sigma=sigma,
            n_sde_steps=CONFIG["n_sde_steps"],
            device=device,
            data_sample=x1_ref_np,
            plot_dir=CONFIG["plot_dir"],
            n_particles=50
        )
        
        # Renomme pour la comparaison
        import shutil
        old_path = os.path.join(CONFIG["plot_dir"], f"spaghetti_imf{imf_iter + 1:02d}.png")
        new_path = os.path.join(CONFIG["plot_dir"], f"spaghetti_sigma{sigma}_imf{imf_iter + 1:02d}.png")
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            
        if imf_iter < CONFIG["n_imf"] - 1:
            print(f"  Couplage updated for sigma={sigma}, iter={imf_iter+1}")
            x0_new = get_source_data(CONFIG["n_samples"], device)
            x1_new = euler_maruyama(model, x0_new, sigma, CONFIG["n_sde_steps"], device)
            x0_coupled = x0_new.detach()
            x1_coupled = x1_new.detach()

if __name__ == "__main__":
    for sig in [0.8, 0.3, 0.05]:
        run_experiment(sig)
    print("\n✅ Expérience terminée. Graphiques dans", CONFIG["plot_dir"])
