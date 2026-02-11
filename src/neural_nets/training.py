import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.neural_nets.models import VelocityField

from sklearn.datasets import make_moons
from src.visualization.plots import visualize_field
from src.algorithms.interpolants import LinearInterpolant, StochasticInterpolant

def get_data(n_samples=1000):
    x, _ = make_moons(n_samples=n_samples, noise=CONFIG["data_noise"])
    x = (x - 0.5) / 0.6
    return torch.from_numpy(x).float()

# Initialisation du modèle et de l'optimiseur
model = VelocityField()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

CONFIG = {
    "n_samples": 4096,
    "n_epochs": 10000,
    "lr": 2e-3,
    "min_lr": 1e-6, 
    "weight_decay": 1e-5,
    "data_noise": 0.05, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "checkpoints",
    "plot_dir": "figures/velocity",
    "interpolant": "stochastic"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
os.makedirs(CONFIG["plot_dir"], exist_ok=True)

def train_velocity_field(interpolant_str: str):
    print(f"🚀 Training Teacher (Velocity Field) on {CONFIG['device']}...")

    try:
        if interpolant_str == 'linear':
            interpolant = LinearInterpolant(CONFIG["device"])
        elif interpolant_str == 'stochastic':
            interpolant = StochasticInterpolant(CONFIG["device"])
    except:
        NotImplementedError('interpolant not implemented yet, go check src/algorithms/interpolant')
    
    # 1. Initialisation du modèle
    model = VelocityField(data_dim=2, hidden_dim=128, time_dim=64).to(CONFIG["device"])
    optimizer = optim.Adam(
        model.parameters(), 
        lr=CONFIG["lr"], 
        weight_decay=CONFIG["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG["n_epochs"], 
        eta_min=CONFIG["min_lr"]
    )
    
    loss_history = []
    for epoch in range(1, CONFIG["n_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        x1 = get_data(CONFIG["n_samples"]).to(CONFIG["device"]) 
        x0 = torch.randn_like(x1).to(CONFIG["device"])
        
        t = torch.rand(CONFIG["n_samples"], 1).to(CONFIG["device"])
        
        xt, ut = interpolant.calc_xt_ut(x0, x1, t)
        
        vt_pred = model(xt, t)
        
        loss = torch.mean((vt_pred - ut) ** 2)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        
        if epoch % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f} | LR: {current_lr:.1e}")
            
        if epoch % 2000 == 0:
            data_visu = get_data(500).numpy()
            visualize_field(model, plot_dir=  f"{CONFIG['plot_dir']}",interpolant= interpolant_str,epoch=epoch, device=CONFIG["device"], data_sample=data_visu)

    save_path = f"{CONFIG['save_dir']}/velocity_teacher_{interpolant_str}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Teacher entraîné et sauvegardé sous : {save_path}")


if __name__ == "__main__":
    train_velocity_field('stochastic')