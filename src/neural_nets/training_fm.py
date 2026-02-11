import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from sklearn.datasets import make_moons
from src.neural_nets.models import VelocityField, FlowMapNetwork
from src.algorithms.losses import LagrangianMapDistillationLossAutograd
from src.visualization.plots import visualize_transport

def get_data(n_samples=1000):
    x, _ = make_moons(n_samples=n_samples, noise=0.05)
    return torch.from_numpy(x).float()

def get_fixed_validation_data(n_samples, device):
    # 1. Fixer la seed temporairement pour les lunes
    x_target, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
    x_target = (x_target - 0.5) / 0.6
    x_target = torch.from_numpy(x_target).float().to(device)
    
    # 2. Fixer la seed pour le bruit de départ (x0)
    # On utilise un Generator local pour ne pas impacter le reste du code
    g = torch.Generator()
    g.manual_seed(42)
    x_source = torch.randn(n_samples, 2, generator=g).to(device)
    
    return x_source, x_target


def load_teacher(device):
    print(f"Loading Teacher from {CONFIG['teacher_path']}...")
    model = VelocityField(data_dim=2, hidden_dim=128, time_dim=64).to(device)
    model.load_state_dict(torch.load(CONFIG["teacher_path"], map_location=device))
    model.eval()
    return model


CONFIG = {
    "n_samples": 4096,
    "n_epochs": 15000,
    "lr": 2e-3, 
    "min_lr": 1e-6,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "teacher_path": "checkpoints/velocity_teacher_stochastic.pth",
    "save_path": "checkpoints/flow_map_student_stochastic_teacher.pth",
    "plot_dir": "figures/flow_map/stochastic"
}
os.makedirs(CONFIG["plot_dir"], exist_ok=True)


def train_flow_map():
    # 1. Configuration
    device = CONFIG["device"]
    print(f"🚀 Training Student (Flow Map) on {device}...")

    teacher = load_teacher(device)
    student = FlowMapNetwork(data_dim=2, hidden_dim=128, time_dim=64).to(device)
    optimizer = optim.Adam(student.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["n_epochs"], eta_min=CONFIG["min_lr"])
    
    # 3. Loss
    loss_fn = LagrangianMapDistillationLossAutograd(teacher, student)
    
    optimizer = optim.Adam(student.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["n_epochs"], eta_min=CONFIG["min_lr"])

    loss_history = []

    print("Generating fixed validation set for visualization...")
    val_source, val_target = get_fixed_validation_data(1000, device)
    
    for epoch in range(1, CONFIG["n_epochs"] + 1):
        student.train()
        optimizer.zero_grad()
        
        # Données (servent à générer des points valides x_s sur la trajectoire)
        x1 = get_data(CONFIG["n_samples"]).to(device)
        x0 = torch.randn_like(x1).to(device)
        
        loss = loss_fn(x0, x1)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.1e}")
            visualize_transport(
                student_model=student, 
                epoch=epoch, 
                device=device, 
                fixed_source=val_source, 
                fixed_target=val_target, 
                plot_dir=CONFIG["plot_dir"]
            )
            
    # Sauvegarde
    torch.save(student.state_dict(), CONFIG["save_path"])
    print(f"✅ Student sauvegardé dans {CONFIG['save_path']}")
    
    # Plot Loss
    plt.figure()
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title('Student Training Loss | Stochastic teacher')
    plt.savefig(f"{CONFIG['plot_dir']}/loss.png")

if __name__ == "__main__":
    train_flow_map()