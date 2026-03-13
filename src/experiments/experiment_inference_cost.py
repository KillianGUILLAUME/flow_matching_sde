import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.neural_nets.models import VelocityField, DriftNetwork, FlowMapNetwork
from src.algorithms.sde_utils import euler_maruyama

CONFIG = {
    "n_samples": 10000,
    "n_steps": 100,      # Pas pour CFM (ODE) et DSBM (SDE)
    "sigma": 0.5,        # Bruit pour DSBM
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "plot_dir": "figures/inference_cost",
}

os.makedirs(CONFIG["plot_dir"], exist_ok=True)

def benchmark_cfm(n_samples, n_steps, device):
    """
    Conditional Flow Matching (Teacher).
    Résolution de l'ODE dZ_t = v_θ(Z_t, t) dt par méthode d'Euler.
    """
    # On simule un modèle initialisé aléatoirement, on mesure juste l'inférence
    model = VelocityField(data_dim=2, hidden_dim=128, time_dim=64).to(device)
    model.eval()
    
    x = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps
    
    # Warmup
    for _ in range(5):
        t_tensor = torch.zeros(n_samples, 1, device=device)
        _ = model(x, t_tensor)
        
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for i in range(n_steps):
        t_val = i * dt
        t_tensor = torch.full((n_samples, 1), t_val, device=device, dtype=x.dtype)
        with torch.no_grad():
            v = model(x, t_tensor)
        x = x + v * dt
        
    if torch.cuda.is_available(): torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    return end_time - start_time


def benchmark_dsbm(n_samples, n_steps, sigma, device):
    """
    Diffusion Schrödinger Bridge Matching.
    Résolution SDE dx_t = u_θ(x_t, t) dt + σ dW_t par Euler-Maruyama.
    """
    model = DriftNetwork(data_dim=2, hidden_dim=256, time_dim=64).to(device)
    model.eval()
    
    x = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps
    sqrt_dt = dt ** 0.5
    
    # Warmup
    for _ in range(5):
        t_tensor = torch.zeros(n_samples, 1, device=device)
        _ = model(x, t_tensor)
        
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for i in range(n_steps):
        t_val = i * dt
        t_tensor = torch.full((n_samples, 1), t_val, device=device, dtype=x.dtype)
        
        with torch.no_grad():
            u = model(x, t_tensor)
            
        noise = torch.randn_like(x)
        x = x + u * dt + sigma * sqrt_dt * noise
        
    if torch.cuda.is_available(): torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    return end_time - start_time


def benchmark_lmd(n_samples, device):
    """
    Lagrangian Map Distillation (Student).
    Génération en 1 SEUL PAS (One-Step Generation) de s=0 à t=1.
    """
    model = FlowMapNetwork(data_dim=2, hidden_dim=128, time_dim=64).to(device)
    model.eval()
    
    x = torch.randn(n_samples, 2, device=device)
    s = torch.zeros(n_samples, 1, device=device)
    t = torch.ones(n_samples, 1, device=device)
    
    # Warmup
    for _ in range(5):
        _ = model(x, s, t)
        
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        x1 = model(x, s, t)
        
    if torch.cuda.is_available(): torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    return end_time - start_time


if __name__ == "__main__":
    n = CONFIG["n_samples"]
    d = CONFIG["device"]
    steps = CONFIG["n_steps"]
    
    print(f"\n==================================================")
    print(f" Benchmark d'Inférence : {n} échantillons sur {d}")
    print(f"==================================================")
    
    # 1. CFM (Teacher - Flow Matching classique)
    print(f"\n⏱️  CFM  (ODE, {steps} étapes)...")
    time_cfm = benchmark_cfm(n, steps, d)
    print(f"   => Temps : {time_cfm:.4f} s  ({n/time_cfm:.0f} samples/sec)")
    
    # 2. DSBM (Schrödinger Bridge Matching)
    print(f"\n⏱️  DSBM (SDE, {steps} étapes, σ={CONFIG['sigma']})...")
    time_dsbm = benchmark_dsbm(n, steps, CONFIG["sigma"], d)
    print(f"   => Temps : {time_dsbm:.4f} s  ({n/time_dsbm:.0f} samples/sec)")
    
    # 3. LMD (Student - Lagrangian Map Distillation)
    print(f"\n⏱️  LMD  (Réseau 1-step)...")
    time_lmd = benchmark_lmd(n, d)
    print(f"   => Temps : {time_lmd:.4f} s  ({n/time_lmd:.0f} samples/sec)")
    
    # --- Calcul du Speedup ---
    speedup_cfm = time_cfm / time_lmd
    speedup_dsbm = time_dsbm / time_lmd
    print(f"\n🚀 LMD est {speedup_cfm:.1f}x plus rapide que CFM")
    print(f"🚀 LMD est {speedup_dsbm:.1f}x plus rapide que DSBM")
    print(f"==================================================\n")
    
    # --- Visualisation ---
    models = ["CFM\n(ODE 100 pts)", "DSBM\n(SDE 100 pts)", "LMD\n(1-Step)"]
    times = [time_cfm, time_dsbm, time_lmd]
    colors = ["steelblue", "forestgreen", "crimson"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, times, color=colors, alpha=0.85)
    
    ax.set_yscale("log")
    ax.set_ylabel("Temps d'inférence (secondes, Log Scale)")
    ax.set_title(f"Coût de Génération pour {n} échantillons ({d.upper()})", fontsize=13)
    
    # Ajouter les valeurs sur les barres
    for bar, t, name in zip(bars, times, models):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval * 1.15,
                f"{t:.4f} s", ha='center', va='bottom', fontweight='bold')
                
    # Annotation du speedup
    ax.annotate(f"Speedup:\n{speedup_cfm:.0f}x plus rapide", 
                xy=(1.8, time_lmd*1.5), xytext=(0, time_cfm/2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                ha='center', fontsize=11, color="crimson", fontweight='bold')
                
    plt.tight_layout()
    save_path = os.path.join(CONFIG["plot_dir"], "inference_cost_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Graphique de comparaison sauvegardé : {save_path}")
