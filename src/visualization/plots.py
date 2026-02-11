import matplotlib.pyplot as plt
import numpy as np
import torch
import os



def visualize_field(model, plot_dir, interpolant, epoch, device, data_sample):
    """
    Affiche le champ de vecteurs appris par le Teacher.
    On veut voir des flèches qui poussent le bruit vers les lunes.
    """
    model.eval()
    # Grille de points pour visualiser le champ
    grid_size = 20
    x = np.linspace(-2.5, 2.5, grid_size)
    y = np.linspace(-2.5, 2.5, grid_size)
    xx, yy = np.meshgrid(x, y)
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    # On regarde le champ à t=0.5 (milieu du transport)
    t_val = 0.5
    t_tensor = torch.ones(grid_tensor.shape[0], 1).to(device) * t_val
    
    with torch.no_grad():
        vectors = model(grid_tensor, t_tensor).cpu().numpy()
        
    plt.figure(figsize=(8, 8))
    plt.title(f"Velocity Field at t={t_val} (Epoch {epoch}) | Interpolant {interpolant}")
    
    # Affichage des vecteurs (Quiver)
    plt.quiver(xx, yy, vectors[:, 0].reshape(grid_size, grid_size), 
               vectors[:, 1].reshape(grid_size, grid_size), color='blue', alpha=0.6)
    
    plt.scatter(data_sample[:, 0], data_sample[:, 1], color='red', s=5, alpha=0.3, label="Target Data")
    
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.legend()
    save_folder = os.path.join(plot_dir, interpolant)

    # 2. On crée le dossier s'il n'existe pas (exist_ok=True évite une erreur s'il existe déjà)
    os.makedirs(save_folder, exist_ok=True)

    # 3. Maintenant on peut sauvegarder sans danger
    save_path = os.path.join(save_folder, f"epoch_{epoch}.png")
    plt.savefig(save_path)
    
    plt.close()


def visualize_transport(student_model, epoch, device, fixed_source, fixed_target, plot_dir):
    """
    On prend du bruit (x0) et on demande au student 
    de le transporter à t=1. Ça doit former des Lunes.
    """
    student_model.eval()
    n_vis = fixed_source.shape[0]
    
    
    # 2. Temps : Départ s=0, Arrivée t=1
    s = torch.zeros(n_vis, 1).to(device)
    t = torch.ones(n_vis, 1).to(device)
    
    with torch.no_grad():
        # Le student transporte le bruit FIXE
        x1_pred = student_model(fixed_source, s, t).cpu().numpy()
        
    x0_np = fixed_source.cpu().numpy()
    target_np = fixed_target.cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Source (Toujours la même image car fixed_source ne change pas)
    plt.subplot(1, 3, 1)
    plt.scatter(x0_np[:,0], x0_np[:,1], s=5, alpha=0.5, color='gray')
    plt.title("Fixed Source: Noise")
    plt.xlim(-2.5, 2.5); plt.ylim(-2.5, 2.5)
    
    # Plot 2: Prediction (C'est ça qui doit changer et s'améliorer)
    plt.subplot(1, 3, 2)
    plt.scatter(x1_pred[:,0], x1_pred[:,1], s=5, alpha=0.6, color='blue')
    plt.title(f"Student Output (Epoch {epoch})")
    plt.xlim(-2.5, 2.5); plt.ylim(-2.5, 2.5)

    # Plot 3: Target (Toujours la même image car fixed_target ne change pas)
    plt.subplot(1, 3, 3)
    plt.scatter(target_np[:,0], target_np[:,1], s=5, alpha=0.5, color='red')
    plt.title("Fixed Ground Truth")
    plt.xlim(-2.5, 2.5); plt.ylim(-2.5, 2.5)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/transport_epoch_{epoch}.png")
    plt.close()