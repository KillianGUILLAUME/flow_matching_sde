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


def visualize_dsb_samples(
    drift_model,
    epoch: int,
    sigma: float,
    n_sde_steps: int,
    device: str,
    data_sample,          # numpy (N, 2)  — vraie distribution two-moons
    plot_dir: str,
    x0_samples=None,      # torch.Tensor (N, 2) ou None
):
    """
    Génère des échantillons via Euler-Maruyama à partir du drift DSBM appris
    et les visualise face à la vraie distribution two-moons.

    Affiche trois panneaux :
      1. Source  — bruit blanc x₀
      2. Générés — x_T après intégration SDE du drift u_θ
      3. Target  — vraie distribution two-moons
    """
    from src.algorithms.sde_utils import euler_maruyama

    drift_model.eval()
    os.makedirs(plot_dir, exist_ok=True)

    n_vis = data_sample.shape[0]

    # Source
    if x0_samples is None:
        x0 = torch.randn(n_vis, 2, device=device)
    else:
        x0 = x0_samples.to(device)

    # Intégration SDE
    x_gen = euler_maruyama(drift_model, x0, sigma=sigma, n_steps=n_sde_steps, device=device)
    x_gen_np = x_gen.cpu().numpy()
    x0_np    = x0.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"DSBM — IMF Iteration {epoch}", fontsize=13)

    axes[0].scatter(x0_np[:, 0], x0_np[:, 1], s=6, alpha=0.5, color="gray")
    axes[0].set_title("Source: Bruit blanc")
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3)
    axes[0].set_aspect("equal")

    axes[1].scatter(x_gen_np[:, 0], x_gen_np[:, 1], s=6, alpha=0.6, color="steelblue")
    axes[1].set_title(f"Générés (SDE drift u_θ, σ={sigma})")
    axes[1].set_xlim(-3, 3); axes[1].set_ylim(-3, 3)
    axes[1].set_aspect("equal")

    axes[2].scatter(data_sample[:, 0], data_sample[:, 1], s=6, alpha=0.5, color="tomato")
    axes[2].set_title("Target: Two Moons")
    axes[2].set_xlim(-3, 3); axes[2].set_ylim(-3, 3)
    axes[2].set_aspect("equal")

    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"imf_iter_{epoch:02d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  📊 Figure sauvegardée : {save_path}")


def visualize_drift_quiver(
    drift_model,
    imf_iter: int,
    device: str,
    data_sample,          # numpy (N, 2) — vraie distribution two-moons
    plot_dir: str,
    t_values: list = None,
    grid_size: int = 18,
    xlim: tuple = (-2.8, 2.8),
    ylim: tuple = (-2.0, 2.0),
):
    """
    Quiver plot du champ de dérive u_θ(x, t) à plusieurs instants.

    Pour chaque t dans t_values (défaut : 0.2, 0.5, 0.8), on évalue le drift
    du modèle DSBM sur une grille régulière et on trace les flèches.
    Les flèches devraient s'organiser vers les deux lunes à mesure qu'IMF progresse.

    Args:
        drift_model : DriftNetwork entraîné
        imf_iter    : indice de l'itération IMF courante (pour le titre)
        device      : str
        data_sample : (N, 2) numpy — two-moons de référence (affiché en fond)
        plot_dir    : dossier de sortie
        t_values    : liste de temps à visualiser (défaut [0.2, 0.5, 0.8])
        grid_size   : résolution de la grille
        xlim, ylim  : bornes du domaine
    """
    if t_values is None:
        t_values = [0.2, 0.5, 0.8]

    drift_model.eval()
    os.makedirs(plot_dir, exist_ok=True)

    n_t = len(t_values)
    fig, axes = plt.subplots(1, n_t, figsize=(5 * n_t, 5))
    if n_t == 1:
        axes = [axes]
    fig.suptitle(f"Champ de dérive u_θ(x, t) — IMF iter {imf_iter}", fontsize=13)

    # Grille de points
    xs = np.linspace(*xlim, grid_size)
    ys = np.linspace(*ylim, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.c_[XX.ravel(), YY.ravel()].astype(np.float32)
    grid_t = torch.from_numpy(grid).to(device)

    for ax, t_val in zip(axes, t_values):
        t_tensor = torch.full((len(grid_t), 1), t_val, device=device)
        with torch.no_grad():
            u = drift_model(grid_t, t_tensor).cpu().numpy()

        ux = u[:, 0].reshape(grid_size, grid_size)
        uy = u[:, 1].reshape(grid_size, grid_size)
        speed = np.sqrt(ux**2 + uy**2) + 1e-8

        # Flèches normalisées en longueur, colorées par vitesse
        ax.quiver(
            XX, YY, ux / speed, uy / speed,
            speed,
            cmap="plasma",
            scale=grid_size * 1.2,
            width=0.003,
            alpha=0.85,
        )

        # Fond : vraie distribution two-moons (très transparent)
        ax.scatter(
            data_sample[:, 0], data_sample[:, 1],
            s=4, color="white", alpha=0.25, zorder=2
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        ax.set_title(f"t = {t_val:.1f}", color="white", fontsize=11)
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    fig.patch.set_facecolor("#1a1a2e")
    plt.tight_layout()

    save_path = os.path.join(plot_dir, f"quiver_imf{imf_iter:02d}.png")
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  🧭 Quiver plot sauvegardé : {save_path}")


def visualize_trajectories_spaghetti(
    drift_model,
    imf_iter: int,
    sigma: float,
    n_sde_steps: int,
    device: str,
    data_sample,          # numpy (N, 2) — vraie distribution two-moons
    plot_dir: str,
    n_particles: int = 50,
    xlim: tuple = (-3.0, 3.0),
    ylim: tuple = (-2.5, 2.5),
):
    """
    Spaghetti plot : traces de n_particles particules de t=0 à t=1 via le SDE u_θ.

    À l'itération 1 (couplage indépendant) : trajectoires chaotiques et emmêlées.
    Aux itérations suivantes    : flux de plus en plus laminaires,
    convergeant vers les deux lunes malgré le bruit brownien.

    Chaque particule est colorée par son index (arc-en-ciel) pour distinguer les flux
    vers chacune des deux lunes.

    Args:
        drift_model  : DriftNetwork entraîné
        imf_iter     : indice IMF (pour titre et nom de fichier)
        sigma        : bruit SDE
        n_sde_steps  : pas Euler-Maruyama
        device       : str
        data_sample  : (N, 2) numpy — two-moons de référence (affiché en fond)
        plot_dir     : dossier de sortie
        n_particles  : nombre de particules à tracer (défaut 50)
        xlim, ylim   : bornes du domaine
    """
    from src.algorithms.sde_utils import euler_maruyama_trajectory

    drift_model.eval()
    os.makedirs(plot_dir, exist_ok=True)

    # Départ fixe pour reproductibilité entre itérations
    rng = np.random.default_rng(0)
    x0_np = rng.standard_normal((n_particles, 2)).astype(np.float32)
    x0 = torch.from_numpy(x0_np).to(device)

    # Simulation complète — traj : (n_steps+1, n_particles, 2)
    traj, ts = euler_maruyama_trajectory(drift_model, x0, sigma, n_sde_steps, device)
    traj_np = traj.cpu().numpy()   # (T, N, 2)
    ts_np   = ts.numpy()           # (T,)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    # Fond : vraie distribution two-moons
    ax.scatter(
        data_sample[:, 0], data_sample[:, 1],
        s=6, color="#4dabf7", alpha=0.18, zorder=1, label="Two Moons (target)"
    )

    # Palette arc-en-ciel cyclique — chaque particule a sa couleur
    cmap = plt.get_cmap("hsv")
    colors = [cmap(i / n_particles) for i in range(n_particles)]

    for i in range(n_particles):
        path_x = traj_np[:, i, 0]
        path_y = traj_np[:, i, 1]
        ax.plot(path_x, path_y, color=colors[i], linewidth=0.6, alpha=0.55, zorder=2)
        # Point de départ
        ax.scatter(path_x[0], path_y[0], color=colors[i], s=12, zorder=3, edgecolors="none")
        # Point d'arrivée
        ax.scatter(path_x[-1], path_y[-1], color=colors[i], s=20, marker="*", zorder=4, edgecolors="none")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(
        f"Trajectoires SDE — IMF iter {imf_iter}  (σ={sigma}, {n_particles} particules)",
        color="white", fontsize=11
    )
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # Légende compacte
    ax.scatter([], [], s=10, color="#4dabf7", alpha=0.5, label="Two Moons (cible)")
    ax.scatter([], [], s=12, color="white", alpha=0.6, label="Départ (●)")
    ax.scatter([], [], s=20, marker="*", color="white", alpha=0.6, label="Arrivée (★)")
    leg = ax.legend(facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=8, loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"spaghetti_imf{imf_iter:02d}.png")
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  🍝 Spaghetti plot sauvegardé : {save_path}")