import torch
import numpy as np

def check_identity_property(student_model, x_data):
    """
    Vérifie X(x, t, t) = x
    Le modèle ne doit pas bouger si le temps de départ = temps d'arrivée.
    Grâce à notre architecture 'Hard Constraint', ceci devrait être exactement 0 (aux erreurs numériques près).
    """
    batch_size = x_data.shape[0]
    device = x_data.device
    
    # On prend un temps t aléatoire
    t = torch.rand(batch_size, 1).to(device)
    
    # On demande le transport de t à t
    with torch.no_grad():
        x_pred = student_model(x_data, t, t)
        
    mse = torch.mean((x_pred - x_data)**2).item()
    return mse

def check_semigroup_property(student_model, x_init):
    """
    Vérifie la propriété de composition : X(x, s, t) ≈ X(X(x, s, u), u, t)
    Aller de A à C doit être pareil que A->B puis B->C.
    C'est le test le plus dur pour le modèle.
    """
    batch_size = x_init.shape[0]
    device = x_init.device
    
    # On définit s < u < t
    s = torch.zeros(batch_size, 1).to(device)       # Départ (0.0)
    u = torch.ones(batch_size, 1).to(device) * 0.5  # Escale (0.5)
    t = torch.ones(batch_size, 1).to(device)        # Arrivée (1.0)
    
    with torch.no_grad():
        # Trajet 1 : Direct (0 -> 1)
        direct = student_model(x_init, s, t)
        
        # Trajet 2 : Escale (0 -> 0.5 puis 0.5 -> 1)
        step_1 = student_model(x_init, s, u)
        step_2 = student_model(step_1, u, t)
        
    mse = torch.mean((direct - step_2)**2).item()
    return mse

def solve_ode_euler(velocity_field, x_init, steps=100):
    """
    Résout l'équation différentielle dx/dt = v(x,t) pas à pas (Euler).
    Sert de 'Ground Truth' physique.
    """
    dt = 1.0 / steps
    x = x_init.clone()
    t = 0.0
    
    device = x.device
    
    for _ in range(steps):
        t_tensor = torch.ones(x.shape[0], 1).to(device) * t
        with torch.no_grad():
            v = velocity_field(x, t_tensor)
        x = x + v * dt
        t += dt
        
    return x

def check_ode_consistency(student_model, teacher_model, x_init):
    """
    Compare le 'Saut' du Student (rapide) avec l'intégration du Teacher (lent).
    Si l'erreur est faible, le Student a bien appris la physique du Teacher.
    """
    batch_size = x_init.shape[0]
    device = x_init.device
    
    # Temps départ s=0, arrivée t=1
    s = torch.zeros(batch_size, 1).to(device)
    t = torch.ones(batch_size, 1).to(device)
    
    with torch.no_grad():
        # 1. Student (Téléportation)
        x_student = student_model(x_init, s, t)
        
        # 2. Teacher (Intégration lente - 100 pas)
        x_teacher_ode = solve_ode_euler(teacher_model, x_init, steps=100)
        
    mse = torch.mean((x_student - x_teacher_ode)**2).item()
    return mse


# ──────────────────────────────────────────────────────────────────────────────
# DSBM Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────────────

def wasserstein_2(x_gen: np.ndarray, x_ref: np.ndarray, max_samples: int = 500) -> float:
    """
    Calcule la distance de Wasserstein-2 (W₂) entre deux nuages de points 2D.

    Implémentation discrète via le problème d'affectation optimal (Hungarian algorithm).
    Le coût est la distance euclidiende au carré : c(x, y) = ‖x - y‖².
    W₂ = √( min_{σ} (1/N) Σᵢ ‖xᵢ - y_{σ(i)}‖² )

    Args:
        x_gen      : (N, D)  — points générés (ex: sortie SDE du drift DSBM)
        x_ref      : (M, D)  — points de référence (ex: vraies two-moons)
        max_samples: sous-échantillonnage pour limiter la complexité O(N³)

    Returns:
        w2 : float  — distance W₂ ≥ 0 (0 = distributions identiques)
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    # Sous-échantillonnage pour contrôler la complexité
    N = min(max_samples, len(x_gen), len(x_ref))
    idx_gen = np.random.choice(len(x_gen), N, replace=False)
    idx_ref = np.random.choice(len(x_ref), N, replace=False)

    X = x_gen[idx_gen].astype(np.float64)
    Y = x_ref[idx_ref].astype(np.float64)

    # Matrice de coût : distance eucldiienne au carré
    C = cdist(X, Y, metric="sqeuclidean")

    # Affectation optimale (transport discret exact)
    row_ind, col_ind = linear_sum_assignment(C)
    w2_sq = C[row_ind, col_ind].mean()

    return float(np.sqrt(w2_sq))


def mmd_gaussian(
    x_gen: np.ndarray,
    x_ref: np.ndarray,
    bandwidth: float = None,
    max_samples: int = 1000,
) -> float:
    """
    Maximum Mean Discrepancy (MMD) avec un noyau gaussien RBF.

    MMD²(P, Q) = E_{x,x'~P}[k(x,x')] - 2 E_{x~P,y~Q}[k(x,y)] + E_{y,y'~Q}[k(y,y')]
    où k(x, y) = exp(-‖x-y‖² / (2h²)) avec h² = médiane heuristique.

    Robuste au mode collapse : si le modèle ne génère qu'une seule lune,
    la MMD reste élevée même si la W₂ peut être trompeuse.

    Args:
        x_gen      : (N, D)  — points générés
        x_ref      : (M, D)  — points de référence
        bandwidth  : h² (None = médiane heuristique)
        max_samples: sous-échantillonnage

    Returns:
        mmd : float  — MMD ≥ 0 (0 = distributions identiques)
    """
    from scipy.spatial.distance import cdist

    N = min(max_samples, len(x_gen), len(x_ref))
    idx_gen = np.random.choice(len(x_gen), N, replace=False)
    idx_ref = np.random.choice(len(x_ref), N, replace=False)

    X = x_gen[idx_gen].astype(np.float64)
    Y = x_ref[idx_ref].astype(np.float64)

    # Heuristique de la médiane pour la bande passante
    if bandwidth is None:
        XY = np.vstack([X, Y])
        dists = cdist(XY, XY, metric="sqeuclidean")
        median_sq = np.median(dists[dists > 0])
        h2 = median_sq / (2.0 * np.log(len(XY) + 1)) if median_sq > 0 else 1.0
    else:
        h2 = bandwidth

    def rbf(A, B):
        D = cdist(A, B, metric="sqeuclidean")
        return np.exp(-D / (2.0 * h2))

    Kxx = rbf(X, X).mean()
    Kyy = rbf(Y, Y).mean()
    Kxy = rbf(X, Y).mean()

    mmd_sq = Kxx - 2.0 * Kxy + Kyy
    return float(np.sqrt(max(mmd_sq, 0.0)))


def kinetic_energy(
    drift_fn,
    x0: torch.Tensor,
    sigma: float,
    n_steps: int,
    device: str,
) -> float:
    """
    Énergie cinétique empirique des trajectoires SDE générées par drift_fn.

    Approxime l'intégrale ∫₀¹ ‖u_θ(x_t, t)‖² dt sur des trajectoires simulées.
    Dans le cadre du SB, ce coût doit décroître entre itérations IMF et converger
    vers l'énergie minimale du transport optimal entropique.

    La formule discrète est :
        KE ≈ (1/N) Σᵢ  (1/T) Σₜ ‖u_θ(x_t^i, t)‖² · dt

    Args:
        drift_fn  : callable (x, t) -> (B, D)  — drift u_θ appris
        x0        : (B, D)  — points de départ (bruit gaussien)
        sigma     : float   — bruit SDE
        n_steps   : int     — pas d'intégration
        device    : str

    Returns:
        ke : float  — énergie cinétique moyenne ≥ 0
    """
    from src.algorithms.sde_utils import euler_maruyama_trajectory

    dt = 1.0 / n_steps
    B, D = x0.shape

    # Simule les trajectoires et collecte les drifts à chaque pas
    drift_fn.eval()
    all_drift_sq = []

    x = x0.clone().to(device)
    sqrt_dt = dt ** 0.5

    for i in range(n_steps):
        t_val = i * dt
        t_tensor = torch.full((B, 1), t_val, device=device, dtype=x.dtype)
        with torch.no_grad():
            u = drift_fn(x, t_tensor)
        # Énergie à ce pas : ‖u‖² moyenné sur le batch
        all_drift_sq.append((u ** 2).sum(dim=-1).mean().item())
        # Avance d'un pas Euler-Maruyama
        noise = torch.randn_like(x)
        x = x + u * dt + sigma * sqrt_dt * noise

    # Intégrale temporelle par règles des rectangles
    ke = float(np.mean(all_drift_sq) * dt * n_steps)  # = Σ ‖u‖² · dt
    return ke
