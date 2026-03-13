"""
Utilitaires pour la simulation de SDEs (Diffusion Schrödinger Bridge Matching).

Référence : Shi et al., "Diffusion Schrödinger Bridge Matching", NeurIPS 2023.
"""

import torch


def euler_maruyama(drift_fn, x0: torch.Tensor, sigma: float, n_steps: int, device: str):
    """
    Intègre le SDE  dx_t = u_θ(x_t, t) dt + σ dW_t  par schéma d'Euler-Maruyama.

    L'intégration va de t=0 à t=1.

    Args:
        drift_fn : callable (x, t_tensor) -> (B, D)  —  drift u_θ
        x0       : (B, D)  —  points de départ
        sigma    : float   —  amplitude du bruit du SDE de référence
        n_steps  : int     —  nombre de pas d'intégration
        device   : str

    Returns:
        x_T : (B, D)  —  position finale après intégration jusqu'à t=1
    """
    dt = 1.0 / n_steps
    sqrt_dt = dt ** 0.5

    x = x0.clone().to(device)
    B = x.shape[0]

    for i in range(n_steps):
        t_val = i * dt
        t_tensor = torch.full((B, 1), t_val, device=device, dtype=x.dtype)

        with torch.no_grad():
            u = drift_fn(x, t_tensor)

        noise = torch.randn_like(x)
        x = x + u * dt + sigma * sqrt_dt * noise

    return x


def euler_maruyama_trajectory(
    drift_fn, x0: torch.Tensor, sigma: float, n_steps: int, device: str
):
    """
    Comme `euler_maruyama` mais renvoie toute la trajectoire.

    Returns:
        traj : (n_steps+1, B, D)  —  positions à chaque pas de temps
        ts   : (n_steps+1,)       —  instants correspondants
    """
    dt = 1.0 / n_steps
    sqrt_dt = dt ** 0.5

    x = x0.clone().to(device)
    B = x.shape[0]

    traj = [x.clone()]
    ts = [0.0]

    for i in range(n_steps):
        t_val = i * dt
        t_tensor = torch.full((B, 1), t_val, device=device, dtype=x.dtype)

        with torch.no_grad():
            u = drift_fn(x, t_tensor)

        noise = torch.randn_like(x)
        x = x + u * dt + sigma * sqrt_dt * noise

        traj.append(x.clone())
        ts.append((i + 1) * dt)

    return torch.stack(traj, dim=0), torch.tensor(ts)


def sample_brownian_bridge_point(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma: float
):
    """
    Tire un point x_t du Brownian Bridge conditionnel entre x0 et x1 au temps t.

    Loi : x_t | x0, x1  ~  N( (1-t) x0 + t x1,  σ² t(1-t) I )

    Args:
        x0, x1 : (B, D)
        t       : (B, 1)   — temps dans [0, 1]
        sigma   : float

    Returns:
        x_t : (B, D)
    """
    mean = (1.0 - t) * x0 + t * x1
    std = sigma * torch.sqrt(t * (1.0 - t) + 1e-8)
    z = torch.randn_like(x0)
    return mean + std * z


def bridge_target_drift(x_t: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    """
    Calcule la dérive analytique du Brownian Bridge conditionnel en x_t au temps t.

    u*(x_t, t | x1) = (x1 - x_t) / (1 - t)

    C'est la dérive optimale qui conduit de x_t à x1 en temps restant (1-t).
    Clippe t < 1-ε pour éviter la division par zéro.

    Args:
        x_t : (B, D)
        x1  : (B, D)
        t   : (B, 1)

    Returns:
        drift : (B, D)
    """
    eps = 1e-3
    denom = torch.clamp(1.0 - t, min=eps)
    return (x1 - x_t) / denom
