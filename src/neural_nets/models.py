import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class VelocityField(nn.Module):
    """
    Teacher : v_θ(x_t, t)
    Apprend le champ de vitesse instantané.
    """
    def __init__(self, data_dim=2, hidden_dim=128, time_dim=64):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t):
        """
        x: (B, data_dim) - position courante
        t: (B, 1) - temps
        Returns: (B, data_dim) - vitesse instantanée
        """
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


class FlowMapNetwork(nn.Module):
    """
    Student : X_φ(x, s, t)
    Apprend la carte de transport avec hard constraint.
    """
    def __init__(self, data_dim=2, hidden_dim=128, time_dim=64):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        
        self.net = nn.Sequential(
            nn.Linear(data_dim + 2*time_dim, hidden_dim),
            nn.SiLU(),  # SiLU partout
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, s, t):
        """
        x: (B, data_dim) - position initiale
        s: (B, 1) - temps départ
        t: (B, 1) - temps arrivée
        Returns: (B, data_dim) - position à t
        """
        s_emb = self.time_embed(s)
        t_emb = self.time_embed(t)
        h = torch.cat([x, s_emb, t_emb], dim=-1)
        
        # Prédit la vitesse moyenne sur [s, t]
        avg_velocity = self.net(h)
        
        # Hard constraint : X(x, s, s) = x car (t-s)=0
        # Interprétation : déplacement = vitesse_moyenne × durée
        return x + (t - s) * avg_velocity


class DriftNetwork(nn.Module):
    """
    Réseau de drift pour le Diffusion Schrödinger Bridge Matching (DSBM).

    Apprend u_θ(x_t, t) — le drift du SDE :
        dx_t = u_θ(x_t, t) dt + σ dW_t

    Architecturalement identique à VelocityField, mais sémantiquement différent :
    ici on apprend le drift d'un processus STOCHASTIQUE via la Bridge Matching Loss,
    et non le champ de vitesse d'un ODE déterministe.

    Référence : Shi et al., NeurIPS 2023, Algorithm 2 (DSBM).
    """

    def __init__(self, data_dim: int = 2, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)

        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, data_dim)  — position courante x_t
        t : (B, 1)         — temps courant
        Returns : (B, data_dim)  — drift u_θ(x_t, t)
        """
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)
