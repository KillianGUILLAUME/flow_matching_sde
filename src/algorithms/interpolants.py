import torch

class Interpolant:
    """
    Classe de base pour gérer la trajectoire x_t entre le bruit (x0) et la data (x1).
    """
    def __init__(self, device):
        self.device = device

    def calc_xt_ut(self, x0, x1, t):
        raise NotImplementedError
    

class LinearInterpolant(Interpolant):
    """
    Rectified Flow
    Coefficients : alpha = 1-t, beta = t, gamma = 0
    """
    def calc_xt_ut(self, x0, x1, t):
        # t est de taille (Batch, 1)
        # x0, x1 sont de taille (Batch, Dim)

        xt = (1 - t) * x0 + t * x1
        
        ut = x1 - x0 #derivatitve with respect to time
        
        return xt, ut
    
class StochasticInterpolant(Interpolant):
    """$
    Trajectoire : Pont Brownien.
    """
    def __init__(self, device, sigma=0.1):
        super().__init__(device)
        self.sigma = sigma

    def calc_xt_ut(self, x0, x1, t):
        z = torch.randn_like(x0).to(self.device)
        
        # Gamma_t = sigma * sqrt(t(1-t))
        term_bruit = torch.sqrt(t * (1 - t) + 1e-8)
        xt = (1 - t) * x0 + t * x1 + self.sigma * term_bruit * z
        
        # velocity (derivatitve with respect to time)
        # d/dt sqrt(t - t^2) = (1 - 2t) / (2 * sqrt(...))
        d_gamma = (1 - 2*t) / (2 * term_bruit)
        ut = (x1 - x0) + self.sigma * d_gamma * z
        
        return xt, ut