import torch
import torch.nn.functional as F

# ============================================
# Loss pour entrainer le velocity field (teacher)
# ============================================


class ConditionalFlowMatchingLoss:
    """
    Loss pour entraîner le velocity field (Teacher).
    Basée sur l'équation (3.4) du papier.
    """
    
    def __init__(self, velocity_field):
        self.velocity_field = velocity_field
    
    def __call__(self, x_0, x_1):
        """
        Args:
            x_0: (B, data_dim) - échantillons source p_0
            x_1: (B, data_dim) - échantillons target p_1
        
        Returns:
            loss: scalaire
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample temps uniformément
        t = torch.rand(batch_size, 1, device=device)
        
        # Interpolation conditionnelle (conditional flow)
        x_t = (1 - t) * x_0 + t * x_1
        
        # Target velocity (formule analytique de dx_t/dt)
        v_target = x_1 - x_0
        
        # Prédiction du réseau
        v_pred = self.velocity_field(x_t, t)
        
        # Loss MSE
        loss = F.mse_loss(v_pred, v_target)
        
        return loss


# Fonction helper pour entraînement complet
def train_teacher(velocity_field, data_loader, num_epochs=1000, lr=1e-3):
    """
    Entraîne le velocity field avec Conditional Flow Matching.
    
    Args:
        velocity_field: VelocityField à entraîner
        data_loader: DataLoader donnant des échantillons de p_1
        num_epochs: nombre d'époques
        lr: learning rate
    """
    cfm_loss = ConditionalFlowMatchingLoss(velocity_field)
    optimizer = torch.optim.Adam(velocity_field.parameters(), lr=lr)
    
    velocity_field.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for x_1 in data_loader:
            # x_0 ~ N(0, I) (source gaussienne)
            x_0 = torch.randn_like(x_1)
            
            optimizer.zero_grad()
            loss = cfm_loss(x_0, x_1)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if epoch % 100 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return velocity_field



# ============================================
# Lagrangian loss pour entrainer le flow map (student)
# ============================================


class LagrangianMapDistillationLoss:
    """
    Implémente la loss de l'Algorithm 1 : Lagrangian map distillation (LMD)
    
    L(φ) = E_{x_0~p_0, x_1~p_1, s,t~U[0,1]} [||∂_t X_φ(x_s, s, t) - v_θ(X_φ(x_s, s, t), t)||²]
    
    Où:
    - x_s = (1-s)x_0 + s·x_1  (interpolation linéaire, conditional flow matching)
    - X_φ est le student (FlowMapNetwork)
    - v_θ est le teacher pré-entraîné (VelocityField)
    """
    
    def __init__(self, velocity_field, flow_map):
        """
        Args:
            velocity_field: VelocityField pré-entraîné (frozen)
            flow_map: FlowMapNetwork à entraîner
        """
        self.velocity_field = velocity_field
        self.flow_map = flow_map
        
        # On freeze le teacher
        for param in self.velocity_field.parameters():
            param.requires_grad = False
        self.velocity_field.eval()
    
    def compute_time_derivative(self, x, s, t, epsilon=1e-4):
        """
        Calcule ∂_t X_φ(x, s, t) par différences finies.
        
        ∂_t X ≈ [X(x, s, t+ε) - X(x, s, t-ε)] / (2ε)
        
        Args:
            x: (B, data_dim) - position initiale
            s: (B, 1) - temps départ
            t: (B, 1) - temps courant
            epsilon: pas pour différences finies
        
        Returns:
            (B, data_dim) - dérivée temporelle
        """
        # Clamp pour rester dans [0, 1]
        t_plus = torch.clamp(t + epsilon, 0, 1)
        t_minus = torch.clamp(t - epsilon, 0, 1)
        
        # Évalue le flow map aux deux points
        x_t_plus = self.flow_map(x, s, t_plus)
        x_t_minus = self.flow_map(x, s, t_minus)
        
        # Différence finie centrée
        dt_X = (x_t_plus - x_t_minus) / (t_plus - t_minus)
        
        return dt_X
    
    def __call__(self, x_0, x_1):
        """
        Calcule la loss de distillation.
        
        Args:
            x_0: (B, data_dim) - échantillons de p_0 (source, ex: gaussienne)
            x_1: (B, data_dim) - échantillons de p_1 (target, vraies données)
        
        Returns:
            loss: scalaire
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # 1. Sample s, t ~ U[0, 1]
        s = torch.rand(batch_size, 1, device=device)
        t = torch.rand(batch_size, 1, device=device)
        
        # 2. Calcule x_s = (1-s)x_0 + s·x_1 (Conditional Flow Matching interpolation)
        x_s = (1 - s) * x_0 + s * x_1
        
        # 3. Évalue X_φ(x_s, s, t)
        X_phi_t = self.flow_map(x_s, s, t)
        
        # 4. Calcule ∂_t X_φ(x_s, s, t)
        dt_X_phi = self.compute_time_derivative(x_s, s, t)
        
        # 5. Évalue v_θ(X_φ(x_s, s, t), t) - le teacher frozen
        with torch.no_grad():
            v_theta = self.velocity_field(X_phi_t.detach(), t)
        
        # 6. Loss L2
        loss = F.mse_loss(dt_X_phi, v_theta)
        
        return loss


# ============================================
# VERSION ALTERNATIVE : Dérivée automatique
# ============================================

class LagrangianMapDistillationLossAutograd:
    """
    Version utilisant torch.autograd pour calculer ∂_t X_φ.
    Plus précise mais plus coûteuse en mémoire.
    """
    
    def __init__(self, velocity_field, flow_map):
        self.velocity_field = velocity_field
        self.flow_map = flow_map
        
        for param in self.velocity_field.parameters():
            param.requires_grad = False
        self.velocity_field.eval()
    
    def compute_time_derivative_autograd(self, x, s, t):
        """
        Calcule ∂_t X_φ(x, s, t) avec autograd.
        
        Returns:
            (B, data_dim) - dérivée exacte
        """
        # On doit activer les gradients sur t
        t_with_grad = t.clone().requires_grad_(True)
        
        # Forward pass
        X_phi = self.flow_map(x, s, t_with_grad)
        
        # Calcule grad(X_phi, t) pour chaque dimension
        dt_X = []
        for i in range(X_phi.shape[1]):
            grad_outputs = torch.zeros_like(X_phi)
            grad_outputs[:, i] = 1.0
            
            grad_t = torch.autograd.grad(
                outputs=X_phi,
                inputs=t_with_grad,
                grad_outputs=grad_outputs,
                create_graph=True,  # Pour backprop à travers la dérivée
                retain_graph=True
            )[0]
            
            dt_X.append(grad_t)
        
        return torch.cat(dt_X, dim=-1)
    
    def __call__(self, x_0, x_1):
        batch_size = x_0.shape[0]
        device = x_0.device
        
        s = torch.rand(batch_size, 1, device=device)
        t = torch.rand(batch_size, 1, device=device)
        
        x_s = (1 - s) * x_0 + s * x_1
        
        X_phi_t = self.flow_map(x_s, s, t)
        dt_X_phi = self.compute_time_derivative_autograd(x_s, s, t)
        
        with torch.no_grad():
            v_theta = self.velocity_field(X_phi_t.detach(), t)
        
        loss = F.mse_loss(dt_X_phi, v_theta)
        
        return loss


# ============================================
# VERSION OPTIMISÉE : Utilise la hard constraint
# ============================================

class LagrangianMapDistillationLossOptimized:
    """
    Version exploitant X_φ(x, s, t) = x + (t-s)·v̄(x, s, t).
    
    On a : ∂_t X_φ = ∂_t[(t-s)·v̄] = v̄ + (t-s)·∂_t v̄
    
    Approximation : si on ignore ∂_t v̄, alors ∂_t X_φ ≈ v̄
    Ce qui donne une loss simplifiée !
    """
    
    def __init__(self, velocity_field, flow_map):
        self.velocity_field = velocity_field
        self.flow_map = flow_map
        
        for param in self.velocity_field.parameters():
            param.requires_grad = False
        self.velocity_field.eval()
    
    def __call__(self, x_0, x_1):
        batch_size = x_0.shape[0]
        device = x_0.device
        
        s = torch.rand(batch_size, 1, device=device)
        t = torch.rand(batch_size, 1, device=device)
        
        x_s = (1 - s) * x_0 + s * x_1
        
        # Évalue le flow map
        X_phi_t = self.flow_map(x_s, s, t)
        
        # ASTUCE : Avec notre architecture X = x + (t-s)·net(x,s,t)
        # On peut extraire directement la vitesse moyenne :
        with torch.enable_grad():
            s_emb = self.flow_map.time_embed(s)
            t_emb = self.flow_map.time_embed(t)
            h = torch.cat([x_s, s_emb, t_emb], dim=-1)
            avg_velocity = self.flow_map.net(h)  # C'est v̄
        
        # Le teacher donne la vitesse instantanée
        with torch.no_grad():
            v_theta = self.velocity_field(X_phi_t.detach(), t)
        
        # Loss : la vitesse moyenne doit matcher la vitesse instantanée
        loss = F.mse_loss(avg_velocity, v_theta)
        
        return loss
