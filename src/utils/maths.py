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
