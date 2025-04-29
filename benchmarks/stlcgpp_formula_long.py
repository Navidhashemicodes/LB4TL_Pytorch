import stlcgpp.formula as stlcg
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm

def compute_box_dist_x(x):
    return torch.abs(x[..., 0] - 0 ) 

def compute_box_dist_y(x):
    return torch.abs(x[..., 1] - 0) 

def P_0():
    dx = stlcg.Predicate("box_dist_x", predicate_function = compute_box_dist_x)
    dy = stlcg.Predicate("box_dist_y", predicate_function = compute_box_dist_y)
   
    within = stlcg.And(dx <= 1, dy <= 1)
   
    return within


# --- Build the temporal formula structure ---

def build_formula(T, approximate=False, beta=10.0):
    """Build the STL formula corresponding to the verified MATLAB version."""
    
    if T % 20 != 0:
        raise ValueError("T must be divisible by 35.")
    portion = T// 20
    P0 = P_0()
    
    Evs = stlcg.Eventually(P0, interval=[0, portion])
    
    for i in range(1,20):
        Evs = stlcg.And( Evs , stlcg.Eventually(P0, interval=[i*portion, (i+1)*portion]) )
     
    
    if approximate:
        approx_method = 'logsumexp'
        return lambda x: Evs.robustness(x, approx_method=approx_method, temperature=beta)
    return Evs.robustness


class RobustnessModule(nn.Module):
    def __init__(self, func):
        super(RobustnessModule, self).__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

def get_robustness_function(T, approximate=False, beta=10.0, apply_JIT = False, device=None, bs = 10):
    specification = build_formula(T, approximate=approximate, beta=beta)
    sample_trajectory = torch.randn(1, T+1, 2).to(device)
    rm = RobustnessModule(specification).to(device)
    if apply_JIT:
        rm = torch.jit.trace(rm, (sample_trajectory[0])).to(device)
    return torch.vmap(rm)