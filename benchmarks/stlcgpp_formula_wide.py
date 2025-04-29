import stlcgpp.formula as stlcg
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm

def compute_box_dist_x(x):
    return torch.abs(x[..., 0] - (-3.0) ) ##1

def compute_box_dist_y(x):
    return torch.abs(x[..., 1] - (-0.5)) ##0.3

def P_1():
    dx = stlcg.Predicate("box_dist_x", predicate_function = compute_box_dist_x)
    dy = stlcg.Predicate("box_dist_y", predicate_function = compute_box_dist_y)
   
    within = stlcg.And(dx <= 1.0, dy <= 0.3)
   
    return within


def compute_box_dist_x2(x):
    return torch.abs(x[..., 0] - (-8.5)) ##1.5

def compute_box_dist_y2(x):
    return torch.abs(x[..., 1] - (-0.5)) ##0.3

def P_2():
    dx = stlcg.Predicate("box_dist_x2", predicate_function = compute_box_dist_x2)
    dy = stlcg.Predicate("box_dist_y2", predicate_function = compute_box_dist_y2)
   
    within = stlcg.And(dx <= 1.5, dy <= 0.3)
   
    return within


def compute_box_dist_x3(x):
    return torch.abs(x[..., 0] - (-19.0)) ##5.0

def compute_box_dist_y3(x):
    return torch.abs(x[..., 1] - (-0.4)) ##0.2

def P_3():
    dx = stlcg.Predicate("box_dist_x3", predicate_function = compute_box_dist_x3)
    dy = stlcg.Predicate("box_dist_y3", predicate_function = compute_box_dist_y3)
   
    within = stlcg.And(dx <= 5.0, dy <= 0.2)
   
    return within



def compute_box_dist_x4(x):
    return torch.abs(x[..., 0] - (-67.5)) ##7.5

def compute_box_dist_y4(x):
    return torch.abs(x[..., 1] - 0.4) ##0.2

def P_4():
    dx = stlcg.Predicate("box_dist_x4", predicate_function = compute_box_dist_x4)
    dy = stlcg.Predicate("box_dist_y4", predicate_function = compute_box_dist_y4)
   
    within = stlcg.And(dx <= 7.5, dy <= 0.2)
   
    return within

# --- Build the temporal formula structure ---

def build_formula(T, approximate=False, beta=10.0):
    """Build the STL formula corresponding to the verified MATLAB version."""
    
    if T % 35 != 0:
        raise ValueError("T must be divisible by 35.")
    scale = T// 35
    
    P4 = P_4()
    P3 = P_3()
    P2 = P_2()
    P1 = P_1()
    
    # Following the sequence of operations:
    eventually_P4 = stlcg.Eventually(P4, interval=[8*scale, 9*scale])
    and1 = stlcg.And(eventually_P4, P3)
    
    eventually_and1 = stlcg.Eventually(and1, interval=[6*scale, 7*scale])
    and2 = stlcg.And(eventually_and1, P2)
    
    eventually_and2 = stlcg.Eventually(and2, interval=[6*scale, 11*scale])
    and3 = stlcg.And(eventually_and2, P1)
    
    formula = stlcg.Eventually(and3, interval=[5*scale, 8*scale])
    
    if approximate:
        approx_method = 'logsumexp'
        return lambda x: formula.robustness(x, approx_method=approx_method, temperature=beta)
    return formula.robustness


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
