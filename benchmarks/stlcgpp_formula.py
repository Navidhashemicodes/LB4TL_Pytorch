import stlcgpp.formula as stlcg
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm

def compute_box_dist_x(x):
    return torch.abs(x[..., 0] - 5.5)

def compute_box_dist_y(x):
    return torch.abs(x[..., 1] - 3.5)

def compute_box_dist_x2(x):
    return torch.abs(x[..., 0] - 3.5)

def compute_box_dist_y2(x):
    return torch.abs(x[..., 1] - 0.5)

def compute_box_dist_x3(x):
    return torch.abs(x[..., 0] - 2.5)


def compute_box_dist_y3(x):
    return torch.abs(x[..., 1] - 3.5)

def goal_1():
    dx = stlcg.Predicate("box_dist_x", predicate_function = compute_box_dist_x)
    dy = stlcg.Predicate("box_dist_y", predicate_function = compute_box_dist_y)
   
    within = stlcg.And(dx <= 0.5, dy <= 0.5)
   
    return within

def goal_2():
    dx = stlcg.Predicate("box_dist_x2", predicate_function = compute_box_dist_x2)
    dy = stlcg.Predicate("box_dist_y2", predicate_function = compute_box_dist_y2)
   
    within = stlcg.And(dx <= 0.5, dy <= 0.5)
    return within

def safe():
    dx = stlcg.Predicate("box_dist_x3", predicate_function = compute_box_dist_x3)
    dy = stlcg.Predicate("box_dist_y3", predicate_function = compute_box_dist_y3)
    
    outside = stlcg.Or(dx > 1.5, dy > 1.5)
    return outside

def always_safe(T):
    safe_formula = safe()
    return stlcg.Always(safe_formula, interval = [0,T])

def eventually_goal1_then_eventually_goal2(T):
    goal2 = goal_2()
    goal1 = goal_1()
   
    for i in range(0, T):
       
        eventually_goal2 = stlcg.Eventually(goal2, interval=[i+1, T])
        eventually_goal1 = stlcg.Eventually(goal1, interval=[i, i])
        if i == 0:
            subformulas = stlcg.And(eventually_goal1  , eventually_goal2)
        elif i > 0:
            subformula = stlcg.And(eventually_goal1, eventually_goal2)
            subformulas = stlcg.Or(subformulas,subformula)
   
    return subformulas

def build_formula(T, approximate=False, beta=10.0):
    formula1 = eventually_goal1_then_eventually_goal2(T)
    formula2 = always_safe(T)
    formula = stlcg.And(formula1, formula2)
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 255
    bs = 1000
    epochs = 10
    trajectory = torch.randn( bs, T+1, 2).to(device)
    apply_JIT = False
    rf_with_jit = get_robustness_function(T, approximate=False, beta=10, apply_JIT=True, device=device, bs=bs)
    rf_without_jit = get_robustness_function(T, approximate=False, beta=10, apply_JIT=False, device=device, bs=bs)
    
    start = time.perf_counter()
    for i in tqdm(range(epochs)):
        v2 = rf_without_jit(trajectory)
    end = time.perf_counter()
    print("Time taken for without JIT trace: ", end - start)
    
    start = time.perf_counter()
    for i in tqdm(range(epochs)):
        v1 = rf_with_jit(trajectory)
    end = time.perf_counter()
    print("Time taken for with JIT trace: ", end - start)
    
    
    print("Diff:", torch.max(torch.abs(v1 - v2)))
    