import stlcgpp.formula as stlcg
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm
import stlcg.stlcg as stlcg

class RobustnessModule(torch.nn.Module):
    def __init__(self, func, bs):
        super(RobustnessModule, self).__init__()
        self.func = func
        self.bs = bs

    def forward(self, x):
        # return self.func(x)
        res = torch.stack([self.func(x[i])[0, 0, 0] for i in range(self.bs)], dim=0)
        print("res shape:", res.shape)
        return res 

def goal_1(x):
    dx = stlcg.Expression("box_dist_x", torch.abs(x[..., 0] - 5.5).unsqueeze(-1))
    dy = stlcg.Expression("box_dist_y", torch.abs(x[..., 1] - 3.5).unsqueeze(-1))
    
    within = stlcg.And(dx <= 0.5, dy <= 0.5)
    return (within, (dx, dy))

def goal_2(x):
    dx = stlcg.Expression("box_dist_x", torch.abs(x[..., 0] - 3.5).unsqueeze(-1))
    dy = stlcg.Expression("box_dist_y", torch.abs(x[..., 1] - 0.5).unsqueeze(-1))
    
    within = stlcg.And(dx <= 0.5, dy <= 0.5)
    return (within, (dx, dy))

    
def safe(x):
    dx = stlcg.Expression("box_dist_x", torch.abs(x[..., 0] - 2.5).unsqueeze(-1))
    dy = stlcg.Expression("box_dist_y", torch.abs(x[..., 1] - 3.5).unsqueeze(-1))
    
    within = stlcg.Or(dx > 1.5, dy > 1.5)
    return (within, (dx, dy))


def always_safe(x):
    safe_formula, inputs = safe(x)
    return (stlcg.Always(safe_formula), inputs)

def binary_combine_exprs(expressions, operation):
    # expressions: List of tuples of STL expressions and inputs
    # operation: Binary operation to combine expresisons
    while len(expressions) > 1:
        sub_expr = []
        for i in range(0, len(expressions), 2):
            if i + 1 < len(expressions):
                expr1, in1 = expressions[i]
                expr2, in2 = expressions[i + 1]
                sub_expr += [(operation(expr1, expr2), (in1, in2))]
            else:
                # Handle odd number of epxressions
                sub_expr += [expressions[i]]
        expressions = sub_expr
    return expressions[0]

def eventually_goal1_then_eventually_goal2(x, T):
           
    # for i in range(0, T):
       
    #     eventually_goal2 = stlcg.Eventually(goal2, interval=[i+1, T])
    #     eventually_goal1 = stlcg.Eventually(goal1, interval=[i, i])
    #     if i == 0:
    #         subformulas = stlcg.And(eventually_goal1  , eventually_goal2)
    #     elif i > 0:
    #         subformula = stlcg.And(eventually_goal1, eventually_goal2)
    #         subformulas = stlcg.Or(subformulas,subformula)
   
    # ################################################################
    
    goal2, goal2_input = goal_2(x)
    goal1, goal1_input = goal_1(x)
    subformulas = []
    for i in range(0, T):
        eventually_goal2 = stlcg.Eventually(goal2, interval=[i+1, T])
        if i == 0:
            subformula = stlcg.And(goal1, eventually_goal2)
        elif i > 0:
            subformula = stlcg.And(stlcg.Eventually(goal1, interval=[i, i]), eventually_goal2)
        subformulas += [(subformula, (goal1_input, goal2_input))]
    
    return binary_combine_exprs(subformulas, stlcg.Or)



def robustness(x, scale=-1.0):
    x = torch.flip(x, (0,)).unsqueeze(0)

    formula1, inputs1 = eventually_goal1_then_eventually_goal2(x, x.shape[1]-1)
    formula2, inputs2 = always_safe(x)
    formula = formula1 & formula2
    inputs = (inputs1, inputs2)
    pscale = 1
    scale  = scale
    rho = formula.robustness(inputs, pscale=pscale, scale=scale)

    return rho

from functools import partial
def get_robustness_function(T, approximate=False, beta=10.0, apply_JIT = False, device=None, bs = 10):
    f = partial(robustness, scale = beta if approximate else -1.0)
    rm = RobustnessModule(f, bs).to(device)
    
    if apply_JIT:
        sample_trajectory = torch.randn((bs, T+1, 2)).to(device)
        rm = torch.jit.trace(rm, sample_trajectory).to(device)
    
    return rm
    
    
    
    

if __name__ == "__main__":
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    T = 20
    bs = 2
    epochs = 1000
    trajectory = torch.randn( bs, T+1, 2).to(device)
    apply_JIT = False
    
    rf_without_jit = get_robustness_function(T, approximate=False, beta=10, apply_JIT=apply_JIT, device=device, bs=bs)
    
    start = time.perf_counter()
    for i in tqdm(range(epochs)):
        v2 = rf_without_jit(trajectory)
    end = time.perf_counter()
    print("Time taken without JIT trace: ", end - start)
    
    exit()
    
    
    
    rf_with_jit = get_robustness_function(T, approximate=False, beta=10, apply_JIT=True, device=device, bs=bs)
    start = time.perf_counter()
    for i in tqdm(range(epochs)):
        v1 = rf_with_jit(trajectory)
    end = time.perf_counter()
    print("Time taken with JIT trace: ", end - start)
    
    
    print("Diff:", torch.max(torch.abs(v1 - v2)))
    