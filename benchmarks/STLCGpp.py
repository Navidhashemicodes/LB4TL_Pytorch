# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:34:24 2024

@author: navid
"""

import torch
import time
import sys
import pathlib
import benchmarks.stlcgpp.formula as stlcg
from tqdm.auto import tqdm

T = 40
Batch = 1
Epochs = 100
device = torch.device("cpu")

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
    goal2, goal2_input = goal_2(x)
    eventually_goal2 = stlcg.Eventually(goal2)
    
    goal1, goal1_input = goal_1(x)
    
    subformulas = []
    for i in range(0, T):
        eventually_goal2 = stlcg.Eventually(goal2, interval=[i+1, T])
        if i == 0:
            subformula = stlcg.And(goal1, eventually_goal2)
        elif i > 0:
            subformula = stlcg.And(stlcg.Eventually(goal1, interval=[0, i]), eventually_goal2)
        subformulas += [(subformula, (goal1_input, goal2_input))]
    
    return binary_combine_exprs(subformulas, stlcg.Or)

def robustness(x):
    
    x = torch.flip(x, (0,)).unsqueeze(0)
    formula1, inputs1 = eventually_goal1_then_eventually_goal2(x, x.shape[1]-1)
    formula2, inputs2 = always_safe(x)
    formula = formula1 & formula2
    inputs = (inputs1, inputs2)
    pscale = 1
    scale  = -1
    rho = formula.robustness(inputs, pscale=pscale, scale=scale)
    return rho

print("STLCG started")
start_time = time.time()
for i in tqdm(range(Epochs)):
    trajectory = torch.randn( T+1, 3 ).to(device)
    objective_value = robustness(trajectory)

end_time = time.time()

print(end_time-start_time)