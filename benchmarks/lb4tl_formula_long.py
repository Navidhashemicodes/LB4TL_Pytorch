import stlcgpp.formula as stlcg
import numpy as np
import torch
import torch.nn as nn
import sys
import pathlib
import time 
from tqdm.auto import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


from networks.neural_net_generator import generate_network
from formula_factory import FormulaFactory


def generate_formula(args):
    
    T = args['T'] - 1
    FF = FormulaFactory(args)
    
    if T % 20 != 0:
        raise ValueError("T must be divisible by 35.")
    
    portion = T // 20

    # Initialize M (constraints) and c (offsets)
    M = torch.zeros((4, 2))  # 16 predicates, 2-dimensional state
    c = torch.zeros((4, 1))  # 16 scalar thresholds

    # Fill M and c according to predicate_transformation
    M[0, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[0, 0] = torch.tensor([1], dtype=torch.float32)

    M[1, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[1, 0] = torch.tensor([1], dtype=torch.float32)

    M[2, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[2, 0] = torch.tensor([1], dtype=torch.float32)

    M[3, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[3, 0] = torch.tensor([1], dtype=torch.float32)

    

    # Define predicates
    p1 = FF.LinearPredicate(M[0, :], c[0, 0])
    p2 = FF.LinearPredicate(M[1, :], c[1, 0])
    p3 = FF.LinearPredicate(M[2, :], c[2, 0])
    p4 = FF.LinearPredicate(M[3, :], c[3, 0])

    P0 = FF.And([p3, p4, p2, p1])
    
    Evs = FF.F(P0 , 0 , portion)
    for i in range(1, 20):
        Evs = FF.And( [Evs , FF.F( P0, i*portion , (i+1)*portion ) ] )
    
    return Evs


def get_robustness_function(T, approximate=False, beta=10.0, apply_JIT = False, device=None, bs = 10):
    args = {'T': T+1, 'd_state': 2, 'Batch': 1, 'approximation_beta': beta, 'device': device, 'detailed_str_mode': False}
    specification = generate_formula(args)
    sample_trajectory = torch.randn(bs, T+1, 2).to(device)
    robustness_function = generate_network(specification, approximate=approximate, beta=beta, sparse=True).to(device)
    robustness_function.eval()
    if apply_JIT:
        robustness_function = torch.jit.trace(robustness_function, (sample_trajectory))
    return robustness_function