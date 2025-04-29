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
    
    if T % 35 != 0:
        raise ValueError("T must be divisible by 35.")
    
    scale = T // 35

    # Initialize M (constraints) and c (offsets)
    M = torch.zeros((16, 2))  # 16 predicates, 2-dimensional state
    c = torch.zeros((16, 1))  # 16 scalar thresholds

    # Fill M and c according to predicate_transformation
    M[0, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[0, 0] = torch.tensor([4], dtype=torch.float32)

    M[1, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[1, 0] = torch.tensor([-2], dtype=torch.float32)

    M[2, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[2, 0] = torch.tensor([-0.2], dtype=torch.float32)

    M[3, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[3, 0] = torch.tensor([0.8], dtype=torch.float32)

    M[4, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[4, 0] = torch.tensor([10], dtype=torch.float32)

    M[5, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[5, 0] = torch.tensor([-7], dtype=torch.float32)

    M[6, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[6, 0] = torch.tensor([-0.2], dtype=torch.float32)

    M[7, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[7, 0] = torch.tensor([0.8], dtype=torch.float32)

    M[8, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[8, 0] = torch.tensor([24], dtype=torch.float32)

    M[9, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[9, 0] = torch.tensor([-14], dtype=torch.float32)

    M[10, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[10, 0] = torch.tensor([-0.2], dtype=torch.float32)

    M[11, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[11, 0] = torch.tensor([0.6], dtype=torch.float32)

    M[12, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[12, 0] = torch.tensor([75], dtype=torch.float32)

    M[13, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[13, 0] = torch.tensor([-60], dtype=torch.float32)

    M[14, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[14, 0] = torch.tensor([0.6], dtype=torch.float32)

    M[15, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[15, 0] = torch.tensor([-0.2], dtype=torch.float32)

    # Define predicates
    p1 = FF.LinearPredicate(M[0, :], c[0, 0])
    p2 = FF.LinearPredicate(M[1, :], c[1, 0])
    p3 = FF.LinearPredicate(M[2, :], c[2, 0])
    p4 = FF.LinearPredicate(M[3, :], c[3, 0])

    p5 = FF.LinearPredicate(M[4, :], c[4, 0])
    p6 = FF.LinearPredicate(M[5, :], c[5, 0])
    p7 = FF.LinearPredicate(M[6, :], c[6, 0])
    p8 = FF.LinearPredicate(M[7, :], c[7, 0])
    
    p9 = FF.LinearPredicate(M[8, :], c[8, 0])
    p10 = FF.LinearPredicate(M[9, :], c[9, 0])
    p11 = FF.LinearPredicate(M[10, :], c[10, 0])
    p12 = FF.LinearPredicate(M[11, :], c[11, 0])

    p13 = FF.LinearPredicate(M[12, :], c[12, 0])
    p14 = FF.LinearPredicate(M[13, :], c[13, 0])
    p15 = FF.LinearPredicate(M[14, :], c[14, 0])
    p16 = FF.LinearPredicate(M[15, :], c[15, 0])

    # Build conjunctions
    P1 = FF.And([p3, p4, p2, p1])
    P2 = FF.And([p7, p8, p6, p5])
    P3 = FF.And([p11, p12, p10, p9])
    P4 = FF.And([p15, p16, p14, p13])

    # Build temporal structure step-by-step
    first = FF.F(P4, 8*scale, 9*scale)
    second = FF.And([first, P3])
    third = FF.F(second, 6*scale, 7*scale)
    fourth = FF.And([third, P2])
    fifth = FF.F(fourth, 6*scale, 11*scale)
    sixth = FF.And([fifth, P1])
    seventh = FF.F(sixth, 5*scale, 8*scale)
    
    # Final formula
    my_formula = seventh
    
    return my_formula


def get_robustness_function(T, approximate=False, beta=10.0, apply_JIT = False, device=None, bs = 10):
    args = {'T': T+1, 'd_state': 2, 'Batch': 1, 'approximation_beta': beta, 'device': device, 'detailed_str_mode': False}
    specification = generate_formula(args)
    sample_trajectory = torch.randn(bs, T+1, 2).to(device)
    robustness_function = generate_network(specification, approximate=approximate, beta=beta, sparse=True).to(device)
    robustness_function.eval()
    if apply_JIT:
        robustness_function = torch.jit.trace(robustness_function, (sample_trajectory))
    return robustness_function