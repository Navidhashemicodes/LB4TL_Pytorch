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

    M = torch.zeros((12, 2))  # 12 rows, 2 columns
    c = torch.zeros((12, 1))  # 12 rows, 1 column

    M[0, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[0, 0] = torch.tensor([1], dtype=torch.float32)

    M[1, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[1, 0] = torch.tensor([-4], dtype=torch.float32)

    M[2, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[2, 0] = torch.tensor([-5], dtype=torch.float32)

    M[3, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[3, 0] = torch.tensor([2], dtype=torch.float32)

    M[4, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[4, 0] = torch.tensor([-3], dtype=torch.float32)

    M[5, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[5, 0] = torch.tensor([4], dtype=torch.float32)

    M[6, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[6, 0] = torch.tensor([0], dtype=torch.float32)

    M[7, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[7, 0] = torch.tensor([1], dtype=torch.float32)

    M[8, :] = torch.tensor([1, 0], dtype=torch.float32)
    c[8, 0] = torch.tensor([-5], dtype=torch.float32)

    M[9, :] = torch.tensor([-1, 0], dtype=torch.float32)
    c[9, 0] = torch.tensor([6], dtype=torch.float32)

    M[10, :] = torch.tensor([0, 1], dtype=torch.float32)
    c[10, 0] = torch.tensor([-3], dtype=torch.float32)

    M[11, :] = torch.tensor([0, -1], dtype=torch.float32)
    c[11, 0] = torch.tensor([4], dtype=torch.float32)


    p1 = FF.LinearPredicate(M[0,:], c[0,0])
    p2 = FF.LinearPredicate(M[1,:], c[1,0])
    p3 = FF.LinearPredicate(M[2,:], c[2,0])
    p4 = FF.LinearPredicate(M[3,:], c[3,0])
    p5 = FF.LinearPredicate(M[4,:], c[4,0])
    p6 = FF.LinearPredicate(M[5,:], c[5,0])
    p7 = FF.LinearPredicate(M[6,:], c[6,0])
    p8 = FF.LinearPredicate(M[7,:], c[7,0])
    p9 = FF.LinearPredicate(M[8,:], c[8,0])
    p10 = FF.LinearPredicate(M[9,:], c[9,0])
    p11 = FF.LinearPredicate(M[10,:], c[10,0])
    p12 = FF.LinearPredicate(M[11,:], c[11,0])


    or1  = FF.Or([p1,p2,p3,p4])
    and2 = FF.And([p5,p6,p7,p8])
    and1 = FF.And([p9,p10,p11,p12])


    ordered = FF.Ordered(and1 , and2 , 0 , T)

    SFE = FF.G(  or1,   0,   T  )


    my_formula  =  FF.And( [ ordered , SFE ] )
    
    return my_formula


def get_robustness_function(T, approximate=False, beta=10.0, apply_JIT = False, device=None):
    args = {'T': T+1, 'd_state': 2, 'Batch': 1, 'approximation_beta': beta, 'device': device, 'detailed_str_mode': False}
    specification = generate_formula(args)
    sample_trajectory = torch.randn(1, T+1, 2).to(device)
    robustness_function = generate_network(specification, approximate=approximate, beta=beta, sparse=True).to(device)
    robustness_function.eval()
    if apply_JIT:
        robustness_function = torch.jit.trace(robustness_function, (sample_trajectory))
    return robustness_function
    
if __name__ == "__main__":
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    T = 20
    bs = 100
    epochs = 1000
    trajectory = torch.randn( bs, T+1, 2).to(device)
    apply_JIT = False
    rf_with_jit = get_robustness_function(T, approximate=False, beta=10, apply_JIT=True, device=device)
    rf_without_jit = get_robustness_function(T, approximate=False, beta=10, apply_JIT=False, device=device)
    
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
    