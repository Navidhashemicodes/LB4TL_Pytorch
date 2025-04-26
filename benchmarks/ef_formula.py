import stlcgpp.formula as stlcg
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm
import stlcg.stlcg as stlcg



class RobustnessModule(torch.nn.Module):
    def __init__(self, func):
        super(RobustnessModule, self).__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)
        


def boltzmann(x, beta, exact):
    if not exact:    
        m = nn.Softmax(dim=1)
        y = m(beta * x)
        z = x * y
        return torch.sum(z, dim = 1)
    else:
        return -softmin(-x, beta, exact)


def softmax(x, beta, exact):
    return boltzmann(x, beta, exact)

def softmin(x, beta, exact):
    if not exact:

        result = - torch.logsumexp(-beta * x, dim=1) / beta
        return result
    else:
        return torch.min(x, dim=1)[0]

def h_g1(s, beta, exact, bs):
    hs = torch.zeros(bs, 4, device=s.device)
    hs[:, 0] = 1 * 6 - s[:, 0]
    hs[:, 1] = s[:, 0] - 5
    hs[:, 2] = s[:, 1] - 3
    hs[:, 3] = 4 - s[:, 1]
    return softmin(hs, beta, exact)

def h_g2(s, beta, exact, bs):
    hs = torch.zeros(bs, 4, device=s.device)
    hs[:, 0] = 4 - s[:, 0]
    hs[:, 1] = s[:, 0] - 3
    hs[:, 2] = s[:, 1]
    hs[:, 3] = 1 - s[:, 1]
    return softmin(hs, beta, exact)

def h_g3(s, beta, exact, bs):
    hs = torch.zeros(bs, 4, device=s.device)
    hs[:, 0] = 1 - s[:, 0]
    hs[:, 1] = s[:, 0] - 4
    hs[:, 2] = s[:, 1] - 5
    hs[:, 3] = 2 - s[:, 1]
    return softmax(hs, beta, exact)

def robustness(x, beta, exact, T_, bs):
    T = T_ + 1 # 21
    rob0 = torch.zeros(bs, T - 1, device=x.device)
    for i in range(1, T):
        
        rob1 = h_g1(x[:, i-1, :], beta=beta, exact=exact, bs = bs)
        B1 = rob1
        
        rob2 = torch.zeros(bs, T-i, device=x.device)
        for j in range(i, T): # 20
            rob2[:, j-i] = h_g2(x[:, j, :], beta=beta, exact=exact, bs = bs) # 0 to 19
        B2 = softmax(rob2, beta=beta, exact=exact)
        
        rob0[:, i - 1] = softmin(torch.stack([B1, B2], dim = 1), beta=beta, exact=exact)
    
    Rob1 = softmax(rob0, beta=beta, exact=exact)
    
    rob0 = torch.zeros(bs, T, device=x.device)
    for i in range(0, T):
        rob0[:, i] = h_g3(x[:, i, :], beta=beta, exact=exact, bs = bs)
    
    Rob2 = softmin(rob0, beta=beta, exact=exact)
    
    rho = softmin(torch.stack([Rob1, Rob2], dim = 1), beta=beta, exact=exact)
   
    return rho

from functools import partial
def get_robustness_function(T, approximate=False, beta=10.0, apply_JIT = False, device=None, bs = 10):
    f = partial(robustness, beta = beta, exact= not approximate, T_=T, bs = bs)
    rm = RobustnessModule(f).to(device)
    
    if apply_JIT:
        sample_trajectory = torch.randn((bs, T+1, 2)).to(device)
        rm = torch.jit.trace(rm, sample_trajectory).to(device)
    
    return rm
    
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 20
    bs = 10
    epochs = 100
    trajectory = torch.randn( bs, T+1, 2).to(device)
    apply_JIT = False
    
    rf_without_jit = get_robustness_function(T, approximate=False, beta=10, apply_JIT=apply_JIT, device=device, bs=bs)
    
    start = time.perf_counter()
    for i in tqdm(range(epochs)):
        v2 = rf_without_jit(trajectory)
    end = time.perf_counter()
    print("Time taken without JIT trace: ", end - start)
    

    rf_with_jit = get_robustness_function(T, approximate=False, beta=10, apply_JIT=True, device=device, bs=bs)
    start = time.perf_counter()
    for i in tqdm(range(epochs)):
        v1 = rf_with_jit(trajectory)
    end = time.perf_counter()
    print("Time taken with JIT trace: ", end - start)
    
    
    print("Diff:", torch.max(torch.abs(v1 - v2)))
    