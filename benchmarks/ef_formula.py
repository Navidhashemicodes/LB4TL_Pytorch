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
        res = torch.cat([self.func(x[i]) for i in range(self.bs)], dim=0)
        return res

def boltzmann(x, beta, exact):
    if not exact:    
        m = nn.Softmax(dim=0)
        y = m(beta * x)
        z = x * y
        return torch.sum(z).flatten()
    else:
        return -softmin(-x, beta, exact)

def softmin(x, beta, exact):
    if not exact:

        result = - torch.logsumexp(-beta * x, dim=0) / beta
        return result.flatten()
    else:
        return torch.min(x, dim=0)[0].flatten()

def h_g1(s, beta, exact):
    hs = torch.zeros(4)
    hs[0] = 1 * 6 - s[0]
    hs[1] = s[0] - 5
    hs[2] = s[1] - 3
    hs[3] = 4 - s[1]
    return softmin(hs, beta, exact)

def h_g2(s, beta, exact):
    hs = torch.zeros(4)
    hs[0] = 4 - s[0]
    hs[1] = s[0] - 3
    hs[2] = s[1]
    hs[3] = 1 - s[1]
    return softmin(hs, beta, exact)

def h_g3(s, beta, exact):
    hs = torch.zeros(4)
    hs[0] = 1 - s[0]
    hs[1] = s[0] - 4
    hs[2] = s[1] - 5
    hs[3] = 2 - s[1]
    return boltzmann(hs, beta, exact)

def robustness(x, beta, exact, T_):
    T = T_ + 1 # 21
    rob0 = torch.zeros(T - 1)
    for i in range(1, T):
        
        
        rob1 = h_g1(x[i-1, :], beta=beta, exact=exact)
        B1 = rob1
        
        rob2 = torch.zeros(T-i) # 20
        for j in range(i, T): # 20
            rob2[j-i] = h_g2(x[j, :], beta=beta, exact=exact) # 0 to 19
        B2 = boltzmann(rob2, beta=beta, exact=exact)
        
        rob0[i - 1] = softmin(torch.cat([B1, B2]), beta=beta, exact=exact)
    
    Rob1 = boltzmann(rob0, beta=beta, exact=exact)
    
    rob0 = torch.zeros(T)
    for i in range(0, T):
        rob0[i] = h_g3(x[i, :], beta=beta, exact=exact)
    
    Rob2 = softmin(rob0, beta=beta, exact=exact)
    
    rho = softmin(torch.cat([Rob1, Rob2]), beta=beta, exact=exact)
   
    return rho

from functools import partial
def get_robustness_function(T, approximate=False, beta=10.0, apply_JIT = False, device=None, bs = 10):
    f = partial(robustness, beta = beta, exact= not approximate, T_=T)
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
    