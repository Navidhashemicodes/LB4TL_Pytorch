import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import time
import sys
from tqdm.auto import tqdm
import pathlib

# method = 'LB4TL_wide'
method = 'STLCGPP_wide'
device = "cpu" if torch.cuda.is_available() else "cpu"
epochs =100
bs = 1
apply_JIT = True

if method == 'LB4TL_wide':
    import lb4tl_formula_wide
    get_robustness_function = lb4tl_formula_wide.get_robustness_function
elif method == 'STLCGPP_wide':
    import stlcgpp_formula_wide
    get_robustness_function = stlcgpp_formula_wide.get_robustness_function
else:
    raise ValueError("Invalid method. Choose either 'LB4TL' or 'STLCGPP'.")

ROBUSTNESS_TIMES = []

for i in tqdm(list(range(7, 0, -1))):
    
    T = 35*(i+1)
    print(f"Testing T = {T}")
    exact_robust_function = get_robustness_function(T, approximate=False, beta=1.0, apply_JIT=apply_JIT, device=device, bs = bs)
    
    times = []
    for _ in tqdm(range(epochs)):
        trajectory = torch.randn( bs, T+1, 2).to(device)
        begin_time = time.perf_counter()
        objective_value1 = exact_robust_function(trajectory)
        end_time = time.perf_counter()
        times.append((end_time - begin_time) / bs)
    ROBUSTNESS_TIMES.append(np.mean(times))
    print(f"Robustness time: {ROBUSTNESS_TIMES[-1]:.10f} seconds")
    
import matplotlib.pyplot as plt   
plt.plot(ROBUSTNESS_TIMES, label='Robustness Time')
plt.show()

import os
save_path = 'results/'
os.makedirs(save_path, exist_ok=True)
torch.save(ROBUSTNESS_TIMES, save_path + f'{method}_jit_{apply_JIT}_device_{device}_bs_{bs}_epochs_{epochs}2.pt')