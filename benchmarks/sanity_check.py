import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import time
import sys
import pathlib
import random
import stlcgpp.formula as stlcg

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from networks.neural_net_generator import generate_network
from formula_factory import FormulaFactory

device = "cuda" if torch.cuda.is_available() else "cpu"

bs = 10
T = 5
apply_JIT = True

approximate = False
beta = 10.0

import lb4tl_formula
get_robustness_function_lb4tl = lb4tl_formula.get_robustness_function
robustness_func_lbtl = get_robustness_function_lb4tl(T, approximate=approximate, beta=beta, apply_JIT=apply_JIT, device=device, bs=bs)

import stlcgpp_formula
get_robustness_function_stlcgpp = stlcgpp_formula.get_robustness_function
robustness_func_stlcgpp = get_robustness_function_stlcgpp(T, approximate=approximate, beta=beta, apply_JIT=apply_JIT, device=device, bs=bs)

import stlcg_formula
get_robustness_function_stlcg = stlcg_formula.get_robustness_function
robustness_func_stlcg = get_robustness_function_stlcg(T, approximate=approximate, beta=beta, apply_JIT=apply_JIT, device=device, bs=bs)

import sop_formula
get_robustness_function_sop = sop_formula.get_robustness_function
robustness_func_sop = get_robustness_function_sop(T, approximate=approximate, beta=beta, apply_JIT=apply_JIT, device=device, bs=bs)

import ef_formula
get_robustness_function_ef = ef_formula.get_robustness_function
robustness_func_ef = get_robustness_function_ef(T, approximate=approximate, beta=beta, apply_JIT=apply_JIT, device=device, bs=bs)

random_trajectory = torch.randn((bs, T+1, 2), device=device, requires_grad=True)

r1 = robustness_func_lbtl(random_trajectory)
r2 = robustness_func_stlcgpp(random_trajectory)
r3 = robustness_func_stlcg(random_trajectory)
r4 = robustness_func_sop(random_trajectory)
r5 = robustness_func_ef(random_trajectory)

print(torch.abs(r1 - r2).sum())
print(torch.abs(r1 - r3).sum())
print(torch.abs(r1 - r4).sum())
print(torch.abs(r1 - r5).sum())


