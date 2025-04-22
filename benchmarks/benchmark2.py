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

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


from networks.neural_net_generator import generate_network
from formula_factory import FormulaFactory

###################################Accurate


def generate_formula(args):
    
    FF = FormulaFactory(args)


    M = torch.zeros((12, 3))  # 12 rows, 3 columns
    c = torch.zeros((12, 1))  # 12 rows, 1 column

    M[0, :] = torch.tensor([-1, 0, 0], dtype=torch.float32)
    c[0, 0] = torch.tensor([1], dtype=torch.float32)

    M[1, :] = torch.tensor([1, 0, 0], dtype=torch.float32)
    c[1, 0] = torch.tensor([-4], dtype=torch.float32)

    M[2, :] = torch.tensor([0, 1, 0], dtype=torch.float32)
    c[2, 0] = torch.tensor([-5], dtype=torch.float32)

    M[3, :] = torch.tensor([0, -1, 0], dtype=torch.float32)
    c[3, 0] = torch.tensor([2], dtype=torch.float32)

    M[4, :] = torch.tensor([1, 0, 0], dtype=torch.float32)
    c[4, 0] = torch.tensor([-3], dtype=torch.float32)

    M[5, :] = torch.tensor([-1, 0, 0], dtype=torch.float32)
    c[5, 0] = torch.tensor([4], dtype=torch.float32)

    M[6, :] = torch.tensor([0, 1, 0], dtype=torch.float32)
    c[6, 0] = torch.tensor([0], dtype=torch.float32)

    M[7, :] = torch.tensor([0, -1, 0], dtype=torch.float32)
    c[7, 0] = torch.tensor([1], dtype=torch.float32)

    M[8, :] = torch.tensor([1, 0, 0], dtype=torch.float32)
    c[8, 0] = torch.tensor([-5], dtype=torch.float32)

    M[9, :] = torch.tensor([-1, 0, 0], dtype=torch.float32)
    c[9, 0] = torch.tensor([6], dtype=torch.float32)

    M[10, :] = torch.tensor([0, 1, 0], dtype=torch.float32)
    c[10, 0] = torch.tensor([-3], dtype=torch.float32)

    M[11, :] = torch.tensor([0, -1, 0], dtype=torch.float32)
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





Times = []
for i in range(0,20):
    
    
    T = 5*(i+1)

    device = torch.device("cuda")
    Batch = 1000
    args = {'T': T+1, 'd_state': 3, 'Batch': Batch, 'approximation_beta': 1, 'device': device, 'detailed_str_mode': False}
    
    my_formula = generate_formula(args)
    neural_net = generate_network(my_formula, approximate=False, beta=10, sparse=True).to(args['device'])

    start_time = time.perf_counter()
    
    trajectory = torch.randn( Batch, T+1, 3 ).to(device)
    objective_value1 = neural_net(trajectory)


    end_time = time.perf_counter()
    
    times =  end_time-start_time
    print(times)

    print(f"\nTotal time: {times:.4f} seconds")

    Times.append(times/Batch)
    print(Times)
    
savemat("Times_stl2nn2.mat", {"Times": np.array(Times, dtype=np.float64)})