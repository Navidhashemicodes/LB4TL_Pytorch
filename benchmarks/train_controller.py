import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import time
import sys
import pathlib
import random
from tqdm.auto import tqdm


method = 'STLCG'
device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 100000
bs = 3
T = 40
apply_JIT = False
beta = 10.0


if method == 'LB4TL':
    import lb4tl_formula
    get_robustness_function = lb4tl_formula.get_robustness_function
elif method == 'STLCGPP':
    import stlcgpp_formula
    get_robustness_function = stlcgpp_formula.get_robustness_function
elif method == 'STLCG':
    import stlcg_formula
    get_robustness_function = stlcg_formula.get_robustness_function
elif method == 'SOP':
    import sop_formula
    get_robustness_function = sop_formula.get_robustness_function
elif method == 'EF':
    import ef_formula
    get_robustness_function = ef_formula.get_robustness_function
else:
    raise ValueError("Invalid method. Choose either 'LB4TL' or 'STLCGPP'.")



def model(s, a):

    dt = 0.05
    L = 1

    v = 2.5 * torch.tanh(0.5 * a[:, 0]) + 2.5
    gam = (torch.pi / 4) * torch.tanh(a[:, 1])

    f1 = s[:, 0] + (L / torch.tan(gam)) * (torch.sin(s[:, 2] + (v / L) * torch.tan(gam) * dt) - torch.sin(s[:, 2]))
    f2 = s[:, 1] + (L / torch.tan(gam)) * (-torch.cos(s[:, 2] + (v / L) * torch.tan(gam) * dt) + torch.cos(s[:, 2]))
    f3 = s[:, 2] + (v / L) * torch.tan(gam) * dt

    s_next = torch.stack([f1, f2, f3], dim = -1)

    return s_next

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_trajectory(initial_state, env_model, controller_net, T, bs):
    trajectory = []
    trajectory.append(initial_state)
    state = initial_state
    for t in range(0, T):
        Time = torch.zeros([bs, 1], dtype=torch.float32) + t
        Time = Time.to(device)
        sa = torch.cat([state, Time], dim=1)
        state = env_model( state, controller_net(sa) )
        trajectory.append(state)
    return torch.stack(trajectory, dim=1)



Time = []
Epoch = []
Net = []
seeds = 20
for seed in tqdm(range(seeds)):
    seed_everything(seed)
    
    controller_hidden_size = 30
    controller_net = nn.Sequential(
        nn.Linear(4, controller_hidden_size),
        nn.ReLU(),
        nn.Linear(controller_hidden_size, 2)
    ).to(device)

    exact_robust_function = get_robustness_function(T, approximate=False, beta=beta, apply_JIT=apply_JIT, device=device, bs=bs)
    approximate_robust_function = get_robustness_function(T, approximate=True, beta=beta, apply_JIT=apply_JIT, device=device, bs=bs)

    # Define the optimizer for f_network
    optimizer = optim.Adam(controller_net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

    max_time_seconds = 600
    # Record the start time
    start_time = time.time()
    theta_set = torch.linspace(-6*torch.pi/8, -4*torch.pi/8, 10000)
    theta_candidate_set = [-6*torch.pi/8, -5*torch.pi/8, -4*torch.pi/8]
    assert len(theta_candidate_set) == 3, "theta_candidate_set should have 3 elements."
    for epoch in range(num_epochs):
        theta = torch.tensor(theta_candidate_set)
        # theta = theta[torch.randint(0, len(theta_set), (bs,))]
        
        x_init, y_init = torch.zeros(bs) + 6, torch.zeros(bs) + 8
        init_state = torch.stack([x_init, y_init, theta], dim=1).to(device)
    
        trajectory = run_trajectory(init_state, model, controller_net, T, bs)
        objective_value = approximate_robust_function(trajectory[:, :, :-1]).mean(dim=0)
    
        # Backward pass and optimization to maximize the objective function
        optimizer.zero_grad()
        (-objective_value).backward()
        optimizer.step()
        scheduler.step()

        print(f'Epoch {epoch + 1}, Objective Value: {objective_value.item()}')
    
        if epoch % 10 == 0:
        
            bs_test = 100
            theta_random = torch.tensor(theta_set[torch.randint(0, len(theta_set), (bs_test,))])
            x_init_test, y_init_test = torch.zeros(bs_test) + 6, torch.zeros(bs_test) + 8
            init_state_test = torch.stack([x_init_test, y_init_test, theta_random], dim=1).to(device)
        
            with torch.no_grad():
                trajectory_test = run_trajectory(init_state_test, model, controller_net, T, bs_test)
                exact_robustness = exact_robust_function(trajectory_test[:, :, :-1]).min(dim=0).values
                print(f'Epoch {epoch + 1}, Exact Robustness: {exact_robustness.item()} vs Approximate Robustness: {objective_value.item()}')
            if exact_robustness > 0:
                print("Found a robust solution. Breaking out of the loop.")
                break
    
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time_seconds:
            print("Time limit exceeded the threshold. Breaking out of the loop.")
            break

    elapsed_time = time.time() - start_time
    Time.append(elapsed_time)
    Epoch.append(epoch + 1)
    Net.append(controller_net)
    print("Training completed, with time = ", elapsed_time, " seconds, epochs = ", epoch + 1)
    

import os
save_path = 'results/'
os.makedirs(save_path, exist_ok=True)
torch.save([ Time , Epoch, Net ], save_path + f'{method}_jit_{apply_JIT}_beta_{beta}_device_{device}_seeds_{seeds}.pt')