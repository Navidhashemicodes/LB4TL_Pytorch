import torch
import torch.nn as nn
import lb4tl_formula

t,e,n = torch.load( 'results/confirmed/STLCGPP_jit_True_beta_10.0_device_cpu_seeds_20.pt')


method = 'STLCGPP'

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
    
    
device = "cpu" if torch.cuda.is_available() else "cpu"
num_epochs = 100000
bs = 3
bs_test = 100
T = 40
apply_JIT = True
beta = 10.0




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



exact_robust_function = get_robustness_function(T, approximate=False, beta=beta, apply_JIT=apply_JIT, device=device, bs=bs_test)
STL2NN  = lb4tl_formula.get_robustness_function(T, approximate=False, beta=beta, apply_JIT=False, device=device, bs=bs_test)

theta_set = torch.linspace(-6*torch.pi/8, -4*torch.pi/8, 10000)

for idx, net in enumerate(n):    

    controller_net = net.to(device)
    controller_net.eval() 
    
    bs_test = 100
    theta_random = torch.tensor(theta_set[torch.randint(0, len(theta_set), (bs_test,))])
    x_init_test, y_init_test = torch.zeros(bs_test) + 6, torch.zeros(bs_test) + 8
    init_state_test = torch.stack([x_init_test, y_init_test, theta_random], dim=1).to(device)
    
    with torch.no_grad():
        trajectory_test = run_trajectory(init_state_test, model, controller_net, T, bs_test)
        exact_robustness1 = exact_robust_function(trajectory_test[:, :, :-1]).min(dim=0).values
        exact_robustness2 =                STL2NN(trajectory_test[:, :, :-1]).min(dim=0).values
    
    print(exact_robustness1)
    print(exact_robustness2)
    print('#####################')