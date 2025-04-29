import torch
load_path = 'results/'

robustness_time_lb4tl   = torch.load(load_path + 'LB4TL_long_jit_False_device_cpu_bs_1_epochs_1000.pt')
robustness_time_stlcgpp = torch.load(load_path + 'STLCGPP_long_jit_False_device_cpu_bs_1_epochs_1000.pt'  )
robustness_time_stlcgppjit = torch.load(load_path + 'STLCGPP_long_jit_True_device_cpu_bs_1_epochs_1000.pt'  )


import matplotlib.pyplot as plt

x_ticks = list(range(20*(1+1) , 20*(6+1+1) , 20)) 


plt.plot(x_ticks, robustness_time_stlcgpp[::-1], label='STLCGpp Robustness Time')
plt.plot(x_ticks, robustness_time_stlcgppjit[::-1], label='STLCGpp(JIT) Robustness Time')
plt.plot(x_ticks, robustness_time_lb4tl[::-1], label='LB4TL Robustness Time')

plt.xlabel('Formula Size')
plt.ylabel('Time (seconds)')

plt.title('Robustness Time Comparison')
plt.legend()
plt.grid()
plt.show()

# from scipy.io import savemat
# import numpy as np
# import os
# save_path = 'results/'
# os.makedirs(save_path, exist_ok=True)