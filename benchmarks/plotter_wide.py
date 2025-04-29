import torch
load_path = 'results/'

robustness_time_lb4tl   = torch.load(load_path + 'LB4TL_wide_jit_False_device_cpu_bs_1_epochs_1000.pt')
robustness_time_stlcgpp = torch.load(load_path + 'STLCGPP_wide_jit_False_device_cpu_bs_1_epochs_1000.pt'  )
robustness_time_stlcgppjit = torch.load(load_path + 'STLCGPP_wide_jit_True_device_cpu_bs_1_epochs_1000.pt'  )


import matplotlib.pyplot as plt

x_ticks = list(range(35*(0+1) , 35*(4+1+1) , 35)) 


plt.plot(x_ticks, robustness_time_stlcgpp, label='STLCGpp Robustness Time')
plt.plot(x_ticks, robustness_time_stlcgppjit, label='STLCGpp(JIT) Robustness Time')
plt.plot(x_ticks, robustness_time_lb4tl, label='LB4TL Robustness Time')

plt.xlabel('Formula Size')
plt.ylabel('Time (seconds)')

plt.title('Robustness Time Comparison')
plt.legend()
plt.grid()
plt.show()




# robustness_time_lb4tl   = torch.load(load_path + 'LB4TL_wide_jit_False_device_cuda_bs_3000_epochs_100.pt')
# robustness_time_stlcgpp = torch.load(load_path + 'STLCGPP_wide_jit_False_device_cuda_bs_3000_epochs_100.pt'  )


# import matplotlib.pyplot as plt

# x_ticks = list(range(35*(0+1) , 35*(4+1+1) , 35)) 


# plt.plot(x_ticks, robustness_time_stlcgpp, label='STLCGpp Robustness Time')
# plt.plot(x_ticks, robustness_time_lb4tl, label='LB4TL Robustness Time')

# plt.xlabel('Formula Size')
# plt.ylabel('Time (seconds)')

# plt.title('Robustness Time Comparison')
# plt.legend()
# plt.grid()
# plt.show()
# # from scipy.io import savemat
# # import numpy as np
# # import os
# # save_path = 'results/'
# # os.makedirs(save_path, exist_ok=True)