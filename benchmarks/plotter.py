import torch
load_path = 'results/'

build_time_stlcgpp, robustness_time_stlcgpp= torch.load(load_path + 'STLCGpp_GPU_Batched.pt')
build_time_lb4tl, robustness_time_lb4tl = torch.load(load_path + 'LB4TL_GPU_Batched.pt')

import matplotlib.pyplot as plt

plt.plot(build_time_stlcgpp, label='STLCGpp Formula Building Time')
plt.plot(build_time_lb4tl, label='LB4TL Formula Building Time')
plt.xlabel('Formula Size')
plt.ylabel('Time (seconds)')

plt.title('Formula Building Time Comparison')
plt.legend()
plt.grid()
plt.show()

plt.plot(robustness_time_stlcgpp, label='STLCGpp Robustness Time')
plt.plot(robustness_time_lb4tl, label='LB4TL Robustness Time')
plt.xlabel('Formula Size')
plt.ylabel('Time (seconds)')

plt.title('Robustness Time Comparison')
plt.legend()
plt.grid()
plt.show()

