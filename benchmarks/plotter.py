import torch
load_path = 'results/'

_ , robustness_time_stlcgpp = torch.load(load_path + 'STLCGpp_GPU_Batched.pt')
_ , robustness_time_lb4tl   = torch.load(load_path + 'LB4TL_GPU_Batched.pt'  )

import matplotlib.pyplot as plt

x_ticks = list(range(5*(1+1), 5*(30+1+1), 5)) + list(range(5*(35+1) , 5*(50+1+1), 25)) 


plt.plot(x_ticks, robustness_time_stlcgpp[::-1], label='STLCGpp Robustness Time')
plt.plot(x_ticks, robustness_time_lb4tl[::-1], label='LB4TL Robustness Time')
plt.xlabel('Formula Size')
plt.ylabel('Time (seconds)')

plt.title('Robustness Time Comparison')
plt.legend()
plt.grid()
plt.show()


################################################

Times_GPU_stlcgpp, Epochs_GPU_stlcgpp = torch.load(load_path + 'STLCGpp_GPU_training_time.pt')
Times_GPU_Lb4TL, Epochs_GPU_Lb4TL     = torch.load(load_path + 'Lb4TL_GPU_training_time.pt')

Times_GPU_stlcgpp = torch.tensor(Times_GPU_stlcgpp, dtype=torch.float32)
Times_GPU_Lb4TL = torch.tensor(Times_GPU_Lb4TL, dtype=torch.float32)

# Compute speed ratio
speed_ratio = Times_GPU_stlcgpp / Times_GPU_Lb4TL

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1,len(speed_ratio)+1 , 1), speed_ratio, marker='o', linestyle='-', color='blue')
plt.axhline(y=1, color='red', linestyle='--', label='y = 1 (Equal Time)')
plt.xticks(range(1, 21))
plt.xlabel('Seed Number')
plt.ylabel('Speed Ratio (Speed of LB4TL / speed of STLCGpp)')
plt.title('Training Time Speed Ratio (GPU)')
plt.grid(True)
plt.tight_layout()
plt.show()

Epochs_GPU_stlcgpp = torch.tensor(Epochs_GPU_stlcgpp, dtype=torch.float32)
Epochs_GPU_Lb4TL = torch.tensor(Epochs_GPU_Lb4TL, dtype=torch.float32)

# Compute speed ratio
Gradient_step_ratio = Epochs_GPU_stlcgpp / Epochs_GPU_Lb4TL

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1,len(speed_ratio)+1 , 1), Gradient_step_ratio, marker='o', linestyle='-', color='blue')
plt.axhline(y=1, color='red', linestyle='--', label='y = 1 (Equal Time)')
plt.xticks(range(1, 21))
plt.xlabel('Seed Number')
plt.ylabel('Speed Ratio (Gradient-steps in STLCGpp / Gradient-steps in LB4TL)')
plt.title('The Ratio between the number of Gradient steps (GPU)')
plt.grid(True)
plt.tight_layout()
plt.show()



######################################################

Times_CPU_stlcgpp, Epochs_CPU_stlcgpp = torch.load(load_path + 'STLCGpp_CPU_training_time.pt')
Times_CPU_Lb4TL, Epochs_CPU_Lb4TL     = torch.load(load_path + 'Lb4TL_CPU_training_time.pt')

Times_CPU_stlcgpp = torch.tensor(Times_CPU_stlcgpp, dtype=torch.float32)
Times_CPU_Lb4TL = torch.tensor(Times_CPU_Lb4TL, dtype=torch.float32)

# Compute speed ratio
speed_ratio = Times_CPU_stlcgpp / Times_CPU_Lb4TL

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1,len(speed_ratio)+1 , 1), speed_ratio, marker='o', linestyle='-', color='blue')
plt.axhline(y=1, color='red', linestyle='--', label='y = 1 (Equal Time)')
plt.xticks(range(1, 101), rotation=90, fontsize=6)
plt.xlabel('Seed Number')
plt.ylabel('Speed Ratio (Speed of LB4TL / speed of STLCGpp)')
plt.title('Training Time Speed Ratio (CPU)')
plt.grid(True)
plt.tight_layout()
plt.show()

Epochs_CPU_stlcgpp = torch.tensor(Epochs_CPU_stlcgpp, dtype=torch.float32)
Epochs_CPU_Lb4TL = torch.tensor(Epochs_CPU_Lb4TL, dtype=torch.float32)

# Compute speed ratio
Gradient_step_ratio = Epochs_CPU_stlcgpp / Epochs_CPU_Lb4TL

# Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1,len(speed_ratio)+1 , 1), Gradient_step_ratio, marker='o', linestyle='-', color='blue')
plt.axhline(y=1, color='red', linestyle='--', label='y = 1 (Equal Time)')
plt.xticks(range(1, 101), rotation=90, fontsize=6)
plt.xlabel('Seed Number')
plt.ylabel('Speed Ratio (Gradient-steps in STLCGpp / Gradient-steps in LB4TL)')
plt.title('The Ratio between the number of Gradient steps (CPU)')
plt.grid(True)
plt.tight_layout()
plt.show()