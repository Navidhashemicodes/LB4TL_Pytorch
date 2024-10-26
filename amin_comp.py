import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import time
import sys
import pathlib
from networks.neural_net_generator import generate_network

RP_dir = str (pathlib.Path().resolve())
sys.path.append( RP_dir )

from formula_factory import FormulaFactory

FOLDERNAME = RP_dir



###################################Accurate

class RNetwork_acc(nn.Module):
    def __init__(self, weights_file, biases_file):
        super(RNetwork_acc, self).__init__()

        # Load weights and biases from MATLAB files
        weights_Linear_mat = loadmat(weights_file)['WL']
        biases_Linear_mat = loadmat(biases_file)['BL']
        weights_nLinear_mat = loadmat(weights_file)['Wn']
        biases_nLinear_mat = loadmat(biases_file)['Bn']

        # Convert cell array elements to PyTorch tensors with type conversion
        self.weights_L = [torch.from_numpy(wi.astype(np.float32)) for wi in weights_Linear_mat[0]]
        self.weights_n = [torch.from_numpy(wi.astype(np.float32)) for wi in weights_nLinear_mat[0]]
        self.biases_L = [torch.from_numpy(bi.astype(np.float32)) for bi in biases_Linear_mat[0]]
        self.biases_n = [torch.from_numpy(bi.astype(np.float32)) for bi in biases_nLinear_mat[0]]

        # Calculate the number of hidden layers based on the length of weights
        num_hidden_layers = len(self.weights_L) - 1

        # Define the hidden layers using predefined weights and biases
        self.hidden_layers = nn.ModuleList([
            nn.Linear( (self.weights_n[i]).shape[1], (self.weights_n[i]).shape[0])
            for i in range(num_hidden_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear((self.weights_L[-1]).shape[1], (self.weights_L[-1]).shape[0])
        
        
        for i, layer in enumerate(self.hidden_layers):
            layer.weight.data = self.weights_n[i]  # Transpose for proper shape
            layer.bias.data = self.biases_n[i]

        self.output_layer.weight.data = self.weights_L[-1]
        self.output_layer.bias.data = self.biases_L[-1]

    def forward(self, x):
        # Forward pass through hidden layers with ReLU activation
        for i, layer in enumerate(self.hidden_layers):
            xn = nn.functional.relu(layer(x))
            xl = torch.matmul( x , (self.weights_L[i]).t()  ) + self.biases_L[i]
            x=[]
            x.append(xl)
            x.append(xn)
            x = torch.cat(x , dim=1) 
            
        # Output layer
        x = self.output_layer(x)
        return x

T = 40
Batch = 1
device = torch.device("cpu")
args = {'T': T+1, 'd_state': 3, 'Batch': Batch, 'approximation_beta': 1, 'device': device, 'detailed_str_mode': False}
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


# or_list = []
# for i in range(0,T):
#     or_list.append( FF.And( [ FF.F(and1, i, i) , FF.F( and2, i+1 , T) ] ) )
                 
# ordered = FF.Or(or_list)

ordered = FF.Ordered(and1 , and2 , 0 , T)

SFE = FF.G(  or1,   0,   T  )


my_formula  =  FF.And( [ ordered , SFE ] )




weights_file_path_acc = FOLDERNAME + '/weights/weights_acc.mat'
biases_file_path_acc = FOLDERNAME + '/weights/biases_acc.mat'

r_network_acc = RNetwork_acc(weights_file_path_acc, biases_file_path_acc)
neural_net = generate_network(my_formula, approximate=True, beta=10).to(args['device'])
    

start_time =time.time()
for i in range(10):
    
    
    trajectory = torch.randn( Batch, T+1, 3 ).to(device)
    # objective_value1 =  my_formula.evaluate(trajectory)[0]
    objective_value1 = neural_net(trajectory)
    print(objective_value1)
    trajectory_check =  trajectory.reshape(1,-1)
    objective_value2 = r_network_acc(trajectory_check)
    print(objective_value2)
    print('#############')

end_time = time.time()

print(end_time-start_time)