
from sklearn.preprocessing import MinMaxScaler
from numpy.typing import NDArray
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils

class BinaryTreeRNN(nn.Module):
    def __init__(self):
        super(BinaryTreeRNN, self).__init__()

        self.dataset_x = None
        self.dataset_y = None

        self.num_layers = None

        self.last_expression = None

        self.training_steps_by_standard_softmax = None

        self.training_steps_by_softmax_prime = None

        # Initialize a list to store the variable names
        self.variables = None

        self.num_variables = None
        
        # Initialize a list to store the mathematical operations that can be performed
        self.operations = ['+', '*', 'sin', 'id']

        # Get the length of the operations list
        self.operations_length = len(self.operations)

        self.first_layer_weights = None
        self.first_layer_biases = None
          
        # Initialize weights and biases for the remaining layers
        self.weights = None
        self.biases = None

        # Initialize omegas for the remaining layers
        self.omegas = None

        self.lambda_L1 = None

        self.hidden_values = None
        
        self.step_counter = None
        

 ######################### 
    def learn(self, dataset_x, dataset_y, num_layers, training_steps_by_standard_softmax, training_steps_by_softmax_prime, lr, lambda_L1):
        #x = np.array(dataset_x)
        #y = np.array(dataset_y)

        # Check if the dimensions of the arrays are correct
        if dataset_x.shape[0] == dataset_y.shape[0] and dataset_x.shape[1] > 1 and dataset_y.shape[1] == 1:
            print("datasets are okay.")

            self.dataset_x = dataset_x
            self.dataset_y = dataset_y

            self.num_variables = np.array(self.dataset_x).shape[1]

            # Initialize a list to store the variable names
            self.variables = ['x' + str(i+1) for i in range(self.num_variables)]

            self.num_layers = num_layers

            self.training_steps_by_standard_softmax = training_steps_by_standard_softmax
            self.training_steps_by_softmax_prime = training_steps_by_softmax_prime
            total_training_steps =  self.training_steps_by_standard_softmax + self.training_steps_by_softmax_prime

            # Initialize weights and biases for the first layer(Leaf)
            self.first_layer_weights = nn.Parameter(torch.randn(2**(num_layers-1), self.num_variables, dtype=torch.float32))
            self.first_layer_biases = nn.Parameter(torch.randn(2**(num_layers-1), dtype=torch.float32))
            
            # Initialize weights and biases for the remaining layers(interior)
            self.weights = nn.ParameterList([nn.Parameter(torch.randn(2**(level-1), dtype=torch.float32)) for level in range(1, self.num_layers)])
            self.biases = nn.ParameterList([nn.Parameter(torch.randn(2**(level-1), dtype=torch.float32)) for level in range(1, self.num_layers)])


            # Initialize omegas for the remaining layers
            self.omegas = nn.ParameterList([nn.Parameter(torch.randn(2**(level-1), self.operations_length, dtype=torch.float32)) for level in range(1, self.num_layers)])

            # set up the optimizer
            optimizer = optim.SGD([{'params': self.first_layer_weights},
                                    {'params': self.first_layer_biases},
                                    {'params': self.weights},
                                    {'params': self.biases},
                                    {'params': self.omegas}],
                                    lr)
            
            self.lambda_L1 = lambda_L1

            for self.step_counter in range(total_training_steps):

                optimizer.zero_grad()
                loss = 0.0
                
                for sample_x, sample_y in zip(self.dataset_x, self.dataset_y):

                    self.last_expression = [None] * (2**self.num_layers -1)
                    self.hidden_values = [None] * (2**self.num_layers -1)

                    #sample_x = torch.tensor(sample_x)

                    y_hat = self.forward(1, sample_x)

                    sample_y = torch.tensor(sample_y, dtype=y_hat.dtype)

                    # define the loss function
                    loss += nn.functional.mse_loss(y_hat, sample_y)

                if(self.step_counter%2 == 0):
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    print(self.last_expression)
                    #print(self.hidden_values)
                    print("ERROR="+str(loss))
                    #print("sample y ="+str(sample_y))
                    #print("y hat ="+str(y_hat))

                # calculate L1 regularization term
                L1_reg = torch.tensor(0., requires_grad=True)
                for param in self.parameters():
                    L1_reg = L1_reg + torch.norm(param, 1)
                L1_loss = self.lambda_L1 * L1_reg
                loss = loss + L1_loss
                

                # backward pass
                loss.backward(retain_graph=True)

                # clip gradients to avoid exploding gradients
                 # set the maximum norm to 1.0
                max_norm = 1.0
                utils.clip_grad_norm_(self.parameters(), max_norm)

                # update the parameters with gradient descent
                optimizer.step()


        else:
              print("datasets do not match. please try again.")

#########################
    def forward(self, i, sample_x):
        left_child = 2 * i
        right_child = 2 * i + 1
        layer, j = self.layer_position(i)

        if left_child >= self.num_layers ** 2 - 1:
            if right_child >= self.num_layers ** 2 - 1:
              
              # i should be the corrsponding in first layer as all the layers.

                #print("leaf: i=" + str(i) + ", layer=" + str(layer) + ", j=" + str(j))

                #define sample_x as PyTorch tensor with matching dtypes with self.first_layer_weights
                sample_x = torch.tensor(sample_x, dtype=self.first_layer_weights.dtype)

                self.hidden_values[i-1] = torch.matmul(self.first_layer_weights[j], sample_x) + self.first_layer_biases[j]
                #self.last_expression[i-1] = np.random.choice(self.variables)
                self.last_expression[i-1] = self.get_leaf(self.first_layer_weights[j], sample_x)
                return self.hidden_values[i-1]

        if(self.step_counter < self.training_steps_by_standard_softmax):
            self.hidden_values[i-1] = self.weights[layer][j] * torch.matmul(F.softmax(self.omegas[layer][j], dim=0), self.operator_p(self.forward(left_child, sample_x), self.forward(right_child, sample_x))) + self.biases[layer][j]
            self.last_expression[i-1] = self.get_operation(F.softmax(self.omegas[layer][j], dim=0), self.operator_p(self.forward(left_child, sample_x), self.forward(right_child, sample_x)))
        else:
            self.hidden_values[i-1] = self.weights[layer][j] * torch.matmul(self.softmax_prime(self.omegas[layer][j]), self.operator_p(self.forward(left_child, sample_x), self.forward(right_child, sample_x))) + self.biases[layer][j]
            self.last_expression[i-1] = self.get_operation(self.softmax_prime(self.omegas[layer][j]), self.operator_p(self.forward(left_child, sample_x), self.forward(right_child, sample_x)))
        
        
        #print("interior: i=" + str(i) + ", layer=" + str(layer) + ", j=" + str(j))
        return self.hidden_values[i-1]
#########################
    def operator_p(self, left_child_h, right_child_h):
        # initialize an empty vector to hold the results
        p = torch.zeros(len(self.operations))
        # loop over each operator in self.operations and apply it to left_child_h and right_child_h
        for i, op in enumerate(self.operations):
            if op == '+':
                p[i] = left_child_h + right_child_h
            elif op == '*':
                p[i] = left_child_h * right_child_h
            elif op == 'sin':
                p[i] = torch.sin(left_child_h + right_child_h)
            elif op == 'id':
                p[i] = left_child_h + right_child_h

        # return the resulting vector
        return p
#########################
    def softmax_prime(self, omegas):
        # Find the index of the maximum value in self.omegas[layer][j]
        max_idx = torch.argmax(omegas)
        
        # Create a tensor of zeros with the same shape as self.omegas[layer][j]
        zeros = torch.zeros_like(omegas)
        
        # Set the element at the maximum index to 1
        zeros[max_idx] = 1
        
        # Return the tensor
        return zeros
#########################    
    def layer_position(self, i):
        i_backup = i
        layer = 0
        while i > 0:
            i //= 2
            layer += 1
        layer -= 1  # Adjust for 0-based indexing

        leftmost_index = 2**layer - 1
        index_in_layer = i_backup - leftmost_index - 1

        return layer, index_in_layer
#########################
    #def get_operation(self, lst):
     #   max_index = torch.argmax(lst)
      #  return self.operations[max_index]
    def get_operation(self, s, v):
        elementwise_product = s * v
        max_index = torch.argmax(elementwise_product)
        return self.operations[max_index]

    def get_leaf(self, w, x):
        elementwise_product = w * x
        max_index = torch.argmax(elementwise_product)
        return self.variables[max_index]

#########################
#########################

dataset_x = []
dataset_y = []

#x1 = torch.randint(1, 10, size=(5000,))
#x2 = torch.randint(1, 10, size=(5000,))
#x3 = torch.randint(1, 10, size=(5000,))
x1 = torch.FloatTensor(5000).uniform_(1, 10)
x2 = torch.FloatTensor(5000).uniform_(1, 10)
x3 = torch.FloatTensor(5000).uniform_(1, 10)


y = torch.sin(x1+x2)

dataset_x = torch.stack((x1, x2), dim=1)
dataset_y = y.unsqueeze(1)

#########################
scaler = MinMaxScaler()
dataset_x_norm = scaler.fit_transform(dataset_x)
dataset_y_norm = scaler.fit_transform(dataset_y)
#########################

rnn= BinaryTreeRNN()
rnn.learn(dataset_x_norm, dataset_y_norm, num_layers = 2, training_steps_by_standard_softmax = 300, training_steps_by_softmax_prime = 900, lr=.1, lambda_L1 =.001)
"""
print(rnn.first_layer_weights)
print(rnn.weights)
print(rnn.biases)
print(rnn.omegas)
"""
