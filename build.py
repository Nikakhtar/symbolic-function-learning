
from sklearn.preprocessing import MinMaxScaler
from numpy.typing import NDArray
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import math

class BinaryTreeRNN(nn.Module):
    def __init__(self):
        super(BinaryTreeRNN, self).__init__()

        self.num_layers = None

        self.last_expression = None

        self.training_steps_by_standard_softmax = None

        self.training_steps_by_softmax_prime = None

        # Initialize a list to store the variable names
        self.variables = None

        self.num_variables = None
        
        # Initialize a list to store the mathematical operations that can be performed
        self.operations = ['+', 'sin', 'cos', '*']

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

        # Check if the dimensions of the arrays are correct
        if dataset_x.shape[0] == dataset_y.shape[0] and dataset_x.shape[1] > 1:
            print("datasets are okay.")

            self.num_variables = np.array(dataset_x).shape[1]

            # Initialize a list to store the variable names
            self.variables = ['x' + str(i+1) for i in range(self.num_variables)]
            

            self.num_layers = num_layers

            self.training_steps_by_standard_softmax = training_steps_by_standard_softmax
            self.training_steps_by_softmax_prime = training_steps_by_softmax_prime
            total_training_steps =  self.training_steps_by_standard_softmax + self.training_steps_by_softmax_prime

            # Initialize weights and biases for the first layer(Leaf)
            self.first_layer_weights = nn.Parameter(torch.randn(2**(num_layers-1), self.num_variables, dtype=torch.float32), requires_grad=True)

            self.first_layer_biases = nn.Parameter(torch.randn(2**(num_layers-1), dtype=torch.float32), requires_grad=True)
            
            # Initialize weights and biases for the remaining layers(interior)
            self.weights = nn.ParameterList([nn.Parameter(torch.randn(2**(level-1), dtype=torch.float32), requires_grad=True) for level in range(1, self.num_layers)])
            self.biases = nn.ParameterList([nn.Parameter(torch.randn(2**(level-1), dtype=torch.float32), requires_grad=True) for level in range(1, self.num_layers)])


            # Initialize omegas for the remaining layers
            self.omegas = nn.ParameterList([nn.Parameter(torch.randn(2**(level-1), self.operations_length, dtype=torch.float32), requires_grad=True) for level in range(1, self.num_layers)])

            # set up the optimizer
            optimizer = optim.SGD(self.parameters(), lr=lr)
            

            self.lambda_L1 = lambda_L1

            for self.step_counter in range(total_training_steps):

                optimizer.zero_grad()
                loss = 0.0
                
                # create an empty tensor of the desired shape
                tensor_shape = (2**self.num_layers - 1, dataset_x.shape[0])
                self.hidden_values = torch.empty(*tensor_shape, dtype=torch.float)
                #self.last_expression = [[None for j in range(dataset_x.shape[0])] for i in range(2**self.num_layers - 1)]
                self.last_expression = [None] * (2**self.num_layers -1)
                
                torch.autograd.set_detect_anomaly(False)

                y_hat = self.forward(1, dataset_x)

                # define the loss function
                loss = nn.functional.mse_loss(y_hat, dataset_y)


                if(self.step_counter%2 == 0):
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                    print(" y = "+self.parse_math_expr_tree(self.get_nth_element(self.last_expression)))
                    print("ERROR="+str(loss))

                # calculate L1 regularization term
                L1_reg = torch.tensor(0., requires_grad=True)
                for param in self.parameters():
                    L1_reg = L1_reg + torch.norm(param, 1)
                L1_loss = self.lambda_L1 * L1_reg
                loss = loss + L1_loss
                
                
                loss.backward(retain_graph=True) # backward pass

                # clip gradients to avoid exploding gradients
                # set the maximum norm to 1.0
                max_norm = 1.0
                utils.clip_grad_norm_(self.parameters(), max_norm)

                # update the parameters with gradient descent
                optimizer.step()


        else:
              print("datasets do not match. please try again.")

#########################
    def forward(self, i, dataset_x):
        left_child = 2 * i
        right_child = 2 * i + 1
        layer, j = self.layer_position(i)

        if left_child >= self.num_layers ** 2 - 1:
            if right_child >= self.num_layers ** 2 - 1:

                f_l_w = self.first_layer_weights[j]
                f_l_b = self.first_layer_biases[j]
                d_x_t = dataset_x.transpose(0, 1)
                h_v = torch.matmul(f_l_w, d_x_t) + f_l_b

                self.hidden_values[i-1] = h_v.clone()
                
                self.last_expression[i-1] = self.get_leaf(f_l_w, d_x_t)


                return self.hidden_values[i-1]
                  ############

        weight = self.weights[layer][j]
        bias = self.biases[layer][j]

        left_output = self.forward(left_child, dataset_x)
        right_output = self.forward(right_child, dataset_x)
            
        op_p = self.operator_p(left_output, right_output)

        if(self.step_counter < self.training_steps_by_standard_softmax):
            omega = F.softmax(self.omegas[layer][j], dim=0)
        
        else:
            omega = self.softmax_prime(self.omegas[layer][j])

        hidden_value = weight * torch.matmul(omega, op_p) + bias

        self.hidden_values[i-1] = hidden_value.clone()

        self.last_expression[i-1] = self.get_operation(omega, op_p)
        
        return self.hidden_values[i-1]
#########################

    def operator_p(self, left_child_h, right_child_h):
        # initialize an empty vector to hold the results
        p = torch.zeros((len(self.operations), len(left_child_h)), requires_grad=True)
        # loop over each operator in self.operations and apply it to left_child_h and right_child_h
        for i, op in enumerate(self.operations):
            
            p_copy = p.clone()

            if op == '+':
                p_copy[i] = torch.add(left_child_h, right_child_h)
            elif op == '*':
                f = torch.mul(left_child_h.clone(), right_child_h.clone())
                p_copy[i] = f.clone()
            elif op == 'sin':
                add = torch.add(left_child_h, right_child_h)
                p_copy[i] = torch.sin(add)
            elif op == 'cos':
                add = torch.add(left_child_h  ,right_child_h)
                p_copy[i] = torch.cos(add)
            elif op == 'exp':
                add = torch.add(left_child_h  ,right_child_h)
                p_copy[i] = torch.exp(add)
            elif op == 'id':
                p_copy[i] = torch.add(left_child_h, right_child_h)

            p = p_copy

        # return the resulting vector
        return p

######################
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
    def get_operation(self, s, v):
        elementwise_product = torch.mul(s, v.transpose(0,1))
        max_index = torch.argmax(elementwise_product, dim=-1)
        return [self.operations[i] for i in max_index]

#########################
    def get_leaf(self, w, x):
        elementwise_product = torch.mul(w, x.transpose(0,1))
        max_index = torch.argmax(elementwise_product, dim=-1)
        return [self.variables[i] for i in max_index]

#########################
    def get_nth_element(self, lst):
        return [sublst[len(lst)] for n, sublst in enumerate(lst, start=1)]
#########################
    def parse_math_expr_tree(self, expr_tree_list):
        # Define a recursive function to parse the tree
        def parse_node(node_index):
            if node_index > len(expr_tree_list):
                return ''
            
            # Check if this is a leaf node (i.e. a variable)
            if 2*node_index >= len(expr_tree_list):
                return expr_tree_list[node_index-1]
            
            # Otherwise, this is an interior node (i.e. an operator)
            operator = expr_tree_list[node_index-1]
            left_child_index = 2*node_index
            right_child_index = 2*node_index + 1
            
            # Recursively parse the left and right children
            left_expr = parse_node(left_child_index)
            right_expr = parse_node(right_child_index)
            
            # Combine the left and right expressions with the operator
            if operator == 'sin':
                return f'sin({left_expr} + {right_expr})'
            elif operator == 'cos':
                return f'cos({left_expr} + {right_expr})'
            elif operator == 'exp':
                return f'e^({left_expr} + {right_expr})'
            elif operator == '+':
                return f'({left_expr} + {right_expr})'
            elif operator == '-':
                return f'({left_expr} - {right_expr})'
            elif operator == '*':
                return f'({left_expr} * {right_expr})'
            elif operator == '/':
                return f'({left_expr} / {right_expr})'
            else:
                return ''
        
        # Start parsing at the root node (index 1)
        return parse_node(1)
#########################
x1 = torch.FloatTensor(50000).uniform_(1, 10)
x2 = torch.FloatTensor(50000).uniform_(1, 10)
x3 = torch.FloatTensor(50000).uniform_(1, 10)

dataset_x = torch.stack((x1, x2, x3), dim=1)
dataset_y = torch.sin(x1+x2)
#########################
#scaler = MinMaxScaler()
#dataset_x_norm = scaler.fit_transform(dataset_x)
#dataset_y_norm = scaler.fit_transform(dataset_y)
#########################
#dataset_x_norm = np.array(dataset_x_norm)
#dataset_y_norm = np.array(dataset_y_norm)

#dataset_x = torch.from_numpy(dataset_x_norm)
#dataset_x = torch.tensor(dataset_x, dtype=torch.float)
#dataset_y = torch.from_numpy(dataset_y_norm)

rnn= BinaryTreeRNN()
rnn.learn(dataset_x, dataset_y, num_layers = 2, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
