from sklearn.preprocessing import MinMaxScaler
from numpy.typing import NDArray
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import math
import matplotlib.pyplot as plt
import pdb

class BinaryTreeRNN(nn.Module):
    def __init__(self):
        super(BinaryTreeRNN, self).__init__()

        self.num_layers = None

        self.last_expression = None

        self.training_steps_by_standard_softmax = None

        self.training_steps_by_softmax_prime = None
        
        self.total_training_steps = None

        # Initialize a list to store the variable names
        self.variables = None

        self.num_variables = None
        
        # Initialize a list to store the mathematical operations that can be performed
        self.operations = ['+', 'sin', '*']

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
        
        self.test_expression = None
        
        self.best_parsed_math_expr_tree = None
        
        self.best_mean_abs_diff = 999999999.999
        

 ######################### 
    def learn(self, test_expression, dataset_x, dataset_y, num_layers, training_steps_by_standard_softmax, training_steps_by_softmax_prime, lr, lambda_L1):

        # Check if the dimensions of the arrays are correct
        if dataset_x.shape[0] == dataset_y.shape[0] and dataset_x.shape[1] >= 1:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("datasets are okay.")
            
            self.test_expression = test_expression

            self.num_variables = np.array(dataset_x).shape[1]

            # Initialize a list to store the variable names
            self.variables = ['x' + str(i+1) for i in range(self.num_variables)]
            

            self.num_layers = num_layers

            self.training_steps_by_standard_softmax = training_steps_by_standard_softmax
            self.training_steps_by_softmax_prime = training_steps_by_softmax_prime
            self.total_training_steps =  self.training_steps_by_standard_softmax + self.training_steps_by_softmax_prime

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

            for self.step_counter in range(self.total_training_steps):

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
                
                ###############################
                ###############################
                
            
                parsed_math_expr_tree = self.parse_math_expr_tree(self.get_nth_element(self.last_expression))
            
                self.expression_plotter(dataset_x, self.test_expression, parsed_math_expr_tree)
            
            return self.best_mean_abs_diff

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
    def expression_plotter(self, dataset_x, expression_y, expression_y_hat):
        
        #just create a copy of the argument to have a unmodified version
        expression_y_hat_copy = expression_y_hat
        
        #expression_y = "sin(cos(x1+x1)+sin(x1+x1))"
        # add the torch namespace to the expression string
        expression_y = expression_y.replace('sin', 'torch.sin')
        expression_y = expression_y.replace('cos', 'torch.cos')
        expression_y = expression_y.replace('x1', 'dataset_x[:,0]')

        expression_y_hat = expression_y_hat.replace('sin', 'torch.sin')
        expression_y_hat = expression_y_hat.replace('cos', 'torch.cos')
        expression_y_hat = expression_y_hat.replace('x1', 'dataset_x[:,0]')

        #exec(expression_y)

        y = eval(expression_y)
        y_hat = eval(expression_y_hat)
        
        abs_diff = torch.abs(y - y_hat)
        mean_abs_diff = torch.mean(abs_diff)
        
        if mean_abs_diff < self.best_mean_abs_diff:
            self.best_mean_abs_diff = mean_abs_diff
            self.best_parsed_math_expr_tree = expression_y_hat_copy
            
        if self.step_counter == self.total_training_steps - 1:
            
            print("y = "+self.test_expression)
            print("best y_hat = "+self.best_parsed_math_expr_tree)
            print("Real ERROR="+str(self.best_mean_abs_diff))
            
            self.best_parsed_math_expr_tree = self.best_parsed_math_expr_tree.replace('sin', 'torch.sin')
            self.best_parsed_math_expr_tree = self.best_parsed_math_expr_tree.replace('cos', 'torch.cos')
            self.best_parsed_math_expr_tree = self.best_parsed_math_expr_tree.replace('x1', 'dataset_x[:,0]')
            
            y_hat = eval(self.best_parsed_math_expr_tree)
            
            # Create scatter plot of X against y
            plt.scatter(dataset_x[:,0], y)
            plt.scatter(dataset_x[:,0], y_hat)

            plt.show()
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

                   
        #print(y)
        #y = torch.sin(torch.cos(dataset_x[:,0]+dataset_x[:,0])+torch.sin(dataset_x[:,0]+dataset_x[:,0]))
        #y_hat = torch.cos(torch.cos(dataset_x[:,0]+dataset_x[:,0])+(dataset_x[:,0]*dataset_x[:,0]))

#########################
x1 = torch.FloatTensor(500).uniform_(1, 10)
x2 = torch.FloatTensor(500).uniform_(1, 10)
x3 = torch.FloatTensor(500).uniform_(1, 10)

#⚠ CAUTIONS ⚠ IF YOU WANT TO HAVE DATASET_X CONTAINING MORE THAN ONE FEATURE(X1), THEN MAKE THE LINE 116(plotting) AS COMMENTS  ⚠ CAUTIONS ⚠⚠ CAUTIONS ⚠⚠ CAUTIONS ⚠⚠ CAUTIONS ⚠
dataset_x = torch.stack((x1,), dim=1)
#⚠ CAUTIONS ⚠ IF YOU WANT TO HAVE DATASET_X CONTAINING MORE THAN ONE FEATURE(X1), THEN MAKE THE LINE 116(plotting) AS COMMENTS  ⚠ CAUTIONS ⚠⚠ CAUTIONS ⚠⚠ CAUTIONS ⚠⚠ CAUTIONS ⚠

# create a 10x10 matrix of zeros
real_errors = torch.zeros(10, 10)
#########################
# Loop 1
dataset_y = x1+x1
test_expression = "x1+x1"

for i in range(10):
    rnn= BinaryTreeRNN()
    real_errors[0][i]=rnn.learn(test_expression, dataset_x, dataset_y, num_layers = 2, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
    #########################
    
# Loop 2
dataset_y = x1*x1
test_expression = "x1*x1"

for i in range(10):
    rnn= BinaryTreeRNN()
    real_errors[1][i]=rnn.learn(test_expression, dataset_x, dataset_y, num_layers = 2, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
    #########################
    
# Loop 3
dataset_y = torch.sin(x1+x1)
test_expression = "sin(x1+x1)"

for i in range(10):
    rnn= BinaryTreeRNN()
    real_errors[2][i]=rnn.learn(test_expression, dataset_x, dataset_y, num_layers = 2, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
    #########################
    
# Loop 4
dataset_y = torch.sin(x1+x1)+torch.sin(x1+x1)
test_expression = "sin(x1+x1)+sin(x1+x1)"

for i in range(10):
    rnn= BinaryTreeRNN()
    real_errors[3][i]=rnn.learn(test_expression, dataset_x, dataset_y, num_layers = 3, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
    #########################
    
# Loop 5
dataset_y = (x1+x1)*(x1+x1)
test_expression = "(x1+x1)*(x1+x1)"

for i in range(10):
    rnn= BinaryTreeRNN()
    real_errors[4][i]=rnn.learn(test_expression, dataset_x, dataset_y, num_layers = 3, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
    #########################
    
# Loop 6
dataset_y = torch.sin((x1*x1)+torch.sin(x1+x1))
test_expression = "sin((x1*x1)+sin(x1+x1))"

for i in range(10):
    rnn= BinaryTreeRNN()
    real_errors[5][i]=rnn.learn(test_expression, dataset_x, dataset_y, num_layers = 3, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
    #########################
    
# Loop 7
dataset_y = ((x1+x1)*torch.sin(x1+x1))
test_expression = "((x1+x1)*sin(x1+x1))"

for i in range(10):
    rnn= BinaryTreeRNN()
    real_errors[6][i]=rnn.learn(test_expression, dataset_x, dataset_y, num_layers = 3, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
    #########################
    
# Loop 8
dataset_y = ((x1*x1)+(x1*x1))
test_expression = "((x1*x1)+(x1*x1))"

for i in range(10):
    rnn= BinaryTreeRNN()
    real_errors[7][i]=rnn.learn(test_expression, dataset_x, dataset_y, num_layers = 3, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
    #########################
    
# Loop 9
dataset_y = (torch.sin(x1+x1)*torch.sin(x1+x1))
test_expression = "(sin(x1+x1)*sin(x1+x1))"

for i in range(10):
    rnn= BinaryTreeRNN()
    real_errors[8][i]=rnn.learn(test_expression, dataset_x, dataset_y, num_layers = 3, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
    #########################
    
# Loop 10
dataset_y = ((x1*x1)*(x1+x1))
test_expression = "((x1*x1)*(x1+x1))"

for i in range(10):
    rnn= BinaryTreeRNN()
    real_errors[9][i]=rnn.learn(test_expression, dataset_x, dataset_y, num_layers = 3, training_steps_by_standard_softmax = 3000, training_steps_by_softmax_prime = 9000, lr=.1, lambda_L1 =.001)
    ###########################
print("............................................................................................")
print("............................................................................................")
print("real_errors matrix =")    
print(real_errors)
print("..........................")
mean_real_errors = torch.mean(real_errors)
print("mean real_errors = ")
print(mean_real_errors)
