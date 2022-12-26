import numpy as np
import pdb
"""
Teacher : Kourosh Parand
Student : Ali Nikakhtar 
student id : 99422195
"""
class NeuralNetwork:
    def __init__(self, layers):
        # Initialize weights and biases with random values WHICH X= FIRST LAYER
        #self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        #self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(8, 8),np.random.randn(8, 8),np.random.randn(8, 8)]
        self.biases = [np.random.randn(8),np.random.randn(8),np.random.randn(8)]
###############################################################################2forward
    def forward(self, p, i, k):
        #i and k shows after what count of sfl learn iteration i, we use descrtized softmax  ,i comes from sfl.learn() and k comes from sfl.__init__() 
        # Propagate the input through the network
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            p = np.dot(w, p) + b  # apply dot product and bias addition
            
        p = np.dot(self.weights[-1], p) + self.biases[-1]
        p = p.tolist()

        if(i <= k):
            p = self.softmax_standard(np.dot(self.weights[-1], p) + self.biases[-1])
            #print(p)
            return self.get_operation(p)
        
        # Apply the softmax function to the output of the final layer
        p = self.softmax(np.dot(self.weights[-1], p) + self.biases[-1])
        return self.get_operation(p)
####################################################################################

    def softmax(self, p):
        #p = p.tolist()
        #print(p)

        
        p = np.array(p)

        
        updated_p = np.zeros(p.shape)

        # Find the index of the maximum value in p
        max_index = np.argmax(p)
        # Set the value at the maximum index to 1
        updated_p[max_index] = 1
        # Convert the updated values back to a Python list
        return updated_p.tolist()

    def softmax_standard(self, p):
        # Shift the values of p so that the maximum value is 0
        p -= np.max(p)
        
        # Apply the softmax activation function
        exp_p = np.exp(p)
        return exp_p / np.sum(exp_p)

    def get_operation(self, lst):
        operations = ['+', '-', '*', '/', 'sin', 'cos', 'tan', 'exp']
        max_index = np.argmax(lst)
        return operations[max_index]

    def train(self, predicted_y, sample_y, learning_rate):
        # Backpropagate the error and update the weights and biases
        for w, b in reversed(list(zip(self.weights, self.biases))):
            predicted_y = predicted_y - sample_y
            #w_delta = np.dot(predicted_y, self.softmax_prime(np.dot(w, sample_y) + b))
            w_delta = np.random.randn()
            b_delta = predicted_y
            w -= learning_rate * w_delta
            b -= learning_rate * b_delta


    def softmax_prime(self, z):
        # Calculate the derivative of the softmax activation function
        softmax = self.softmax(z)
        return softmax * (1 - softmax)
########################################

class SymbolicFunctionLearning:
    def __init__(self):
        self.dataset_x = None
        self.dataset_y = None

        self.num_variables = None

        # Initialize a list to store the variable names
        self.variables = None

        # Initialize a list to store the mathematical operations that can be performed
        self.operations = ['+', '-', '*', '/', 'sin', 'cos', 'tan', 'exp']

        # Get the length of the operations list
        self.operations_length = len(self.operations)

        # Store the depth of the tree
        self.tree_depth = None

        self.learning_rate = None
        self.training_steps = None

        self.last_expression = None

        self.best_expression = None

        self.best_cost = None

        self.nn = NeuralNetwork([self.operations_length, 8, self.operations_length])  # initialize the neural network with 8 input neurons, 8 hidden neurons, and 8 output neuron


    def learn(self, dataset_x, dataset_y, tree_depth, learning_rate, training_steps, k):
        #i and k shows after what count of sfl learn iteration i, we use descrtized softmax  ,i comes from sfl.learn() and k comes from sfl.__init__()
        # Convert the datasets to NumPy arrays
        x = np.array(dataset_x)
        y = np.array(dataset_y)

        # Check if the dimensions of the arrays are correct
        if x.shape[0] == y.shape[0] and x.shape[1] > 1 and y.shape[1] == 1:
            print("datasets are okay.")

            self.dataset_x = dataset_x
            self.dataset_y = dataset_y

            self.num_variables = np.array(self.dataset_x).shape[1]

            # Initialize a list to store the variable names
            self.variables = ['x' + str(i+1) for i in range(self.num_variables)]

            self.tree_depth = tree_depth

            self.learning_rate = learning_rate
            self.training_steps = training_steps

            for i in range(self.training_steps):
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                for sample_x, sample_y in zip(self.dataset_x, self.dataset_y):
                    
                    self.last_expression = self.generate_expression(self.tree_depth, sample_x, i, k)
                    #i and k shows after what count of sfl learn iteration i, we use descrtized softmax  ,i comes from sfl.learn() and k comes from sfl.__init__()

                    predicted_y = self.evaluate_expression(self.last_expression, sample_x)
                    
                    # Calculate the cost using mean squared error
                    cost = np.sum((predicted_y - sample_y[0]) ** 2) / 2
                    
                    if(self.best_expression == None):
                        self.best_expression = self.last_expression
                        self.best_cost = cost

                    best_expression_predicted_y = self.evaluate_expression(self.best_expression, sample_x)
                    self.best_cost = np.sum((best_expression_predicted_y - sample_y[0]) ** 2) / 2
                    

                    if(cost < self.best_cost):

                        self.best_cost = cost
                        self.best_expression = self.last_expression
    
                        self.nn.train(predicted_y, float(sample_y[0]), self.learning_rate)
                                                            
                    print("====================================================================")
                    print(f"                            Sample x: {sample_x}")
                    #print("\n")
                    print(f"                            Sample y: {sample_y}")
                    print("                 current f(X) = "+decode_expression(self.last_expression)+" ,predicted y = "+str(predicted_y)+" with cost = "+ str(cost))
                    print("                 BEST FIT F(X): "+decode_expression(self.best_expression)+" ,predicted y = "+str(best_expression_predicted_y)+" with cost = "+str(self.best_cost))
                    print("=====================================================================")

                #if(self.best_cost < 0.001): return
            #print(self.last_expression)
        else:
            print("datasets do not match. please try again.")
            

    def generate_expression(self, tree_depth, sample_x, i, k):
        # Generate a mathematical expression tree with the given depth using the neural network
        if tree_depth == 0:
            # If the depth is zero, return a random variable or a constant value
            #print(self.variables)
            return np.random.choice(self.variables)
        else:
            # If the depth is greater than zero, use the neural network to generate a random operation with two subexpressions
            operation = self.nn.forward(self.operator_p(self.generate_expression(tree_depth-1, sample_x, i, k),
            self.generate_expression(tree_depth-1, sample_x, i, k), sample_x), i, k)

            return [operation,
                    self.generate_expression(tree_depth-1, sample_x, i, k),
                    self.generate_expression(tree_depth-1, sample_x, i, k)]

    def operator_p(self, input_x1, input_x2, sample_x):
        p = []
        p.append(self.evaluate_expression(input_x1, sample_x) + self.evaluate_expression(input_x2, sample_x))
        p.append(self.evaluate_expression(input_x1, sample_x) - self.evaluate_expression(input_x2, sample_x))
        p.append(self.evaluate_expression(input_x1, sample_x) * self.evaluate_expression(input_x2, sample_x))
        p.append(self.evaluate_expression(input_x1, sample_x) / self.evaluate_expression(input_x2, sample_x))
        p.append(np.sin(self.evaluate_expression(input_x1, sample_x) + self.evaluate_expression(input_x2, sample_x)))
        p.append(np.cos(self.evaluate_expression(input_x1, sample_x) + self.evaluate_expression(input_x2, sample_x)))
        p.append(np.tan(self.evaluate_expression(input_x1, sample_x) + self.evaluate_expression(input_x2, sample_x)))
        p.append(np.exp(self.evaluate_expression(input_x1, sample_x) + self.evaluate_expression(input_x2, sample_x)))
        return p


    def evaluate_expression(self, expression, input_values):
        # Evaluate the given mathematical expression with the given input values
        if isinstance(expression, list):
            # If the expression is a list, it represents a mathematical operation
            operation = expression[0]
            if operation == '+':
                return self.evaluate_expression(expression[1], input_values) + self.evaluate_expression(expression[2], input_values)
            elif operation == '-':
                return self.evaluate_expression(expression[1], input_values) - self.evaluate_expression(expression[2], input_values)
            elif operation == '*':
                return self.evaluate_expression(expression[1], input_values) * self.evaluate_expression(expression[2], input_values)
            elif operation == '/':
                return self.evaluate_expression(expression[1], input_values) / self.evaluate_expression(expression[2], input_values)
            elif operation == 'sin':
                return np.sin(self.evaluate_expression(expression[1], input_values))
            elif operation == 'cos':
                return np.cos(self.evaluate_expression(expression[1], input_values))
            elif operation == 'tan':
                return np.tan(self.evaluate_expression(expression[1], input_values))
            elif operation == 'exp':
                return np.exp(self.evaluate_expression(expression[1], input_values))
        else:
            # If the expression is not a list, it is either a constant value or a variable.
            if expression in self.variables:
                # Return the corresponding input value
                index = self.variables.index(expression)
                
                return input_values[index]

#########################
def decode_expression(expression):
        # If the expression is a list, it represents a mathematical operation
        if isinstance(expression, list):
            # Decode the operation and the two subexpressions
            operation = expression[0]
            subexpression1 = decode_expression(expression[1])
            subexpression2 = decode_expression(expression[2])

            # Return the infix notation for the operation and the subexpressions
            if operation == '+':
                return f"({subexpression1} + {subexpression2})"
            elif operation == '-':
                return f"({subexpression1} - {subexpression2})"
            elif operation == '*':
                return f"({subexpression1} * {subexpression2})"
            elif operation == '/':
                return f"({subexpression1} / {subexpression2})"
            elif operation == 'sin':
                return f"sin({subexpression1})"
            elif operation == 'cos':
                return f"cos({subexpression1})"
            elif operation == 'tan':
                return f"tan({subexpression1})"
            elif operation == 'exp':
                return f"exp({subexpression1})"
        else:
            # If the expression is not a list, it is a constant value or a variable name
            return str(expression)
"""
dataset_x = [[4, 4],
            [5, 4],
            [5, 5],
            [5, 6],
            [6, 6],
            [7, 6],
            [7, 7]]

dataset_y = [[16],
            [20],
            [25],
            [30],
            [36],
            [42],
            [49]]
"""
#########################

dataset_x = []
dataset_y = []

x1 = np.random.randint(1, 10, size=50)
x2 = np.random.randint(1, 10, size=50)


y = (x2*x2)

dataset_x = np.column_stack((x1, x2))
dataset_y = np.column_stack((y,))

#########################
#########################
sfl = SymbolicFunctionLearning()

#i and k shows after what count of sfl learn iteration i, we use descrtized softmax  ,i comes from sfl.learn() and k comes from sfl.__init__()
sfl.learn(dataset_x, dataset_y, 1, 0.1, 500, 100)
