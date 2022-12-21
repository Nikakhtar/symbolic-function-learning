import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        # Initialize weights and biases with random values WHICH X= FIRST LAYER
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]

    def forward(self, x):
        # Propagate the input through the network   WHICH X= FIRST LAYER
        for w, b in zip(self.weights, self.biases):
            x = self.softmax(np.dot(w, x) + b)
        return x
###############################################################################2forward
    def forward(self, x):
        # Propagate the input through the network
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.dot(w, x) + b  # apply dot product and bias addition

        # Apply the softmax function to the output of the final layer
        x = self.softmax(np.dot(self.weights[-1], x) + self.biases[-1])
        return x
####################################################################################

    def softmax(self, z):
        # Apply the softmax activation function
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def train(self, x, y, learning_rate):
        # Propagate the input through the network
        output = self.forward(x)

        # Calculate the cost using mean squared error
        cost = np.sum((output - y) ** 2) / 2

        # Backpropagate the error and update the weights and biases
        for w, b in reversed(list(zip(self.weights, self.biases))):
            output = output - y
            w_delta = np.dot(output, self.softmax_prime(np.dot(w, x) + b))
            b_delta = output
            w -= learning_rate * w_delta
            b -= learning_rate * b_delta

        return cost

    def softmax_prime(self, z):
        # Calculate the derivative of the softmax activation function
        softmax = self.softmax(z)
        return softmax * (1 - softmax)



///////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
nn = NeuralNetwork([2, 3, 1])  # initialize the neural network with 2 input neurons, 3 hidden neurons, and 1 output neuron

# Set the learning rate
learning_rate = 0.1

# Set the number of training steps
training_steps = 5000

# Train the neural network
for i in range(training_steps):
    # Provide the input and output data for training
    x = ...
    y = ...

    # Train the neural network on the input and output data
    cost = nn.train(x, y, learning_rate)

    # Print the cost after each training step to track the progress
    print(f"Step {i+1}: cost = {cost}")
