# Symbolic Function Learner with Neural Networks
a new deep learning approach to SFL using NN with a discretized softmax-function and an Operator-function P(x[2,4,71..], operators[+,*,/, sin, cos...]), which applies to each node of expression tree and creates the the inputs to the NN.
![download](https://github.com/Nikakhtar/symbolic-function-learning/assets/47872183/de26b145-fcd5-4871-bb6c-5b056e74222d)
## Overview
This repository contains a neural network implementation for learning symbolic mathematical functions from data. The main architecture is based on a binary tree recurrent neural network (BinaryTreeRNN) which can represent and learn various mathematical expressions.

## Features
- **BinaryTreeRNN**: The main neural network architecture to represent and learn mathematical expressions in a tree-like structure.
- **Training and Inference**: The network can be trained with given datasets to infer the mathematical expressions.
- **Visualization**: Ability to visualize the learned expressions compared to the ground truth.

## Usage
1. Prepare your data. An example dataset is provided with `x1`, `x2`, and `x3`. You can use them or replace them with your own datasets.
2. Initialize your network:
   ```python
   net = BinaryTreeRNN()
   ```
3. Train the model with your data:
   ```python
   dataset_x = torch.stack((x1,), dim=1)  # Example for using single feature x1.
   dataset_y = YOUR_TARGET_DATA  # Replace with your target data.
   net.learn(dataset_x, dataset_y, num_layers, training_steps_by_standard_softmax, training_steps_by_softmax_prime, lr, lambda_L1)
   ```
4. Visualize the results using the `expression_plotter` function if needed.

⚠ **CAUTION**: If you want `dataset_x` to contain more than one feature (like `x1` and `x2`), comment out the plotting section on line 116.

## Contributions
Feel free to submit pull requests, open issues, and share feedback to enhance and expand the functionality of this symbolic function learner.
