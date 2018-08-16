import tensorflow as tf 
import os
import numpy as np
from neural_network.neural_network import neural_network
from neural_network.network_structure.Logger import Logger

# Create a Neural Network Class.
nn = neural_network()
# Save network structure in logs.
nn.save_network_structure()

x = [[0.1,0.2]]
y = [[1,1]]

# Create logger file for tensorboard.
# Get the path from neural network class.
logger = Logger(nn.path)

with tf.Session() as sess:
	# Define Tensors.
	nn.create_model()

	# Initialize tensors.
	nn.session_init(sess)

	# Forward pass in neural network.
	op = nn.forward(x)

	# Backpropagation in neural network.
	loss = nn.backward(x,y)

	# Tensorboard log.
	logger.log_scalar(tag='Loss',value=loss,step=1)
	logger.log_scalar(tag='op_x',value=float(op[0][0][0]),step=1)
	logger.log_scalar(tag='op_y',value=op[0][0][1],step=1)

	# Save weights for future purpose.
	nn.save_weights(1)
	print(op[0])	
	print(loss)