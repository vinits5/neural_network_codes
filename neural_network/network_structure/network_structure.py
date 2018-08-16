import tensorflow as tf 


class network_structure():
	# Use it to initialize weights.
	def weights(self,x,y):
		weights_dict = {'weights':tf.Variable(tf.random_normal([x,y])),'biases':tf.Variable(tf.random_normal([y]))}
		return weights_dict

	# Define the complete neural network here.
	def structure(self):
		self.x = tf.placeholder(tf.float32,shape=(1,2))
		self.y = tf.placeholder(tf.float32,shape=(1,2))

		self.nodes_layer1 = 200
		self.hidden_layer1 = self.weights(2,self.nodes_layer1)

		self.nodes_output = 2
		self.output_layer = self.weights(self.nodes_layer1,self.nodes_output)

		self.l1 = tf.add(tf.matmul(self.x,self.hidden_layer1['weights']),self.hidden_layer1['biases'])
		self.l1 = tf.nn.relu(self.l1)

		self.output = tf.add(tf.matmul(self.l1,self.output_layer['weights']),self.output_layer['biases'])

		self.loss = tf.reduce_sum(tf.square(self.output-self.y))
		self.trainer = tf.train.AdamOptimizer()
		self.updateModel = self.trainer.minimize(self.loss)