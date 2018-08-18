import tensorflow as tf 
from network_structure.network_structure import network_structure
import os
import datetime
import shutil

# Class for neural network structure.
ns = network_structure()

class neural_network():
	# Initialize the path for storing data.
	def __init__(self):
		now = datetime.datetime.now()
		path = os.getcwd()
		try:
			os.mkdir('log_data')
		except:
			pass
		self.path = os.path.join(path,'log_data/',now.strftime("%Y-%m-%d-%H-%M-%S"))
		os.mkdir(self.path)

	# Initialize the variables of Neural Network.
	def session_init(self,sess):
		self.sess = sess
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep = 100)

	# Define the network structure.
	def create_model(self):
		ns.structure()

	# Forward pass of neural network.
	def forward(self,ip):
		op = self.sess.run([ns.output],feed_dict={ns.x:ip})
		return op

	# Backpropagation of neural network.
	def backward(self,ip,y):
		l,_ = self.sess.run([ns.loss,ns.updateModel],feed_dict={ns.x:ip,ns.y:y})
		return l		

	# Store weights for further use.
	def save_weights(self,episode):
		path_w = os.path.join(self.path,'weights')
		try:
			os.chdir(path_w)
		except:
			os.mkdir(path_w)
			os.chdir(path_w)
		path_w = os.path.join(path_w,'{}.ckpt'.format(episode))
		self.saver.save(self.sess,path_w)

	# Load the Weights
	def load_weights(self,weights):
		self.saver.restore(self.sess,weights)

	# Store network structure in logs.
	def save_network_structure(self):
		curr_dir = os.getcwd()
		src_path = os.path.join(curr_dir,'neural_network','network_structure','network_structure.py')
		target_path = os.path.join(self.path,'network_structure.py')
		shutil.copy(src_path,target_path)

	# Show the data on a single line.
	def print_data(self,text,step,data):
		text = "\r"+text+" %d: %f"
		sys.stdout.write(text%(step,data))
		sys.stdout.flush()
