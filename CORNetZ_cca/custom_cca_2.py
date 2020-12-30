import tensorflow as tf
import numpy as np
# tf.enable_eager_execution()

class Custom_CCA(object):

	#X & Y are matrices which are supplied to the model incrementally
	def __init__(self, x, y, dim, batch_size, weightage = 1):
		self.x = x
		self.y = y
		self.BATCH_SIZE = batch_size
		self.dim = dim
		self.weightage = weightage

		with tf.compat.v1.variable_scope("cca_weights", reuse=tf.compat.v1.AUTO_REUSE) as scope:
			self.rot_x  = tf.compat.v1.get_variable("rot_x",  shape=[x.shape[1], dim])
			print(self.rot_x)
			self.rot_y  = tf.compat.v1.get_variable("rot_y",  shape=[y.shape[1], dim])
			print(self.rot_y)

	def GetOrthogonality(self, w):
		print('in Getorthog')
		w = tf.transpose(a=w)
		print('w')
		print(w)
		print(w.shape[1])
		print('!!')
		m = tf.matmul(tf.transpose(a=w), tf.Variable(w)) - tf.eye(50)
		print("MMmmmmmMMMMMMMM")
		j = self.weightage * tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(m)))
		print(j)
		return j


	def NewForward(self):
		print("in new forward")



	def forward(self):
		print("IN FORWARD")
		print(self.x)
		print(self.y)
		# print(self.rot_x)
		# print(self.rot_y)
		y1 = tf.matmul(self.x, self.rot_x)
		y2 = tf.matmul(self.y, self.rot_y)
		print("YYYS")
		print(y1)
		print(y2)
		# print('in forward')
		# print(self.BATCH_SIZE)
		# print(self.dim)
		# print("look at the ortho")
		# print(self.GetOrthogonality(self.rot_x, True))
		# print(self.GetOrthogonality(self.rot_y, False))
		# y1 = tf.matmul(tf.transpose(self.x), self.GetOrthogonality(self.rot_x, True))
		# y2 = tf.matmul(tf.transpose(self.y), self.GetOrthogonality(self.rot_y, False))
		# print("???????")
		# q1 = self.GetOrthogonality(y1)
		# q2 = self.GetOrthogonality(y2)
		# print('TTTTTTTTTTRTTT')
		# print(q1)
		# print(q2)





		#Return the projection
		return y1, y2
		# return q1, q2



