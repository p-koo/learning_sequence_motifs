import tensorflow as tf
from tensorflow.python.training import moving_averages
from .base import BaseLayer
from ..utils import Variable
from .. import init

__all__ = [
	"BatchNormLayer"
]



class BatchNormLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, is_training, **kwargs):

		# input data shape
		self.incoming = incoming
		incoming_shape = self.incoming.get_output_shape()

		bn_shape = incoming_shape[-1]

		self.bn_axes = [0]
		if len(incoming_shape) == 4:
			self.bn_axes = [0, 1, 2]
		else:
			self.bn_axes = [0]

		self.gamma = Variable(var=init.Constant(value=1.), shape=[bn_shape], regularize=False)
		self.beta = Variable(var=init.Constant(value=0.), shape=[bn_shape], regularize=False)

		self.epsilon = 1e-08
		if 'epsilon' in kwargs.keys():
			self.epsilon = kwargs['epsilon']
		self.decay = 0.95
		if 'decay' in kwargs.keys():
			self.decay = kwargs['decay']

		self.is_training = is_training
		self.pop_mean = tf.train.ExponentialMovingAverage(decay=0.99)
		self.pop_var = tf.train.ExponentialMovingAverage(decay=0.99)


	def get_output(self):
		batch_mean, batch_var = tf.nn.moments(self.incoming.get_output(), self.bn_axes)

		def update_mean_var():
			pop_mean_op = self.pop_mean.apply([batch_mean])
			pop_var_op = self.pop_var.apply([batch_var])
			with tf.control_dependencies([pop_mean_op, pop_var_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		def population_mean_var():
			return self.pop_mean.average(batch_mean), self.pop_var.average(batch_var)

		mean, var = tf.cond(self.is_training, update_mean_var, population_mean_var)

		#z_score = tf.divide(tf.subtract(self.incoming.get_output(), mean), tf.sqrt(tf.add(var, self.epsilon)))
		#return tf.add(tf.multiply(self.gamma.get_variable(), z_score), self.beta.get_variable())
		return tf.nn.batch_normalization(self.incoming.get_output(), mean, var,
										 self.beta.get_variable(), self.gamma.get_variable(), self.epsilon)

	def get_output_shape(self):
		return self.incoming.get_output_shape()

	def get_variable(self):
		return [self.gamma, self.beta]

	def set_trainable(self, status):
		self.gamma.set_trainable(status)
		self.beta.set_trainable(status)

	def is_trainable(self):
		return self.gamma.is_trainable()

	def is_l1_regularize(self):
		return self.gamma.is_l1_regularize()

	def is_l2_regularize(self):
		return self.gamma.is_l2_regularize()
