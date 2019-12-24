from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .base import BaseLayer
from ..utils import Variable
from .. import init


__all__ = [
	"MaxLayer",
	"MeanLayer",
	"ActivationLayer",
	"BiasLayer",
	"StochasticBiasLayer",
	"ElementwiseSumLayer",
	"ConcatLayer",
	"Softmax2DLayer",
]


class Softmax2DLayer(BaseLayer):
	def __init__(self, incoming, **kwargs):
		
		shape = incoming.get_output_shape().as_list()
		num_classes = shape[2]
		num_categories = shape[1]

		self.incoming_shape = incoming.get_output_shape()
		
		reshape = tf.reshape(incoming.get_output(), [-1, num_classes], **kwargs)

		softmax = tf.nn.softmax(reshape)
		self.output = tf.reshape(softmax, [-1, num_categories, num_classes])
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape


class MaxLayer(BaseLayer):
	def __init__(self, incoming, axis, **kwargs):
		
		self.incoming_shape = incoming.get_output_shape()
		self.output = tf.reduce_max(incoming.get_output(), axis=axis)		
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
		


class MeanLayer(BaseLayer):
	def __init__(self, incoming, axis, **kwargs):
		
		self.incoming_shape = incoming.get_output_shape()
		self.output = tf.reduce_mean(incoming.get_output(), axis=axis)		
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
		

class ElementwiseSumLayer(BaseLayer):
	"""activation layer"""

	def __init__(self, incomings, **kwargs):
		

		self.incoming_shape = incomings[0].get_output_shape()
		self.output = tf.add(incomings[0].get_output(), incomings[1].get_output())
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape



class ConcatLayer(BaseLayer):
	"""concat layer"""

	def __init__(self, incomings, **kwargs):
		

		self.incoming_shape = incomings[0].get_output_shape()
		self.output = tf.concat([incomings[0].get_output(), incomings[1].get_output()], axis=-1)
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape




class ActivationLayer(BaseLayer):
	"""activation layer"""

	def __init__(self, incoming, function=[], **kwargs):
		
		self.function = function
		if not self.function:
			self.function = 'relu'

		self.alpha = []
		if self.function == 'prelu':
			self.alpha = Variable(var=init.Constant(0.2), shape=(), **kwargs)
			self.output = activation(z=incoming.get_output(), 
									function=self.function, 
									alpha=self.alpha.get_variable())
		else:
			self.output = activation(z=incoming.get_output(), 
									function=self.function, 
									**kwargs)

		self.incoming_shape = incoming.get_output_shape()
		

		
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape

	def get_variable(self):	
		return self.alpha

	def is_trainable(self):
		if self.alpha:
			return self.alpha.is_trainable()
		else:
			return False	


class BiasLayer(BaseLayer):
	"""Bias layer"""
	
	def __init__(self, incoming, b=[], **kwargs):
		
		self.incoming_shape = incoming.get_output_shape()
		if len(self.incoming_shape) > 2:
			num_units = self.incoming_shape[3].value
		else:
			num_units = self.incoming_shape[1].value

			
		if not b:
			self.b = Variable(var=init.Constant(0.05), 
						 	  shape=[num_units], 
							  **kwargs)
		else:
			self.b = Variable(var=b, shape=[num_units], **kwargs)
			
		
		self.output = tf.nn.bias_add(incoming.get_output(), self.b.get_variable())
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return self.b
	
	def set_trainable(self, status):
		self.b.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.b.set_l1_regularize(status)    
		
	def set_l2_regularize(self, status):
		self.b.set_l2_regularize(status)    
	
	def is_trainable(self):
		return self.b.is_trainable()
		
	def is_l1_regularize(self):
		return self.b.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.b.is_l2_regularize()  



class StochasticBiasLayer(BaseLayer):
	"""Bias layer"""
	
	def __init__(self, incoming, b=[], **kwargs):
		
		self.incoming_shape = incoming.get_output_shape()
		if len(self.incoming_shape) > 2:
			num_units = self.incoming_shape[3].value
		else:
			num_units = self.incoming_shape[1].value

			
		if not b:
			self.b_mu = Variable(var=init.Constant(0.05), 
						 	  shape=[num_units], 
							  **kwargs)
			self.b_sigma = Variable(var=init.Constant(0.05), 
						 	  shape=[num_units], 
							  **kwargs)
		else:
			self.b_mu = Variable(var=b, shape=[num_units], **kwargs)
			self.b_sigma = Variable(var=b, shape=[num_units], **kwargs)
		z = tf.random_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32) 
		self.b = self.b_mu.get_variable() + tf.multiply(tf.exp(0.5 * self.b_sigma.get_variable()), z)

		self.output = tf.nn.bias_add(incoming.get_output(), self.b)
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return [self.b_mu, self.b_sigma]
	
	def set_trainable(self, status):
		self.b_mu.set_trainable(status)
		self.b_sigma.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.b_mu.set_l1_regularize(status)    
		self.b_sigma.set_l1_regularize(status)    
		
	def set_l2_regularize(self, status):
		self.b_mu.set_l2_regularize(status)    
		self.b_sigma.set_l2_regularize(status)    
	
	def is_trainable(self):
		return self.b_mu.is_trainable()
		
	def is_l1_regularize(self):
		return self.b_mu.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.b_mu.is_l2_regularize()  




#---------------------------------------------------------------------------------
# useful functions
#---------------------------------------------------------------------------------

		
def activation(z, function='relu', **kwargs):
	if function == 'relu':
		output = tf.nn.relu(z, **kwargs)

	elif function == 'linear':
		output = z

	elif function == 'sigmoid':
		output = tf.nn.sigmoid(z, **kwargs)

	elif function == 'softmax':
		output = tf.nn.softmax(z, **kwargs)

	elif function == 'elu':
		output = tf.nn.elu(z, **kwargs)

	elif function == 'softplus':
		output = tf.nn.softplus(z, **kwargs)

	elif function == 'tanh':
		output = tf.nn.tanh(z, **kwargs)

	elif function == 'leaky_relu':
		if 'leakiness' in kwargs.keys():
			leakiness = kwargs['leakiness']
		else:
			leakiness = 0.1
		output = tf.nn.relu(z) - leakiness*tf.nn.relu(-z)
		
	elif function == 'prelu':
		output = tf.nn.relu(z) - kwargs['alpha']*tf.nn.relu(-z)

	elif function == 'exp':
		output = tf.exp(z)
	return output


