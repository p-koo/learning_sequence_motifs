from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .base import BaseLayer
from ..utils import Variable
from .shape import ReshapeLayer
from .. import init


__all__ = [
	"DenseLayer",
	"StochasticDenseLayer"
]

	

class DenseLayer(BaseLayer):
	"""Fully-connected layer"""

	def __init__(self, incoming, num_units, W=[], b=[], **kwargs):

		self.num_units = num_units
		
		if len(incoming.get_output_shape()) > 2:
			incoming = ReshapeLayer(incoming)
			
		num_inputs = incoming.get_output_shape()[1].value
		shape = [num_inputs, num_units]
		self.shape = shape
		self.incoming_shape = incoming.get_output_shape()
		
		
		if not W:
			self.W = Variable(var=init.HeUniform(**kwargs), shape=shape, **kwargs)
		else:
			self.W = Variable(var=W, shape=shape, **kwargs)
			
		if b is None:
			self.b = []
		else:
			if not b:
				self.b = Variable(var=init.Constant(0.05), shape=[num_units], **kwargs)
			else:
				self.b = Variable(var=b, shape=[num_units], **kwargs)
			
		self.output = tf.matmul(incoming.get_output(), self.W.get_variable())
		if self.b:
			self.output = tf.nn.bias_add(self.output,self.b.get_variable())
			
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		if self.b:
			return [self.W, self.b]
		else:
			return self.W
	
	def set_trainable(self, status):
		self.W.set_trainable(status)
		if self.b:
			self.b.set_trainable(status)
			
	def set_l1_regularize(self, status):
		self.W.set_l1_regularize(status)    
		if self.b:
			self.b.set_l1_regularize(status)
		
	def set_l2_regularize(self, status):
		self.W.set_l2_regularize(status)    
		if self.b:
			self.b.set_l2_regularize(status)
	
	def is_trainable(self):
		return self.W.is_trainable()
		
	def is_l1_regularize(self):
		return self.W.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.W.is_l2_regularize()  
	


class StochasticDenseLayer(BaseLayer):
	"""Fully-connected layer with stochastic parameters"""

	def __init__(self, incoming, num_units, W=[],  **kwargs):

		self.num_units = num_units
		
		if len(incoming.get_output_shape()) > 2:
			incoming = ReshapeLayer(incoming)
			
		num_inputs = incoming.get_output_shape()[1].value
		shape = [num_inputs, num_units]
		self.shape = shape
		self.incoming_shape = incoming.get_output_shape()
		
		if not W:
			self.W_mu = Variable(var=init.HeUniform(), shape=shape)
			self.W_sigma = Variable(var=init.HeUniform(), shape=shape)
		else:
			self.W_mu = Variable(var=W, shape=shape)
			self.W_sigma = Variable(var=W, shape=shape)

		z = tf.random_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32) 
		self.W = self.W_mu.get_variable() + tf.multiply(tf.exp(0.5 * self.W_sigma.get_variable()), z)

		self.output = tf.matmul(incoming.get_output(), self.W)
			
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self, shape=False):
		return [self.W_mu.get_variable(), self.W_sigma.get_variable()]

	def set_trainable(self, status):
		self.W_mu.set_trainable(status)
		self.W_sigma.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.W_mu.set_l1_regularize(status)    
		self.W_sigma.set_l1_regularize(status)    
		
	def set_l2_regularize(self, status):
		self.W_mu.set_l2_regularize(status)    
		self.W_sigma.set_l2_regularize(status)    
		
	def is_trainable(self):
		return self.W_mu.is_trainable()
		
	def is_l1_regularize(self):
		return self.W_mu.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.W_mu.is_l2_regularize()  
		
		
