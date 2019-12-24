from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .base import BaseLayer
from ..utils import Variable
from .. import init

__all__ = [
	"Conv1DLayer", 
	"Conv2DLayer",
	"TransposeConv1DLayer",
	"TransposeConv2DLayer",
	"StochasticConv1DLayer", 
	"StochasticConv2DLayer",
]



class Conv1DLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, num_filters, W=[], b=None,
				  strides=[], padding=[], **kwargs):

		self.filter_size = filter_size
		self.num_filters = num_filters
		
		dim = incoming.get_output_shape()[3].value
		shape = [filter_size, 1, dim, num_filters]
		self.shape = shape

		if not W:
			self.W = Variable(var=init.HeUniform(**kwargs), shape=shape)
		else:
			self.W = Variable(var=W, shape=shape, **kwargs)
			
		if not strides:
			self.strides = [1, 1, 1, 1]
		else:
			self.strides = [1, strides, 1, 1]

		self.padding = padding
		if not padding:
			self.padding = 'VALID'
			
		# input data shape
		self.incoming_shape = incoming.get_output_shape()
		
		# output of convolution
		self.output = tf.nn.conv2d( input=incoming.get_output(), 
									filter=self.W.get_variable(), 
									strides=self.strides, 
									padding=self.padding)

		if b is None:
			self.b = []
		else:
			if not b:
				self.b = Variable(var=init.Constant(0.05, kwargs), shape=[num_units])
			else:
				self.b = Variable(var=b, shape=[num_units], **kwargs)
			self.output = tf.nn.bias_add(self.output,self.b.get_variable())

		# shape of the output
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return self.W
	
	def set_trainable(self, status):
		self.W.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.W.set_l1_regularize(status)    
		
	def set_l2_regularize(self, status):
		self.W.set_l2_regularize(status)    
		
	def is_trainable(self):
		return self.W.is_trainable()
		
	def is_l1_regularize(self):
		return self.W.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.W.is_l2_regularize()  
		


class Conv2DLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, num_filters, W=[],
				  strides=[], padding=[], **kwargs):

		self.filter_size = filter_size
		self.num_filters = num_filters
		
		dim = incoming.get_output_shape()[3].value

		if not isinstance(filter_size, (list, tuple)):
			self.shape = [filter_size, filter_size, dim, num_filters]
		else:
			self.shape = [filter_size[0], filter_size[1], dim, num_filters]

		if not W:
			self.W = Variable(var=init.HeUniform(**kwargs), shape=self.shape, **kwargs)
		else:
			self.W = Variable(var=W, shape=self.shape, **kwargs)
			

		if not strides:		
			self.strides = [1, 1, 1, 1]
		else:
			if not isinstance(strides, (list, tuple)):
				self.strides = [1, strides, strides, 1]
			else:
				self.strides = [1, strides[0], strides[1], 1]

		self.padding = padding
		if not padding:
			self.padding = 'VALID'
			
		# input data shape
		self.incoming_shape = incoming.get_output_shape()
		
		# output of convolution
		self.output = tf.nn.conv2d( input=incoming.get_output(), 
									filter=self.W.get_variable(), 
									strides=self.strides, 
									padding=self.padding)
		# shape of the output
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return self.W
	
	def set_trainable(self, status):
		self.W.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.W.set_l1_regularize(status)    
		
	def set_l2_regularize(self, status):
		self.W.set_l2_regularize(status)    
		
	def is_trainable(self):
		return self.W.is_trainable()
		
	def is_l1_regularize(self):
		return self.W.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.W.is_l2_regularize()  
		



#---------------------------------------------------------------------------


def deconv_output_length(input_length, filter_size, padding, stride):
	if input_length is None:
		return None
	input_length *= stride
	if padding.upper() == 'VALID':
		input_length += max(filter_size - stride, 0)
	elif padding.upper() == 'FULL':
		input_length -= (stride + filter_size - 2)
	return input_length



class TransposeConv1DLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, num_filters, W=[],
				  strides=[], padding=[], **kwargs):

		self.filter_size = filter_size
		self.num_filters = num_filters
		
		dim = incoming.get_output_shape()[3].value
		shape = [filter_size, 1, num_filters, dim]
		self.shape = shape

		if not W:
			self.W = Variable(var=init.HeUniform(**kwargs), shape=shape, **kwargs)
		else:
			self.W = Variable(var=W, shape=shape, **kwargs)
			
		if not strides:		
			self.strides = [1, 1, 1, 1]
		else:
			if not isinstance(strides, (list, tuple)):
				self.strides = [1, strides, 1, 1]
			else:
				self.strides = strides

		self.padding = padding
		if not padding:
			self.padding = 'VALID'
			
		# input data shape
		inputs_shape = incoming.get_output_shape().as_list()#array_ops.shape(Z_reshape)
		height, dim = inputs_shape[1], inputs_shape[3]

		out_height = deconv_output_length(height,
											kernel_h,
											self.padding,
											stride)

		batch_size = tf.shape(incoming.get_output())[0]

		self.incoming_shape = incoming.get_output_shape()
		self.output_shape = (batch_size, out_height, 1, num_filters)
		self.num_filters = num_filters
		
		self.incoming_shape = incoming.get_output_shape()
		X = tf.nn.conv2d_transpose(incoming.get_output(), 
									filter=self.W.get_variable(), 
									output_shape=self.output_shape,
									strides=self.strides, 
									padding=self.padding)
		
		self.output = tf.reshape(X, [-1, out_height, 1, num_filters])

	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return self.W
	
	def set_trainable(self, status):
		self.W.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.W.set_l1_regularize(status)    
		
	def set_l2_regularize(self, status):
		self.W.set_l2_regularize(status)    
		
	def is_trainable(self):
		return self.W.is_trainable()
		
	def is_l1_regularize(self):
		return self.W.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.W.is_l2_regularize()  




class TransposeConv2DLayer(BaseLayer):
	"""1D convolutional layer"""

	def __init__(self, incoming, filter_size, num_filters, W=[],
				  strides=(1,1), padding='SAME', **kwargs):

		self.padding = padding.upper()
		
		inputs_shape = incoming.get_output_shape().as_list()#array_ops.shape(Z_reshape)

		height, width, dim = inputs_shape[1], inputs_shape[2], inputs_shape[3]

		if not isinstance(filter_size, (list, tuple)):
			self.filter_size = [filter_size, filter_size, num_filters, dim]
		else:
			self.filter_size = [filter_size[0], filter_size[1], num_filters, dim]
		_, kernel_h, kernel_w, _ = self.filter_size

		if not strides:		
			self.strides = [1, 1, 1, 1]
		else:
			if not isinstance(strides, (list, tuple)):
				self.strides = [1, strides, strides, 1]
			else:
				self.strides = [1, strides[0], strides[1], 1]
		_, stride_h, stride_w, _ = self.strides

		out_height = deconv_output_length(height,
											kernel_h,
											self.padding,
											stride_h)

		out_width = deconv_output_length(width,
									   kernel_w,
									   self.padding,
									   stride_w)

		batch_size = tf.shape(incoming.get_output())[0]

		#output_shape = tf.stack([batch_size, 40, 40, 32])
		self.output_shape = (batch_size, out_height, out_width, num_filters)
		self.num_filters = num_filters
		
		if not W:
			self.W = Variable(var=init.HeUniform(**kwargs), shape=self.filter_size, **kwargs)
		else:
			self.W = Variable(var=W, shape=self.filter_size, **kwargs)
			
		# input data shape
		self.incoming_shape = incoming.get_output_shape()
		X = tf.nn.conv2d_transpose(incoming.get_output(), 
									filter=self.W.get_variable(), 
									output_shape=self.output_shape,
									strides=self.strides, 
									padding=self.padding)

		self.output = tf.reshape(X, [-1, out_height, out_width, num_filters])


	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return self.W
	
	def set_trainable(self, status):
		self.W.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.W.set_l1_regularize(status)    
		
	def set_l2_regularize(self, status):
		self.W.set_l2_regularize(status)    
		
	def is_trainable(self):
		return self.W.is_trainable()
		
	def is_l1_regularize(self):
		return self.W.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.W.is_l2_regularize()  

		
		
class StochasticConv1DLayer(BaseLayer):
	"""1D convolutional layer with stochastic parameters"""

	def __init__(self, incoming, filter_size, num_filters, W=[],
				  strides=[], padding=[], **kwargs):

		self.filter_size = filter_size
		self.num_filters = num_filters
		
		dim = incoming.get_output_shape()[3].value
		shape = [filter_size, 1, dim, num_filters]
		self.shape = shape

		if not W:
			self.W_mu = Variable(var=init.HeUniform(), shape=shape)
			self.W_sigma = Variable(var=init.HeUniform(), shape=shape)
		else:
			self.W_mu = Variable(var=W, shape=shape)
			self.W_sigma = Variable(var=W, shape=shape)
		z = tf.random_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32) 
		self.W = self.W_mu.get_variable() + tf.multiply(tf.exp(0.5 * self.W_sigma.get_variable()), z)

		if not strides:
			self.strides = [1, 1, 1, 1]
		else:
			self.strides = [1, strides, 1, 1]

		self.padding = padding
		if not padding:
			self.padding = 'VALID'
			
		# input data shape
		self.incoming_shape = incoming.get_output_shape()
		
		# output of convolution
		self.output = tf.nn.conv2d( input=incoming.get_output(), 
									filter=self.W, 
									strides=self.strides, 
									padding=self.padding)

		# shape of the output
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
		

class StochasticConv2DLayer(BaseLayer):
	"""1D convolutional layer with stochastic parameters"""

	def __init__(self, incoming, filter_size, num_filters, W=[],
				  strides=[], padding=[], **kwargs):

		self.filter_size = filter_size
		self.num_filters = num_filters
		
		dim = incoming.get_output_shape()[3].value

		if not isinstance(filter_size, (list, tuple)):
			self.shape = [filter_size, filter_size, dim, num_filters]
		else:
			self.shape = [filter_size[0], filter_size[1], dim, num_filters]

		if not W:
			self.W_mu = Variable(var=init.HeUniform(), shape=shape)
			self.W_sigma = Variable(var=init.HeUniform(), shape=shape)
		else:
			self.W_mu = Variable(var=W, shape=shape)
			self.W_sigma = Variable(var=W, shape=shape)
			
		z = tf.random_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32) 
		self.W = self.W_mu.get_variable() + tf.multiply(tf.exp(0.5 * self.W_sigma.get_variable()), z)

		if not strides:		
			self.strides = [1, 1, 1, 1]
		else:
			if not isinstance(strides, (list, tuple)):
				self.strides = [1, strides, strides, 1]
			else:
				self.strides = [1, strides[0], strides[1], 1]

		self.padding = padding
		if not padding:
			self.padding = 'VALID'
			
		# input data shape
		self.incoming_shape = incoming.get_output_shape()
		
		# output of convolution
		self.output = tf.nn.conv2d( input=incoming.get_output(), 
									filter=self.W, 
									strides=self.strides, 
									padding=self.padding)
		# shape of the output
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
		
		
