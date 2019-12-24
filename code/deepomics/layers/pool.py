import tensorflow as tf
from .base import BaseLayer

		
__all__ = [
	"GlobalPoolLayer",
	"MaxPool1DLayer",
	"MaxPool2DLayer",
	"MeanPool1DLayer",
	"MeanPool2DLayer",
	"Upsample1DLayer",
	"Upsample2DLayer",
]


class GlobalPoolLayer(BaseLayer):
	def __init__(self, incoming, func='max', **kwargs):
				
		self.incoming_shape = incoming.get_output_shape()
		
		#self.output = tf.nn.global_pool(incoming.get_output(), 
		#								func=func, 
		#								**kwargs)
		pool_size = [1, self.incoming_shape[1], self.incoming_shape[2], 1]
		strides = [1, self.incoming_shape[1], self.incoming_shape[2], 1]
		padding = 'SAME'
		if func == 'max':
			self.output = tf.nn.max_pool(incoming.get_output(), 
									ksize=pool_size, 
									strides=strides, 
									padding=padding, 
									**kwargs)
		elif func == 'mean':
			self.output = tf.nn.avg_pool(incoming.get_output(), 
									ksize=pool_size, 
									strides=strides, 
									padding=padding, 
									**kwargs)
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape


class MaxPool1DLayer(BaseLayer):
	def __init__(self, incoming, pool_size, strides=[], padding=[], **kwargs):
		
		self.pool_size = [1, pool_size, 1, 1]

		if not strides:
			strides = pool_size
		self.strides = [1, strides, 1, 1]
		
		self.padding = padding
		if not self.padding:
			self.padding = 'SAME'
		
		self.incoming_shape = incoming.get_output_shape()
		
		self.output = tf.nn.max_pool(incoming.get_output(), 
									ksize=self.pool_size, 
									strides=self.strides, 
									padding=self.padding, 
									**kwargs)
		
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
		
		
class MaxPool2DLayer(BaseLayer):
	def __init__(self, incoming, pool_size, strides=[], padding=[], **kwargs):
		if not isinstance(pool_size, (list, tuple)):
			self.pool_size = [1, pool_size, pool_size, 1]
		else:
			self.pool_size = [1, pool_size[0], pool_size[1], 1]

		if not strides:		
			self.strides = self.pool_size
		else:
			if not isinstance(strides, (list, tuple)):
				self.strides = [1, strides, strides, 1]
			else:
				self.strides = [1, strides[0], strides[1], 1]
		
		self.padding = padding
		if not self.padding:
			self.padding = 'VALID'
		
		self.incoming_shape = incoming.get_output_shape()
		
		self.output = tf.nn.max_pool(incoming.get_output(), 
									ksize=self.pool_size, 
									strides=self.strides, 
									padding=self.padding, 
									**kwargs)
		
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
		


class MeanPool1DLayer(BaseLayer):
	def __init__(self, incoming, pool_size, strides=[], padding=[], **kwargs):
		
		self.pool_size = [1, pool_size, 1, 1]

		if not strides:
			strides = pool_size
		self.strides = [1, strides, 1, 1]
		
		self.padding = padding
		if not self.padding:
			self.padding = 'SAME'
		
		self.incoming_shape = incoming.get_output_shape()
		
		self.output = tf.nn.avg_pool(incoming.get_output(), 
									ksize=self.pool_size, 
									strides=self.strides, 
									padding=self.padding, 
									**kwargs)
		
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
		
		
class MeanPool2DLayer(BaseLayer):
	def __init__(self, incoming, pool_size, strides=[], padding=[], **kwargs):
		if not isinstance(pool_size, (list, tuple)):
			self.pool_size = [1, pool_size, pool_size, 1]
		else:
			self.pool_size = [1, pool_size[0], pool_size[1], 1]

		if not strides:		
			self.strides = self.pool_size
		else:
			if not isinstance(strides, (list, tuple)):
				self.strides = [1, strides, strides, 1]
			else:
				self.strides = [1, strides[0], strides[1], 1]
		
		self.padding = padding
		if not self.padding:
			self.padding = 'VALID'
		
		self.incoming_shape = incoming.get_output_shape()
		
		self.output = tf.nn.avg_pool(incoming.get_output(), 
									ksize=self.pool_size, 
									strides=self.strides, 
									padding=self.padding, 
									**kwargs)
		
		self.output_shape = self.output.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape


class Upsample1DLayer(BaseLayer):
	def __init__(self, incoming, pool_size, **kwargs):
		
		self.incoming_shape = self.incoming.get_output_shape().as_list()
		self.output_shape = tf.stack([self.incoming_shape[0], self.incoming_shape[1]*pool_size, self.incoming_shape[2], self.incoming_shape[3]])

		self.output = tf.image.resize_images(input_tensor, self.output_shape)
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape


class Upsample2DLayer(BaseLayer):
	def __init__(self, incoming, pool_size,**kwargs):
		self.incoming_shape = self.incoming.get_output_shape().as_list()
		self.output_shape = tf.stack([self.incoming_shape[0], self.incoming_shape[1]*pool_size, self.incoming_shape[2]*pool_size, self.incoming_shape[3]])

		self.output = tf.image.resize_images(input_tensor, self.output_shape)

		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape