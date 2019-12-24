import tensorflow as tf
from .base import BaseLayer


__all__ = [
	"ReshapeLayer"
]

class ReshapeLayer(BaseLayer):
	def __init__(self, incoming, shape=[], **kwargs):
		
		self.shape = shape
		if not self.shape:
			input_dim = 1
			for dim in incoming.get_output_shape():
				if dim.value:
					input_dim *= dim.value
			self.shape = [-1, input_dim]
			
		self.incoming_shape = incoming.get_output_shape()
		
		self.output = tf.reshape(incoming.get_output(), self.shape, **kwargs)
		
		self.output_shape = self.output.get_shape()
	
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
