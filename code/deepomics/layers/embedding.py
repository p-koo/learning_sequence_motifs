from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .base import BaseLayer
from ..utils import Variable
from .. import init


__all__ = [
	"EmbeddingLayer",
]


class EmbeddingLayer(BaseLayer):
	def __init__(self, incoming, vocab_size, embedding_size, max_norm=None, W=None, b=None, **kwargs):

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size

		if not W:
			self.W = Variable(var=init.Uniform(minval=-1, maxval=1), shape=[vocab_size, embedding_size], **kwargs)
		else:
			self.W = Variable(var=W, shape=[vocab_size, embedding_size], **kwargs)
		self.b = Variable(var=init.GlorotUniform(), shape=[embedding_size])


		inputs_index = tf.argmax(incoming.get_output(), axis=2)
		self.output = tf.nn.bias_add(tf.nn.embedding_lookup(self.W.get_variable(), inputs_index, max_norm=max_norm), self.b.get_variable())

		self.incoming_shape = incoming.get_output_shape()
		
		# shape of the output
		self.output_shape = self.output.get_shape()


	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape
	
	def get_variable(self):
		return [self.W, self.b]
	
	def set_trainable(self, status):
		self.W.set_trainable(status)
		self.b.set_trainable(status)
		
	def set_l1_regularize(self, status):
		self.W.set_l1_regularize(status)   
		self.b.set_l1_regularize(status)    
		
	def set_l2_regularize(self, status):
		self.W.set_l2_regularize(status)  
		self.b.set_l2_regularize(status)    
		
	def is_trainable(self):
		return self.W.is_trainable()
		
	def is_l1_regularize(self):
		return self.W.is_l1_regularize()    
		
	def is_l2_regularize(self):
		return self.W.is_l2_regularize()  
		
