
__all__ = [
	"BaseLayer",
	"InputLayer",
]


class BaseLayer(object):
	"""Base class for neural network layers."""
	def __init__(self, name=None):
		self.name = name

	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		raise NotImplementedError()
		
	def get_output_shape(self):
		raise NotImplementedError()



class InputLayer(BaseLayer):
	"""Input layer to feed in data"""
	def __init__(self, incoming, **kwargs):
		
		self.incoming_shape = incoming.get_shape()
		self.output = incoming        
		self.output_shape = incoming.get_shape()
		
	def get_input_shape(self):
		return self.incoming_shape
	
	def get_output(self):
		return self.output
	
	def get_output_shape(self):
		return self.output_shape    

