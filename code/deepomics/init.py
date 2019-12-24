
import tensorflow as tf
import numpy as np


__all__ = [
	"Constant",
	"Uniform",
	"Normal",
	"TruncatedNormal",
	"GlorotUniform",
	"GlorotNormal",
	"HeUniform",
	"HeNormal",
	"Orthogonal"
]

class Initializer(object):
	"""Base class for parameter tensor initializers."""
	def __call__(self, shape):
		return self.generate(shape)

	def generate(self, shape):
		raise NotImplementedError()


class Constant(Initializer):
	"""Consant"""
	def __init__(self, value=0.05, dtype=tf.float32, **kwargs):
		self.value = value
		self.dtype = dtype
		self.kwargs = kwargs
		
	def generate(self, shape):
		return tf.constant(shape=shape, value=self.value, dtype=self.dtype, **self.kwargs)


class Uniform(Initializer):
	"""Uniform random number"""
	def __init__(self, minval=-0.1, maxval=0.1, dtype=tf.float32, **kwargs):
		self.minval = minval
		self.maxval = maxval
		self.dtype = dtype
		self.kwargs = kwargs
		
	def generate(self, shape):
		return tf.random_uniform(shape=shape, minval=self.minval, maxval=self.maxval, dtype=self.dtype, **self.kwargs)


class Normal(Initializer):
	"""Normal distribution"""
	def __init__(self, mean=0.0, stddev=0.1, dtype=tf.float32, **kwargs):
		self.mean = mean
		self.stddev = stddev
		self.dtype = dtype
		self.kwargs = kwargs
		
	def generate(self, shape):
		return tf.random_normal(shape=shape, mean=self.mean, stddev=self.stddev, dtype=self.dtype, **self.kwargs)


class TruncatedNormal(Initializer):
	"""Truncated normal distribution"""
	def __init__(self, mean=0.0, stddev=0.1, dtype=tf.float32, **kwargs):
		self.mean = mean
		self.stddev = stddev
		self.dtype = dtype
		self.kwargs = kwargs
		
	def generate(self, shape):
		return tf.truncated_normal(shape=shape, mean=self.mean, stddev=self.stddev, dtype=self.dtype, **self.kwargs)


def get_fans(shape):
	"""Get number of input neurons (fan_in) and output neurons (fan_out)"""
	if len(shape) == 2:
		fan_in = shape[0]
		fan_out = shape[1]

	elif len(shape) == 4:
		receptive_field_size = np.prod(shape[:2])
		fan_in = shape[-2]*receptive_field_size
		fan_out = shape[-1]*receptive_field_size
	else:
		# no specific assumptions
		fan_in = np.sqrt(np.prod(shape))
		fan_out = np.sqrt(np.prod(shape))
	return fan_in, fan_out



class GlorotUniform(Initializer):
	"""
	Glorot Uniform init
	
	References
	----------
	.. [1] Xavier Glorot and Yoshua Bengio (2010):
		   Understanding the difficulty of training deep feedforward neural
		   networks. International conference on artificial intelligence and
		   statistics.
	"""
	def __init__(self, dtype=tf.float32, **kwargs):
		self.dtype = dtype
		self.kwargs = kwargs
		
	def generate(self, shape):
		fan_in, fan_out = get_fans(shape)
		stddev = np.sqrt(6./(fan_in + fan_out))
		return tf.random_uniform(shape=shape, minval=-stddev, maxval=stddev, dtype=self.dtype, **self.kwargs)


class GlorotNormal(Initializer):
	"""
	Glorot Normal init

	References
	----------
	.. [1] Xavier Glorot and Yoshua Bengio (2010):
		   Understanding the difficulty of training deep feedforward neural
		   networks. International conference on artificial intelligence and
		   statistics.
	"""
	def __init__(self, mean=0.0, dtype=tf.float32, **kwargs):
		self.mean = mean
		self.dtype = dtype
		self.kwargs = kwargs
		
	def generate(self, shape):
		fan_in, fan_out = get_fans(shape)
		stddev = np.sqrt(2./(fan_in + fan_out))
		return tf.truncated_normal(shape=shape, mean=self.mean, stddev=stddev, dtype=self.dtype, **self.kwargs)



class HeUniform(Initializer):
	"""
	He Uniform init

	References
	----------
	.. [1] Kaiming He et al. (2015):
		   Delving deep into rectifiers: Surpassing human-level performance on
		   imagenet classification. arXiv preprint arXiv:1502.01852.

	"""
	def __init__(self, dtype=tf.float32, **kwargs):
		self.dtype = dtype
		self.kwargs = kwargs

	def generate(self, shape):
		fan_in, fan_out = get_fans(shape)
		stddev =  np.sqrt(6./fan_in)		

		return tf.random_uniform(shape=shape, minval=-stddev, maxval=stddev, dtype=self.dtype, **self.kwargs)


class HeNormal(Initializer):
	"""
	He Normal init

	References
	----------
	.. [1] Kaiming He et al. (2015):
		   Delving deep into rectifiers: Surpassing human-level performance on
		   imagenet classification. arXiv preprint arXiv:1502.01852.

	"""
	def __init__(self, mean=0.0, dtype=tf.float32, **kwargs):
		self.mean = mean
		self.dtype = dtype
		self.kwargs = kwargs
		
	def generate(self, shape):
		fan_in, fan_out = get_fans(shape)
		stddev = np.sqrt(2./fan_in)

		return tf.truncated_normal(shape=shape, mean=self.mean, stddev=stddev, dtype=self.dtype, **self.kwargs)


class Orthogonal(Initializer):
	"""
	Orthogonal init
	"""
	def __init__(self, gain=1.1, dtype=tf.float32, **kwargs):
		self.gain = gain
		self.dtype = dtype
		self.kwargs = kwargs
		
	def generate(self, shape):
		flat_shape = (shape[0], np.prod(shape[1:]))
		a = np.random.normal(0.0, 1.0, flat_shape)
		u, _, v = np.linalg.svd(a, full_matrices=False)

		# pick the one with the correct shape
		q = u if u.shape == flat_shape else v
		q = q.reshape(shape)
		return tf.cast(self.gain*q[:shape[0],:shape[1]], dtype=dtype)
