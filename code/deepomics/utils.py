from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from .init import *



__all__ = [
	"placeholder",
	"Variable",
	"make_directory",
	"normalize_pwm",
	"meme_generate"
]


def initialize_session(placeholders=None):
	# run session
	sess = tf.Session()

	# initialize variables
	if placeholders is None:
		sess.run(tf.global_variables_initializer())

	else:

		# below fixes a bug in older versions of tensorflow (not needed in version r1.0)
		if 'is_training' in placeholders:
			sess.run(tf.global_variables_initializer(), feed_dict={placeholders['is_training']: True})
		else:
			sess.run(tf.global_variables_initializer())
	return sess


def placeholder(shape, dtype=tf.float32, name=None):
	return tf.placeholder(dtype=dtype, shape=shape, name=name)


class Variable():
	def __init__(self, var, shape, **kwargs):

		self.l1_regularize = True
		if 'l1' in kwargs.keys():
			self.l1_regularize = kwargs['l1']

		self.l2_regularize = True
		if 'l2' in kwargs.keys():
			self.l2_regularize = kwargs['l2']

		if 'regularize' in kwargs.keys():
			if not kwargs['regularize']:
				self.l1_regularize = False
				self.l2_regularize = False

		self.trainable = True
		if 'trainable' in kwargs.keys():
			self.trainable = kwargs['trainable']

		self.name = None
		if 'name' in kwargs.keys():
			self.name = kwargs['name']

		self.shape = shape
		variable = var(shape)
		self.variable = tf.Variable(variable)


	def get_variable(self):
		return self.variable

	def get_shape(self):
		return self.shape

	def set_l1_regularize(self, status):
		self.l1_regularize = status

	def set_l2_regularize(self, status):
		self.l2_regularize = status

	def set_trainable(self, status):
		self.trainable = status

	def is_l1_regularize(self):
		return self.l1_regularize

	def is_l2_regularize(self):
		return self.l2_regularize

	def is_trainable(self):
		return self.trainable


def make_directory(path, foldername, verbose=1):
	"""make a directory"""

	if not os.path.isdir(path):
		os.mkdir(path)
		print("making directory: " + path)

	outdir = os.path.join(path, foldername)
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		print("making directory: " + outdir)
	return outdir


def normalize_pwm(pwm, factor=None, max=None):

	if not max:
		max = np.max(np.abs(pwm))
	pwm = pwm/max
	if factor:
		pwm = np.exp(pwm*factor)
	norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
	return pwm/norm


def meme_generate(W, output_file='meme.txt', prefix='filter', factor=None):

	# background frequency
	nt_freqs = [1./4 for i in range(4)]

	# open file for writing
	f = open(output_file, 'w')

	# print intro material
	f.write('MEME version 4\n')
	f.write('\n')
	f.write('ALPHABET= ACGT\n')
	f.write('\n')
	f.write('Background letter frequencies:\n')
	f.write('A %.4f C %.4f G %.4f T %.4f \n' % tuple(nt_freqs))
	f.write('\n')

	W = np.squeeze(W.transpose([3, 2, 0, 1]))
	for j in range(len(W)):
		if factor:
			pwm = normalize_pwm(W[j], factor=factor)
		else:
			pwm = W[j]
		f.write('MOTIF %s%d \n' % (prefix, j))
		f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (pwm.shape[1], pwm.shape[1]))
		for i in range(pwm.shape[1]):
			f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[:,i]))
		f.write('\n')

	f.close()
