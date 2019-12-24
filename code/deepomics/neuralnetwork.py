from __future__ import print_function
import os, sys, time
import numpy as np
from six.moves import cPickle
import tensorflow as tf
from deepomics import optimize, metrics, utils
from deepomics.neuralbuild import NeuralBuild


__all__ = [
	"NeuralNet",
	"NeuralTrainer",
	"MonitorPerformance",
	"BatchGenerator"
]

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
	return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

#------------------------------------------------------------------------------------------
# Neural Network model class
#------------------------------------------------------------------------------------------

class NeuralNet:
	"""Class to build a neural network and perform basic functions."""

	def __init__(self, seed=None, network=[], placeholders=[], feed_dict={}, optimization={}):

		self.network = network
		self.placeholders = placeholders
		self.feed_dict = feed_dict
		self.optimization = optimization

		self.predictions = []
		self.targets = []
		self.mean_loss = []
		self.sample_loss = []
		self.updates = []
		self.train_step = []
		self.metric = []
		self.seed = seed


	def build_layers(self, model_layers, optimization=None, method=None, supervised=True, use_scope=True, reset=True):

		if reset:
			tf.reset_default_graph()

		if use_scope:
			from deepomics.neuralbuild_scope import NeuralBuild
		else:
			from deepomics.neuralbuild import NeuralBuild

		nnbuild = NeuralBuild(self.seed)

		if method == 'guided':
			g = tf.get_default_graph()
			with g.gradient_override_map({'Relu': 'GuidedRelu'}):
				self.network, self.placeholders, self.feed_dict = nnbuild.build_layers(model_layers, supervised)
		else:
			self.network, self.placeholders, self.feed_dict = nnbuild.build_layers(model_layers, supervised)
		self.build_optimizer(optimization, supervised)
		self.train_metric()


	def add_placeholder(self, variable, name, value):
		self.feed_dict[name] = value
		self.placeholders[name] = variable


	def build_optimizer(self, optimization=None, supervised=False):
		if optimization is None:
			optimization = self.optimization
		else:
			self.optimization = optimization

		if not supervised:
			output_layer = 'X'
		else:
			output_layer = self.network.keys()[-1]

		# get predictions
		self.predictions = self.network[output_layer].get_output()
		self.targets = self.placeholders['targets']

		self.sample_loss, self.regularization = optimize.build_loss(self.network, self.predictions, self.targets, optimization)
		self.mean_loss = tf.reduce_mean(self.sample_loss+self.regularization)

		# setup optimizer
		self.updates = optimize.build_updates(self.optimization)

		# get list of trainable parameters (default is trainable)
		trainable_params = self.get_trainable_parameters()
		self.train_step = self.updates.minimize(self.mean_loss, var_list=trainable_params)


	def train_metric(self):
		"""metric to monitor performance during training"""

		# categorical cross entropy (objective for softmax classification)
		if self.optimization['objective'] == 'categorical':
			predictions = tf.argmax(self.predictions, axis=1)
			targets = tf.argmax(self.placeholders['targets'], axis=1)
			self.metric = tf.reduce_mean(tf.cast(tf.equal(predictions, targets), tf.float32))

		# binary cross entropy (objective for binary classification)
		elif self.optimization['objective'] == 'binary':
			predictions = tf.cast(tf.greater_equal(self.predictions, 0.5), tf.float32)
			self.metric = tf.reduce_mean(tf.cast(tf.equal(predictions, self.placeholders['targets']), tf.float32))

		# mean squared error (objective for regression)
		elif self.optimization['objective'] == 'squared_error':
			self.metric = tf.reduce_mean(tf.square(self.predictions - self.placeholders['targets']))

		# variational lower bound -- currently just printing squared error
		else:
			self.metric = tf.reduce_mean(tf.square(self.predictions - self.placeholders['targets']))


	def get_trainable_parameters(self):
		"""get all trainable parameters (tensorflow variables) in network"""

		params = []
		for layer in self.network:
			if hasattr(self.network[layer], 'is_trainable'):
				if self.network[layer].is_trainable():
					variables = self.network[layer].get_variable()
					if isinstance(variables, list):
						for var in variables:
							params.append(var.get_variable())
					else:
						params.append(variables.get_variable())
		return params


	def inspect_layers(self):
		"""print(each layer type and parameters"""

		print('----------------------------------------------------------------------------')
		print('Network architecture:')
		print('----------------------------------------------------------------------------')
		counter = 1
		for layer in self.network:
			output_shape = self.network[layer].get_output_shape()

			print('layer'+str(counter) + ': ' + layer )
			print(output_shape)
			counter += 1
		print('----------------------------------------------------------------------------')


	def get_activations(self, sess, feed_X, layer='output'):
		"""get the real-valued feature maps of a given convolutional layer"""

		return sess.run(self.network[layer].get_output(), feed_dict=feed_X)


	def save_model_parameters(self, sess, file_path='model.ckpt', verbose=1):
		"""save model parameters to a file"""
		if verbose:
			print("  saving model to: ", file_path)
		saver = tf.train.Saver()
		saver.save(sess, save_path=file_path)


	def load_model_parameters(self, sess, file_path='model.ckpt', verbose=1):
		"""initialize network with all_param_values"""
		if verbose:
			print("loading model from: ", file_path)
		saver = tf.train.Saver()
		saver.restore(sess, file_path)


	def get_parameters(self, sess, layer=[]):
		"""return all the parameters of the network"""

		layer_params = []
		if layer:
			variables = self.network[layer].get_variable()
			if isinstance(variables, list):
				params = []
				for var in variables:
					params.append(sess.run(var.get_variable()))
				layer_params.append(params)
			else:
				layer_params.append(sess.run(variables.get_variable()))
		else:

			for layer in self.network:
				if hasattr(self.network[layer], 'is_trainable'):
					if self.network[layer].is_trainable():
						variables = self.network[layer].get_variable()
						if isinstance(variables, list):
							for var in variables:
								layer_params.append(sess.run(var.get_variable()))
						else:
							layer_params.append(sess.run(variables.get_variable()))
		return layer_params


	def calculate_saliency(self, sess, y, dx, feed_dict, class_index=None, func=tf.reduce_max):
		if class_index is None:
			dy = func(y, axis=1)
		else:
			if len(y.get_shape()) == 4:
				dy = func(y[:,:,:,class_index], axis=1)
			else:
				dy = y[:,class_index] #  tf.sign(y[:,class_index])*tf.square(y[:,class_index])
		return sess.run(tf.gradients(dy, dx), feed_dict=feed_dict)


	def stochastic_saliency(self, sess, X, y, dx, stochastic_feed, num_average=200, threshold=1.0, class_index=None, func=tf.reduce_max):

		def choose_sign(x,y):
			y2 = np.copy(y)
			y2[y2<0] = 0
			pos = np.sum(x*y2)

			y2 = np.copy(y)
			y2[y2>0] = 0
			neg = np.sum(-x*y2)
			if neg > pos:
				return -1
			else:
				return 1

		augment = np.multiply(np.ones((num_average,1,1,1)), X)
		stochastic_feed.update({dx: augment})

		if class_index is None:
			dy = func(y, axis=1)
		else:
			if len(y.get_shape()) == 4:
				dy = func(y[:,:,:,class_index], axis=1)
			else:
				dy = y[:,class_index] #  tf.sign(y[:,class_index])*tf.square(y[:,class_index])
		val = sess.run([tf.gradients(dy, dx), dy], feed_dict=stochastic_feed)
		dydx = val[0][0]
		pred = val[1]

		# average based on weights
		saliency_ave = 0
		norm = 0
		counter = 0
		if threshold is not None:
			indices = np.where(pred > threshold)[0]
		else:
			indices = range(num_average)
		for k in indices:
			sign = choose_sign(X[0], dydx[k])
			weight = pred[k]
			saliency_ave = saliency_ave + sign*dydx[k]*weight
			norm += weight
			counter += 1
		if norm > 0:
			saliency_ave /= norm

		return saliency_ave, counter



#----------------------------------------------------------------------------------------------------
# Train neural networks class
#----------------------------------------------------------------------------------------------------

class NeuralTrainer():
	""" class to train a neural network model """

	def __init__(self, nnmodel, save='best', file_path='.', **kwargs):

		# default optimizer if none given
		self.nnmodel = nnmodel
		self.network = nnmodel.network
		self.placeholders = nnmodel.placeholders
		self.objective = nnmodel.optimization['objective']
		self.save = save
		self.file_path = file_path


		self.train_calc = [nnmodel.train_step, nnmodel.mean_loss, nnmodel.metric]
		self.test_calc = [nnmodel.mean_loss, nnmodel.predictions]

		self.initialize_feed_dict(nnmodel.placeholders, nnmodel.feed_dict)

		# instantiate monitor class to monitor performance
		self.train_monitor = MonitorPerformance(name="train", objective=self.objective, verbose=1)
		self.test_monitor = MonitorPerformance(name="test", objective=self.objective, verbose=1)
		self.valid_monitor = MonitorPerformance(name="cross-validation", objective=self.objective, verbose=1)


	def update_feed_dict(self, key, value):

		self.train_feed[self.placeholders[key]] = value
		if 'keep_prob' in key:
			self.test_feed[self.placeholders[key]] = 1.0
			self.stochastic_feed[self.placeholders[key]] = self.train_feed[placeholders[key]]
		if key == 'is_training':
			self.test_feed[self.placeholders[key]] = False
			self.stochastic_feed[self.placeholders[key]] = False
		else:
			self.test_feed[self.placeholders[key]] = value
			self.stochastic_feed[self.placeholders[key]] = value


	def initialize_feed_dict(self, placeholders, feed_dict):

		self.train_feed = {}
		self.test_feed = {}
		self.stochastic_feed = {}
		for key in feed_dict.keys():
			self.train_feed[placeholders[key]] = feed_dict[key]
			self.test_feed[placeholders[key]] = feed_dict[key]
			self.stochastic_feed[placeholders[key]] = feed_dict[key]
			if key != 'targets':
				self.test_feed[placeholders[key]] = feed_dict[key]
				self.stochastic_feed[placeholders[key]] = feed_dict[key]

			if 'keep_prob' in key:
				self.test_feed[placeholders[key]] = 1.0
				self.stochastic_feed[placeholders[key]] = self.train_feed[placeholders[key]]

			if key == 'is_training':
				self.test_feed[placeholders[key]] = False
				self.stochastic_feed[placeholders[key]] = False

		shape = list(placeholders['inputs'].shape)
		shape[0] = 1
		shape = tuple(shape)
		self.train_feed[placeholders['inputs']] = np.zeros(shape)
		self.test_feed[placeholders['inputs']] = np.zeros(shape)
		self.stochastic_feed[placeholders['inputs']] = np.zeros(shape)

		shape = list(placeholders['targets'].shape)
		shape[0] = 1
		shape = tuple(shape)
		self.train_feed[placeholders['targets']] = np.zeros(shape)
		self.test_feed[placeholders['targets']] = np.zeros(shape)
		self.stochastic_feed[placeholders['targets']] = np.zeros(shape)


	def train_epoch(self, sess, data, batch_size=128, verbose=1, shuffle=True):
		"""Train a mini-batch --> single epoch"""

		# set timer for epoch run
		performance = MonitorPerformance('train', self.objective, verbose)
		performance.set_start_time(start_time = time.time())

		# instantiate batch generator
		num_data = data['targets'].shape[0]
		batch_generator = BatchGenerator(num_data, batch_size, shuffle)
		num_batches = batch_generator.get_num_batches()

		value = 0
		metric = 0
		for i in range(num_batches):
			self.train_feed = batch_generator.next_minibatch(data, self.train_feed, self.placeholders)
			results = sess.run(self.train_calc, feed_dict=self.train_feed)
			metric += results[2]
			performance.add_loss(results[1])
			performance.progress_bar(i+1., num_batches, metric/(i+1))
		if verbose > 1:
			print(" ")

		# calculate mean loss and store
		loss = performance.get_mean_loss()
		self.add_loss(loss, 'train')
		return loss


	def test_model(self, sess, data, batch_size=128, name='test', verbose=1):
		"""perform a complete forward pass, store and print(results)"""

		# instantiate monitor performance and batch generator
		performance = MonitorPerformance('test', self.objective, verbose)

		num_data = data['targets'].shape[0]
		batch_generator = BatchGenerator(num_data, batch_size, shuffle=False)

		label = []
		prediction = []
		for i in range(batch_generator.get_num_batches()):
			self.test_feed = batch_generator.next_minibatch(data, self.test_feed, self.placeholders)
			results = sess.run(self.test_calc, feed_dict=self.test_feed)
			performance.add_loss(results[0])
			prediction.append(results[1])
			label.append(self.test_feed[self.placeholders['targets']])
		prediction = np.vstack(prediction)
		label = np.vstack(label)
		test_loss = performance.get_mean_loss()

		if name == 'eblo':
			return test_loss

		else:
			if name == "train":
				self.train_monitor.update(test_loss, prediction, label)
				mean, std = self.train_monitor.get_metric_values()
				if verbose >= 1:
					self.train_monitor.print_results(name)
			elif name == "valid":
				self.valid_monitor.update(test_loss, prediction, label)
				mean, std = self.valid_monitor.get_metric_values()
				if verbose >= 1:
					self.valid_monitor.print_results(name)
			elif name == "test":
				self.test_monitor.update(test_loss, prediction, label)
				mean, std = self.test_monitor.get_metric_values()
				if verbose >= 1:
					self.test_monitor.print_results(name)
			return test_loss, mean, std


	def get_saliency(self, sess, X, layer, class_index=None, batch_size=500):

		num_data = X.shape[0]
		batch_generator = BatchGenerator(num_data, batch_size, shuffle=False)
		data = {'inputs': X}
		y = layer.get_output()

		saliency = []
		for i in range(batch_generator.get_num_batches()):
			self.test_feed = batch_generator.next_minibatch(data, self.test_feed, self.placeholders)
			val = self.nnmodel.calculate_saliency(sess, y, self.placeholders['inputs'], self.test_feed, class_index=class_index)
			if isinstance(val, (list)):
				saliency.append(val[0])
			else:
				saliency.append(val)
		return np.concatenate(saliency, axis=0)



	def get_stochastic_saliency(self, sess, X, layer, threshold, class_index=None, num_average=200):
		y = layer.get_output()

		saliency = []
		counts = []
		for i in range(X.shape[0]):
			if np.mod(i, 10) == 0:
				print('%d out of %d'%(i, X.shape[0]))

			x = np.expand_dims(X[i], axis=0)
			saliency_ave, counter = self.nnmodel.stochastic_saliency(sess, x, y, self.placeholders['inputs'],
														self.stochastic_feed, num_average, threshold[i], class_index)
			saliency.append(np.expand_dims(saliency_ave, axis=0))
			counts.append(counter)

		#saliency = np.vstack(saliency)
		counts = np.vstack(counts)
		return saliency, counts


	def add_loss(self, loss, name):
		"""add loss score to monitor class"""

		if name == "train":
			self.train_monitor.add_loss(loss)
		elif name == "valid":
			self.valid_monitor.add_loss(loss)
		elif name == "test":
			self.test_monitor.add_loss(loss)


	def save_model(self, sess, addon=None):
		"""save model parameters to file, according to file_path"""


		if addon is not None:
			if self.file_path:
				file_path = self.file_path + '_' + addon + '.ckpt'
				self.nnmodel.save_model_parameters(sess, file_path)
		else:
			if self.file_path:
				if self.save == 'best':
					min_loss, min_epoch, epoch = self.valid_monitor.get_min_loss()
					if self.valid_monitor.loss[-1] <= min_loss:
						print('  lower cross-validation found')
						file_path = self.file_path + '_best.ckpt'
						self.nnmodel.save_model_parameters(sess, file_path)
				elif self.save == 'all':
					epoch = len(self.valid_monitor.loss)
					file_path = self.file_path + '_' + str(epoch) + '.ckpt'
					self.nnmodel.save_model_parameters(sess, file_path)


	def save_all_metrics(self, file_path=None):
		"""save all performance metrics"""

		if not file_path:
			file_path = self.file_path

		if file_path:
			self.train_monitor.save_metrics(file_path)
			self.test_monitor.save_metrics(file_path)
			self.valid_monitor.save_metrics(file_path)
		else:
			print('No file_path provided.')


	def early_stopping(self, current_loss, patience):
		"""check if validation loss is not improving and stop after patience
		runs out"""

		status = True
		if patience:
			min_loss, min_epoch, num_loss = self.valid_monitor.get_min_loss()
			if min_loss < current_loss:
				if patience - (num_loss - min_epoch) < 0:
					status = False
					print("Patience ran out... Early stopping.")
		return status


	def set_best_parameters(self, sess, file_path=[], verbose=1):
		""" set the best parameters from file"""

		if not file_path:
			file_path = self.file_path + '_best.ckpt'

		self.nnmodel.load_model_parameters(sess, file_path, verbose=verbose)


	def get_parameters(self, sess, layer=[]):
		"""return all the parameters of the network"""

		return self.nnmodel.get_parameters(sess, layer)


	def get_activations(self, sess, data, layer='output', batch_size=500):
		"""get the real-valued feature maps of a given convolutional layer"""

		num_data = data['inputs'].shape[0]
		batch_generator = BatchGenerator(num_data, batch_size, shuffle=False)

		activations = []
		for i in range(batch_generator.get_num_batches()):
			self.test_feed = batch_generator.next_minibatch(data, self.test_feed, self.placeholders)
			activations.append(self.nnmodel.get_activations(sess, self.test_feed, layer))
		activations = np.vstack(activations)

		return activations


#----------------------------------------------------------------------------------------------------
# Monitor performance metrics class
#----------------------------------------------------------------------------------------------------

class MonitorPerformance():
	"""helper class to monitor and store performance metrics during
	   training. This class uses the metrics for early stopping. """

	def __init__(self, name='', objective='binary', verbose=1):
		self.name = name
		self.objective = objective
		self.verbose = verbose
		self.loss = []
		self.metric = []
		self.metric_std = []


	def set_verbose(self, verbose):
		self.verbose = verbose


	def add_loss(self, loss):
		if np.isnan(loss):
			loss = 1e10
		self.loss = np.append(self.loss, loss)


	def add_metrics(self, scores):
		self.metric.append(scores[0])
		self.metric_std.append(scores[1])


	def update(self, loss, prediction, label):
		scores = metrics.calculate_metrics(label, prediction, self.objective)
		self.add_loss(loss)
		self.add_metrics(scores)


	def get_mean_loss(self):
		return np.mean(self.loss)


	def get_metric_values(self):
		return self.metric[-1], self.metric_std[-1]


	def get_min_loss(self):
		min_loss = min(self.loss)
		min_index = np.argmin(self.loss)
		num_loss = len(self.loss)
		return min_loss, min_index, num_loss


	def set_start_time(self, start_time):
		self.start_time = start_time


	def print_results(self, name):
		if self.verbose >= 1:
			if name == 'test':
				name += ' '

			print("  " + name + " loss:\t\t{:.5f}".format(self.loss[-1]))
			mean_vals, error_vals = self.get_metric_values()

			if (self.objective == "binary") | (self.objective == "categorical"):
				print("  " + name + " accuracy:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
				print("  " + name + " auc-roc:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
				print("  " + name + " auc-pr:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))
			elif (self.objective == 'squared_error') | (self.objective == 'kl_divergence'):
				print("  " + name + " Pearson's R:\t{:.5f}+/-{:.5f}".format(mean_vals[0], error_vals[0]))
				print("  " + name + " rsquare:\t{:.5f}+/-{:.5f}".format(mean_vals[1], error_vals[1]))
				print("  " + name + " slope:\t\t{:.5f}+/-{:.5f}".format(mean_vals[2], error_vals[2]))


	def progress_bar(self, epoch, num_batches, value, bar_length=30):
		if self.verbose > 1:
			remaining_time = (time.time()-self.start_time)*(num_batches-(epoch+1))/(epoch+1)
			percent = epoch/num_batches
			progress = '='*int(round(percent*bar_length))
			spaces = ' '*int(bar_length-round(percent*bar_length))
			if (self.objective == "binary") | (self.objective == "categorical"):
				sys.stdout.write("\r[%s] %.1f%% -- remaining time=%ds -- loss=%.5f -- accuracy=%.2f%%  " \
				%(progress+spaces, percent*100, remaining_time, self.get_mean_loss(), value*100))
			else:# (self.objective == 'squared_error') | (self.objective == 'elbo_gaussian')| (self.objective == 'elbo_binary'):
				sys.stdout.write("\r[%s] %.1f%% -- remaining time=%ds -- loss=%.5f" \
				%(progress+spaces, percent*100, remaining_time, self.get_mean_loss()))

			if epoch == num_batches:
				if (self.objective == "binary") | (self.objective == "categorical"):
					sys.stdout.write("\r[%s] %.1f%% -- elapsed time=%.2fs -- loss=%.5f -- acc=%.5f\n" \
					    %(progress+spaces, percent*100, time.time()-self.start_time, self.get_mean_loss(), value*100))
				else:# (self.objective == 'squared_error') | (self.objective == 'elbo_gaussian')| (self.objective == 'elbo_binary'):
					sys.stdout.write("\r[%s] %.1f%% -- elapsed time=%ds -- loss=%.5f" \
					%(progress+spaces, percent*100, time.time()-self.start_time, self.get_mean_loss()))
			sys.stdout.flush()

	def save_metrics(self, file_path):
		savepath = file_path + "_" + self.name +"_performance.pickle"
		print("  saving metrics to " + savepath)

		with open(savepath, 'wb') as f:
			cPickle.dump(self.name, f, protocol=cPickle.HIGHEST_PROTOCOL)
			cPickle.dump(self.loss, f, protocol=cPickle.HIGHEST_PROTOCOL)
			cPickle.dump(self.metric, f, protocol=cPickle.HIGHEST_PROTOCOL)
			cPickle.dump(self.metric_std, f, protocol=cPickle.HIGHEST_PROTOCOL)



#--------------------------------------------------------------------------------------------------
# Batch Generator Class
#--------------------------------------------------------------------------------------------------

class BatchGenerator():
	""" helper class to generate mini-batches """

	def __init__(self, num_data, batch_size=128, shuffle=False):

		self.num_data = num_data
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.current_batch = 0
		self.generate_minibatches(batch_size, shuffle)

	def generate_minibatches(self, batch_size=None, shuffle=None):

		if shuffle is None:
			shuffle = self.shuffle
		if batch_size is None:
			batch_size = self.batch_size

		if shuffle == True:
			index = np.random.permutation(self.num_data)
		else:
			index = range(self.num_data)

		self.batch_size = batch_size
		self.num_batches = self.num_data // self.batch_size

		self.indices = []
		for i in range(self.num_batches):
			self.indices.append(index[i*self.batch_size:i*self.batch_size+self.batch_size])

		# get remainder
		index = range(self.num_batches*self.batch_size, self.num_data)
		if index:
			self.indices.append(index)
			self.num_batches += 1

		self.current_batch = 0

	def next_minibatch(self, data, feed_dict, placeholders):
		indices = np.sort(self.indices[self.current_batch])

		for key in data.keys():
			feed_dict.update({placeholders[key]: data[key][indices]})

		self.current_batch += 1
		if self.current_batch == self.num_batches:
			self.current_batch = 0

		return feed_dict

	def get_batch_index(self):
		return self.current_batch

	def get_num_batches(self):
		return self.num_batches
