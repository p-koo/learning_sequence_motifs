from __future__ import print_function
import tensorflow as tf
from deepomics import layers
from deepomics import init, utils

from collections import OrderedDict

__all__ = [
	"NeuralBuild"
]

class NeuralBuild():
	def __init__(self, seed=None):

		self.feed_dict = OrderedDict()
		self.feed_dict['is_training'] = True
		self.feed_dict['learning_rate'] = 0.001

		self.placeholders = OrderedDict()
		self.placeholders['is_training'] = tf.placeholder(tf.bool, name='is_training')
		self.placeholders['learning_rate'] = tf.placeholder(tf.float32)

		self.seed = {'seed': tf.set_random_seed(seed)}

	def build_layers(self, model_layers, supervised=True):

		self.network = OrderedDict()
		name_gen = NameGenerator()
		self.num_dropout = 0
		self.num_inputs = 0
		self.last_layer = ''

		# loop to build each layer of network
		for model_layer in model_layers:
			layer = model_layer['layer']

			# name of layer
			if 'name' in model_layer:
				name = model_layer['name']
			else:
				name = name_gen.generate_name(layer)

			# set scope for each layer
			with tf.name_scope(name) as scope:
				if layer == "input":

					# add input layer
					self.single_layer(model_layer, name)

				elif layer == 'embedding':
					vocab_size = model_layer['vocab_size']
					embedding_size = model_layer['embedding_size']
					if 'max_norm' in model_layer:
						max_norm = model_layer['max_norm']
					else:
						max_norm = None
					self.network[name] = layers.EmbeddingLayer(self.network[self.last_layer], vocab_size, embedding_size, max_norm)
					self.last_layer = name

				elif (layer == 'variational') | (layer == 'variational_normal'):
					if 'name' in model_layer:
						name = model_layer['name']
					else:
						name = 'Z'

					self.network[name+'_mu'] = layers.DenseLayer(self.network[self.last_layer], num_units=model_layer['num_units'], b=init.GlorotUniform(), **self.seed)
					self.network[name+'_logvar'] = layers.DenseLayer(self.network[self.last_layer], num_units=model_layer['num_units'], b=init.GlorotUniform(), **self.seed)
					self.network[name+'_sample'] = layers.VariationalSampleLayer(self.network[name+'_mu'], self.network[name+'_logvar'])
					self.last_layer = name+'_sample'

				elif layer == 'variational_softmax':
					if 'hard' in model_layer:
						hard = model_layer['hard']
					else:
						hard = False
					num_categories, num_classes = model_layer['shape']

					if 'temperature' in model_layer:
						temperature = model_layer['temperature']
					else:
						temperature = 5.0
					self.feed_dict['temperature'] = temperature
					self.placeholders['temperature'] = tf.placeholder(dtype=tf.float32, name="temperature")
					if 'name' in model_layer:
						name = model_layer['name']
					else:
						name = 'Z'

					if 'output' in model_layer:
						output = model_layer['output']
					else:
						output = 'softmax'

					self.network[name+'_logits'] = layers.DenseLayer(self.network[self.last_layer], num_units=num_categories*num_classes, b=init.GlorotUniform())

					self.network[name+'_logits_reshape'] = layers.ReshapeLayer(self.network[name+'_logits'], shape=[-1, num_categories, num_classes])

					self.network[name+'_sample'] = layers.CategoricalSampleLayer(self.network[name+'_logits_reshape'], temperature=temperature, hard=hard)

					self.network[name+'_softmax'] = layers.Softmax2DLayer(self.network[name+'_logits_reshape'])

					if output == 'hard':
						self.network[name] = layers.ReshapeLayer(self.network[name+'_sample'], shape=[-1, num_categories*num_classes])
					else:
						self.network[name] = layers.ReshapeLayer(self.network[name+'_softmax'], shape=[-1, num_categories*num_classes])

					self.last_layer = name


				else:
					if layer == 'conv1d_residual':
						self.conv1d_residual_block(model_layer, name)

					elif layer == 'conv2d_residual':
						self.conv2d_residual_block(model_layer, name)

					elif layer == 'dense_residual':
						self.dense_residual_block(model_layer, name)

					else:
						# add core layer
						self.single_layer(model_layer, name)

					# add Batch normalization layer
					if 'norm' in model_layer:
						if 'batch' in model_layer['norm']:
							with tf.name_scope("norm") as scope:
								new_layer = name + '_batch' #str(counter) + '_' + name + '_batch'
								self.network[new_layer] = layers.BatchNormLayer(self.network[self.last_layer], self.placeholders['is_training'])
								self.last_layer = new_layer

					else:
						if (model_layer['layer'] == 'dense') | (model_layer['layer'] == 'conv1d') | (model_layer['layer'] == 'conv2d'):
							if 'b' in model_layer:
								if model_layer['b'] != None:
									with tf.name_scope("bias") as scope:
										b = init.Constant(model_layer['b'])
										new_layer = name+'_bias'
										self.network[new_layer] = layers.BiasLayer(self.network[self.last_layer], b=b)
										self.last_layer = new_layer

							elif 'norm' not in model_layer:
								with tf.name_scope("bias") as scope:
									b = init.GlorotUniform()
									new_layer = name+'_bias'
									self.network[new_layer] = layers.BiasLayer(self.network[self.last_layer], b=b)
									self.last_layer = new_layer

				# add activation layer
				if 'activation' in model_layer:
					new_layer = name+'_active'
					self.network[new_layer] = layers.ActivationLayer(self.network[self.last_layer], function=model_layer['activation'], name=scope)
					self.last_layer = new_layer

				# add max-pooling layer
				if 'max_pool' in model_layer:
					new_layer = name+'_maxpool'  # str(counter) + '_' + name+'_pool'
					if 'max_pool_strides' in model_layer:
						strides = model_layer['max_pool_strides']
					else:
						strides = model_layer['max_pool']

					if isinstance(model_layer['max_pool'], (tuple, list)):
						self.network[new_layer] = layers.MaxPool2DLayer(self.network[self.last_layer], pool_size=model_layer['max_pool'], strides=strides, name=name+'_maxpool')
					else:
						self.network[new_layer] = layers.MaxPool1DLayer(self.network[self.last_layer], pool_size=model_layer['max_pool'], strides=strides, name=name+'_maxpool')
					self.last_layer = new_layer

				# add mean-pooling layer
				elif 'mean_pool' in model_layer:
					new_layer = name+'_meanpool'  # str(counter) + '_' + name+'_pool'
					if 'mean_pool_strides' in model_layer:
						strides = model_layer['mean_pool_strides']
					else:
						strides = model_layer['mean_pool']

					if isinstance(model_layer['mean_pool'], (tuple, list)):
						self.network[new_layer] = layers.MeanPool2DLayer(self.network[self.last_layer], pool_size=model_layer['mean_pool'], strides=strides, name=name+'_meanpool')
					else:
						self.network[new_layer] = layers.MeanPool1DLayer(self.network[self.last_layer], pool_size=model_layer['mean_pool'], strides=strides, name=name+'_meanpool')
					self.last_layer = new_layer

				# add global-pooling layer
				elif 'global_pool' in model_layer:
					new_layer = name+'_globalpool'
					self.network[new_layer] = layers.GlobalPoolLayer(self.network[self.last_layer], func=model_layer['global_pool'], name=name+'_globalpool')
					self.last_layer = new_layer

				# add dropout layer
				if 'dropout' in model_layer:
					new_layer = name+'_dropout' # str(counter) + '_' + name+'_dropout'
					placeholder_name = 'keep_prob_'+str(self.num_dropout)
					self.placeholders[placeholder_name] = tf.placeholder(tf.float32, name=placeholder_name)
					self.feed_dict[placeholder_name] = 1-model_layer['dropout']
					self.num_dropout += 1
					self.network[new_layer] = layers.DropoutLayer(self.network[self.last_layer], keep_prob=self.placeholders[placeholder_name], name=name+'_dropout')
					self.last_layer = new_layer

				if ('reshape' in model_layer) & (layer != 'reshape'):
					new_layer = name+'_reshape'
					self.network[new_layer] = layers.ReshapeLayer(self.network[self.last_layer], model_layer['reshape'])
					self.last_layer = new_layer

		if supervised:
			self.network['output'] = self.network.pop(self.last_layer)
			shape = self.network['output'].get_output_shape()
			targets = utils.placeholder(shape=shape, name='output')
			self.placeholders['targets'] = targets
			self.feed_dict['targets'] = []
		else:
			self.network['X'] = self.network.pop(self.last_layer)
			self.placeholders['targets'] = self.placeholders['inputs']
			self.feed_dict['targets'] = []

		return self.network, self.placeholders, self.feed_dict


	def single_layer(self, model_layer, name):
		""" build a single layer"""

		# input layer
		if model_layer['layer'] == 'input':

			with tf.name_scope('input') as scope:
				input_shape = str(model_layer['input_shape'])
				inputs = utils.placeholder(shape=model_layer['input_shape'], name=name)
				self.network[name] = layers.InputLayer(inputs)
				self.placeholders[name] = inputs
				self.feed_dict[name] = []

		# dense layer
		elif model_layer['layer'] == 'dense':

			with tf.name_scope('dense') as scope:
				if 'W' not in model_layer.keys():
					model_layer['W'] = init.GlorotUniform(**self.seed)
				self.network[name] = layers.DenseLayer(self.network[self.last_layer], num_units=model_layer['num_units'],
													 W=model_layer['W'],
													 b=None)

		# convolution layer
		elif (model_layer['layer'] == 'conv2d'):

			with tf.name_scope('conv2d') as scope:
				if 'W' not in model_layer.keys():
					W = init.GlorotUniform(**self.seed)
				else:
					W = model_layer['W']
				if 'padding' not in model_layer.keys():
					padding = 'VALID'
				else:
					padding = model_layer['padding']
				if 'strides' not in model_layer.keys():
					strides = (1, 1)
				else:
					strides = model_layer['strides']

				self.network[name] = layers.Conv2DLayer(self.network[self.last_layer], num_filters=model_layer['num_filters'],
													  filter_size=model_layer['filter_size'],
													  W=W,
													  padding=padding,
													  strides=strides)

		elif model_layer['layer'] == 'conv1d':
			with tf.name_scope('conv1d') as scope:
				if 'W' not in model_layer.keys():
					W = init.GlorotUniform(**self.seed)
				else:
					W = model_layer['W']
				if 'padding' not in model_layer.keys():
					padding = 'VALID'
				else:
					padding = model_layer['padding']
				if 'strides' not in model_layer.keys():
					strides = 1
				else:
					strides = model_layer['strides']
				reverse=False
				if 'reverse' in model_layer:
					reverse = model_layer['reverse']


				self.network[name] = layers.Conv1DLayer(self.network[self.last_layer], num_filters=model_layer['num_filters'],
													  filter_size=model_layer['filter_size'],
													  W=W,
													  padding=padding,
													  strides=strides,
													  reverse=reverse)

		# convolution layer
		elif (model_layer['layer'] == 'conv2d_transpose'):

			if 'W' not in model_layer.keys():
				W = init.GlorotUniform(**self.seed)
			else:
				W = model_layer['W']
			if 'padding' not in model_layer.keys():
				padding = 'SAME'
			else:
				padding = model_layer['padding']
			if 'strides' not in model_layer.keys():
				strides = (1, 1)
			else:
				strides = model_layer['strides']

			self.network[name] = layers.TransposeConv2DLayer(self.network[self.last_layer], num_filters=model_layer['num_filters'],
												  filter_size=model_layer['filter_size'],
												  W=W,
												  padding=padding,
												  strides=strides)

		elif model_layer['layer'] == 'conv1d_transpose':
			if 'W' not in model_layer.keys():
				W = init.GlorotUniform(**self.seed)
			else:
				W = model_layer['W']
			if 'padding' not in model_layer.keys():
				padding = 'SAME'
			else:
				padding = model_layer['padding']
			if 'strides' not in model_layer.keys():
				strides = 1
			else:
				strides = model_layer['strides']


			self.network[name] = layers.TransposeConv1DLayer(self.network[self.last_layer], num_filters=model_layer['num_filters'],
												  filter_size=model_layer['filter_size'],
												  W=W,
												  padding=padding,
												  strides=strides)

		# concat layer
		elif model_layer['layer'] == 'concat':
			self.network[name] = layers.ConcatLayer([self.network[self.last_layer], self.network[model_layer['concat']]])

		# element-size sum layer
		elif model_layer['layer'] == 'sum':
			self.network[name] = layers.ElemwiseSumLayer([self.network[self.last_layer], model_layer['sum']])

		# reshape layer
		elif model_layer['layer'] == 'reshape':
			self.network[name] = layers.ReshapeLayer(self.network[self.last_layer], model_layer['reshape'])

		elif model_layer['layer'] == 'reduce_max':
			self.network[name] = layers.MaxLayer(self.network[self.last_layer], axis=1)

		elif model_layer['layer'] == 'reduce_mean':
			self.network[name] = layers.MeanLayer(self.network[self.last_layer], axis=1)

		elif model_layer['layer'] == 'softmax2D':
			self.network[name] = layers.Softmax2DLayer(self.network[self.last_layer])

		self.last_layer = name



	def conv1d_residual_block(self, model_layer, name):
		with tf.name_scope('conv1d_residual_block') as scope:
			last_layer = self.last_layer

			filter_size = model_layer['filter_size']
			if 'function' in model_layer:
				activation = model_layer['function']
			else:
				activation = 'relu'

			# original residual unit
			shape = self.network[last_layer].get_output_shape()
			num_filters = shape[-1].value

			if 'W' not in model_layer.keys():
				W = init.GlorotUniform(**self.seed)
			else:
				W = model_layer['W']
			self.network[name+'_1resid'] = layers.Conv1DLayer(self.network[last_layer], num_filters=num_filters,
												  filter_size=filter_size,
												  W=W,
												  padding='SAME')
			#self.network[name+'_1resid'] = layers.Conv2DLayer(self.network[last_layer], num_filters=num_filters, filter_size=filter_size, padding='SAME', **self.seed)
			self.network[name+'_1resid_norm'] = layers.BatchNormLayer(self.network[name+'_1resid'], self.placeholders['is_training'])
			self.network[name+'_1resid_active'] = layers.ActivationLayer(self.network[name+'_1resid_norm'], function=activation)

			if 'dropout_block' in model_layer:
				placeholder_name = 'keep_prob_'+str(self.num_dropout)
				self.placeholders[placeholder_name] = tf.placeholder(tf.float32, name=placeholder_name)
				self.feed_dict[placeholder_name] = 1-model_layer['dropout_block']
				self.num_dropout += 1
				self.network[name+'_dropout1'] = layers.DropoutLayer(self.network[name+'_1resid_active'], keep_prob=self.placeholders[placeholder_name])
				lastname = name+'_dropout1'
			else:
				lastname = name+'_1resid_active'

			if 'W' not in model_layer.keys():
				W = init.GlorotUniform(**self.seed)
			else:
				W = model_layer['W']
			self.network[name+'_2resid'] = layers.Conv1DLayer(self.network[lastname], num_filters=num_filters,
												  filter_size=filter_size,
												  W=W,
												  padding='SAME')
			#self.network[name+'_2resid'] = layers.Conv2DLayer(self.network[lastname], num_filters=num_filters, filter_size=filter_size, padding='SAME', **self.seed)
			self.network[name+'_2resid_norm'] = layers.BatchNormLayer(self.network[name+'_2resid'], self.placeholders['is_training'])
			self.network[name+'_resid_sum'] = layers.ElementwiseSumLayer([self.network[last_layer], self.network[name+'_2resid_norm']])
			self.network[name+'_resid'] = layers.ActivationLayer(self.network[name+'_resid_sum'], function=activation)

			self.last_layer = name+'_resid'



	def conv2d_residual_block(self, model_layer, name):
		with tf.name_scope('conv2d_residual_block') as scope:
			last_layer = self.last_layer
			filter_size = model_layer['filter_size']
			if 'function' in model_layer:
				activation = model_layer['function']
			else:
				activation = 'relu'

			# original residual unit
			shape = self.network[last_layer].get_output_shape()
			num_filters = shape[-1].value

			if not isinstance(filter_size, (list, tuple)):
				filter_size = (filter_size, 1)

			if 'W' not in model_layer.keys():
				W = init.GlorotUniform(**self.seed)
			else:
				W = model_layer['W']
			self.network[name+'_1resid'] = layers.Conv2DLayer(self.network[last_layer], num_filters=num_filters,
												  filter_size=filter_size,
												  W=W,
												  padding='SAME')
			#self.network[name+'_1resid'] = layers.Conv2DLayer(self.network[last_layer], num_filters=num_filters, filter_size=filter_size, padding='SAME', **self.seed)
			self.network[name+'_1resid_norm'] = layers.BatchNormLayer(self.network[name+'_1resid'], self.placeholders['is_training'])
			self.network[name+'_1resid_active'] = layers.ActivationLayer(self.network[name+'_1resid_norm'], function=activation)


			if 'dropout_block' in model_layer:
				placeholder_name = 'keep_prob_'+str(self.num_dropout)
				self.placeholders[placeholder_name] = tf.placeholder(tf.float32, name=placeholder_name)
				self.feed_dict[placeholder_name] = 1-model_layer['dropout_block']
				self.num_dropout += 1
				self.network[name+'_dropout1'] = layers.DropoutLayer(self.network[name+'_1resid_active'], keep_prob=self.placeholders[placeholder_name])
				lastname = name+'_dropout1'
			else:
				lastname = name+'_1resid_active'

			if 'W' not in model_layer.keys():
				W = init.GlorotUniform(**self.seed)
			else:
				W = model_layer['W']
			self.network[name+'_2resid'] = layers.Conv2DLayer(self.network[lastname], num_filters=num_filters,
												  filter_size=filter_size,
												  W=W,
												  padding='SAME')
			#self.network[name+'_2resid'] = layers.Conv2DLayer(self.network[lastname], num_filters=num_filters, filter_size=filter_size, padding='SAME', **self.seed)
			self.network[name+'_2resid_norm'] = layers.BatchNormLayer(self.network[name+'_2resid'], self.placeholders['is_training'])
			self.network[name+'_resid_sum'] = layers.ElementwiseSumLayer([self.network[last_layer], self.network[name+'_2resid_norm']])
			self.network[name+'_resid'] = layers.ActivationLayer(self.network[name+'_resid_sum'], function=activation)
			self.last_layer = name+'_resid'




	def dense_residual_block(self, model_layer, name):
		with tf.name_scope('dense_residual_block') as scope:
			last_layer = self.last_layer

			if 'function' in model_layer:
				activation = model_layer['function']
			else:
				activation = 'relu'

			# original residual unit
			shape = self.network[last_layer].get_output_shape()
			num_units = shape[-1].value

			self.network[name+'_1resid'] = layers.DenseLayer(self.network[last_layer], num_units=num_units, b=None, **self.seed)
			self.network[name+'_1resid_norm'] = layers.BatchNormLayer(self.network[name+'_1resid'], self.placeholders['is_training'])
			self.network[name+'_1resid_active'] = layers.ActivationLayer(self.network[name+'_1resid_norm'], function=activation)

			if 'dropout_block' in model_layer:
				placeholder_name = 'keep_prob_'+str(self.num_dropout)
				self.placeholders[placeholder_name] = tf.placeholder(tf.float32, name=placeholder_name)
				self.feed_dict[placeholder_name] = 1-model_layer['dropout_block']
				self.num_dropout += 1
				self.network[name+'_dropout1'] = layers.DropoutLayer(self.network[name+'_1resid_active'], keep_prob=self.placeholders[placeholder_name])
				lastname = name+'_dropout1'
			else:
				lastname = name+'_1resid_active'

			self.network[name+'_2resid'] = layers.DenseLayer(self.network[lastname], num_units=num_units, b=None, **self.seed)
			self.network[name+'_2resid_norm'] = layers.BatchNormLayer(self.network[name+'_2resid'], self.placeholders['is_training'])
			self.network[name+'_resid_sum'] = layers.ElementwiseSumLayer([self.network[last_layer], self.network[name+'_2resid_norm']])
			self.network[name+'_resid'] = layers.ActivationLayer(self.network[name+'_resid_sum'], function=activation)
			self.last_layer = name+'_resid'


#--------------------------------------------------------------------------------------------------------------------
# help keep track of names for main layers

class NameGenerator():
	def __init__(self):
		self.num_input = 0
		self.num_conv1d = 0
		self.num_conv2d = 0
		self.num_dense = 0
		self.num_conv1d_residual = 0
		self.num_conv2d_residual = 0
		self.num_dense_residual = 0
		self.num_transpose_conv1d = 0
		self.num_transpose_conv2d = 0
		self.num_concat = 0
		self.num_sum = 0
		self.num_reshape = 0
		self.num_noise = 0
		self.num_lstm = 0
		self.num_bilstm = 0
		self.num_highway = 0
		self.num_variational = 0
		self.num_reduce_max = 0
		self.num_reduce_mean = 0
		self.num_variational_softmax = 0
		self.num_softmax2D = 0
		self.num_embedding = 0

	def generate_name(self, layer):
		if layer == 'input':
			self.num_input += 1
			if self.num_input == 1:
				name = 'inputs'
			else:
				name = 'inputs_' + str(self.num_input)

		elif layer == 'conv1d':
			name = 'conv1d_' + str(self.num_conv1d)
			self.num_conv1d += 1

		elif (layer == 'conv2d') | (layer == 'convolution'):
			name = 'conv2d_' + str(self.num_conv2d)
			self.num_conv2d += 1

		elif layer == 'dense':
			name = 'dense_' + str(self.num_dense)
			self.num_dense += 1

		elif layer == 'conv1d_residual':
			name = 'conv1d_residual_' + str(self.num_conv1d_residual)
			self.num_conv1d_residual += 1

		elif layer == 'conv2d_residual':
			name = 'conv2d_residual_' + str(self.num_conv2d_residual)
			self.num_conv1d_residual += 1

		elif layer == 'dense_residual':
			name = 'dense_residual_' + str(self.num_dense_residual)
			self.num_dense_residual += 1

		elif layer == 'conv1d_transpose':
			name = 'transpose_conv1d_' + str(self.num_transpose_conv1d)
			self.num_transpose_conv1d += 1

		elif (layer == 'conv2d_transpose') | (layer == 'transpose_convolution'):
			name = 'transpose_conv2d_' + str(self.num_transpose_conv2d)
			self.num_transpose_conv2d += 1

		elif layer == 'concat':
			name = 'concat_' + str(self.num_concat)
			self.num_concat += 1

		elif layer == 'sum':
			name = 'sum_' + str(self.num_sum)
			self.num_sum += 1

		elif layer == 'reshape':
			name = 'reshape_' + str(self.num_reshape)
			self.num_reshape += 1

		elif layer == 'noise':
			name = 'noise_' + str(self.num_noise)
			self.num_noise += 1

		elif layer == 'lstm':
			name = 'lstm_' + str(self.num_lstm)
			self.num_lstm += 1

		elif layer == 'bilstm':
			name = 'bilstm_' + str(self.num_bilstm)
			self.num_bilstm += 1

		elif layer == 'highway':
			name = 'highway_' + str(self.num_highway)
			self.num_highway += 1

		elif (layer == 'variational') | (layer == 'variational_normal'):
			name = 'variational_' + str(self.num_variational)
			self.num_variational += 1

		elif layer == 'reduce_max':
			name = 'reduce_max' + str(self.num_reduce_max)
			self.num_reduce_max += 1

		elif layer == 'reduce_mean':
			name = 'reduce_mean' + str(self.num_reduce_mean)
			self.num_reduce_mean += 1

		elif layer == 'variational_softmax':
			name = 'variational_softmax' + str(self.num_variational_softmax)

		elif layer == 'softmax2D':
			name = 'softmax2D' + str(self.num_softmax2D)

		elif layer == 'embedding':
			name = 'embedding' + str(self.num_embedding)
		return name
