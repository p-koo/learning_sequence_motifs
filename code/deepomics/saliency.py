from __future__ import print_function
import os, sys, time
import numpy as np
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops


# --------------------------------------------------------------------------------------------------------------

def backprop(X, layer='output', class_index=None, params=None):
	tf.reset_default_graph()
	if 'use_scope' not in params:
		params['use_scope'] = True

	# build new graph
	model_layers, optimization = params['genome_model'](params['input_shape'], params['output_shape'])
	nnmodel = nn.NeuralNet()
	nnmodel.build_layers(model_layers, optimization, use_scope=params['use_scope'])
	nntrainer = nn.NeuralTrainer(nnmodel, save='best', filepath=params['model_path'])

	# setup session and restore optimal parameters
	sess = utils.initialize_session(nnmodel.placeholders)
	nntrainer.set_best_parameters(sess, params['model_path'], verbose=0)

	# backprop saliency
	if layer == 'output':
		layer = list(nnmodel.network.keys())[-2]
		saliency = nntrainer.get_saliency(sess, X, nnmodel.network[layer], class_index=class_index, batch_size=128)
		saliency = saliency[0]
	else:
		data = {'inputs': X}
		layer_activations = nntrainer.get_activations(sess, data, layer)
		max_activations = np.squeeze(np.max(layer_activations, axis=1))
		active_indices = np.where(max_activations > 0)[0]
		active_indices = active_indices[np.argsort(max_activations[active_indices])[::-1]]

		saliency = []
		for neuron_index in active_indices:
			val = get_saliency(sess, X, nnmodel.network[layer], class_index=neuron_index, batch_size=128)
			saliency.append(val)

	sess.close()
	tf.reset_default_graph()
	return saliency


def guided_backprop(X, layer='output', class_index=None, params=None, batch_size=128):
	tf.reset_default_graph()
	if 'use_scope' not in params:
		params['use_scope'] = True

	# build new graph
	#g = tf.get_default_graph()
	#with g.gradient_override_map({'Relu': 'GuidedRelu'}):
	model_layers, optimization = params['genome_model'](params['input_shape'], params['output_shape'])
	nnmodel = nn.NeuralNet()
	nnmodel.build_layers(model_layers, optimization, method='guided', use_scope=params['use_scope'])
	nntrainer = nn.NeuralTrainer(nnmodel, save='best', filepath=params['model_path'])

	# setup session and restore optimal parameters
	sess = utils.initialize_session(nnmodel.placeholders)
	nntrainer.set_best_parameters(sess, params['model_path'], verbose=0)

	# backprop saliency
	if layer == 'output':
		layer = list(nnmodel.network.keys())[-2]
		saliency = nntrainer.get_saliency(sess, X, nnmodel.network[layer], class_index=class_index, batch_size=batch_size)
	else:
		data = {'inputs': X}
		layer_activations = nntrainer.get_activations(sess, data, layer)
		max_activations = np.squeeze(np.max(layer_activations, axis=1))
		active_indices = np.where(max_activations > 0)[0]
		active_indices = active_indices[np.argsort(max_activations[active_indices])[::-1]]

		saliency = []
		for neuron_index in active_indices:
			val = nntrainer.get_saliency(sess, X, nnmodel.network[layer][:,:,:,neuron_index], class_index=None, batch_size=batch_size)
			saliency.append(val)

	sess.close()
	tf.reset_default_graph()
	return saliency


def stochastic_backprop(X, layer='output', class_index=None, params=None,
						num_average=400, threshold=12.0):
	tf.reset_default_graph()
	if 'use_scope' not in params:
		params['use_scope'] = True

	# build new graph
	model_layers, optimization = params['genome_model'](params['input_shape'], params['output_shape'])
	nnmodel = nn.NeuralNet()
	nnmodel.build_layers(model_layers, optimization, use_scope=params['use_scope'])
	nntrainer = nn.NeuralTrainer(nnmodel, save='best', filepath=params['model_path'])

	# setup session and restore optimal parameters
	sess = utils.initialize_session(nnmodel.placeholders)
	nntrainer.set_best_parameters(sess, params['model_path'], verbose=0)

	# stochastic backprop saliency
	if layer == 'output':
		layer = list(nnmodel.network.keys())[-2]
		saliency, counts = nntrainer.get_stochastic_saliency(sess, X, nnmodel.network[layer], class_index=class_index,
													num_average=num_average, threshold=threshold)
	else:
		data = {'inputs': X}
		layer_activations = nntrainer.get_activations(sess, data, layer)
		max_activations = np.squeeze(np.max(layer_activations, axis=1))
		active_indices = np.where(max_activations > 0)[0]
		active_indices = active_indices[np.argsort(max_activations[active_indices])[::-1]]
		saliency = []
		count = []
		for neuron_index in active_indices:
			val, counts = nntrainer.get_stochastic_saliency(sess, X, nnmodel.network[layer], class_index=neuron_index,
													num_average=num_average, threshold=threshold)
			saliency.append(val)
			counts.append(count)

	sess.close()
	tf.reset_default_graph()

	return np.vstack(saliency), np.array(counts)


def stochastic_guided_backprop(X, layer='output', class_index=None, params=None,
								num_average=400, threshold=12.0):
	tf.reset_default_graph()
	if 'use_scope' not in params:
		params['use_scope'] = True

	# build new graph
	#g = tf.get_default_graph()
	#with g.gradient_override_map({'Relu': 'GuidedRelu'}):
	model_layers, optimization = params['genome_model'](params['input_shape'], params['output_shape'])
	nnmodel = nn.NeuralNet()
	nnmodel.build_layers(model_layers, optimization, method='guided', use_scope=params['use_scope'])
	nntrainer = nn.NeuralTrainer(nnmodel, save='best', filepath=params['model_path'])

	# setup session and restore optimal parameters
	sess = utils.initialize_session(nnmodel.placeholders)
	nntrainer.set_best_parameters(sess, params['model_path'], verbose=0)

	# stochastic guided saliency
	if layer == 'output':
		layer = list(nnmodel.network.keys())[-2]
		saliency, counts = nntrainer.get_stochastic_saliency(sess, X,nnmodel. network[layer], class_index=class_index,
													num_average=num_average, threshold=threshold)
	else:
		data = {'inputs': X}
		layer_activations = nntrainer.get_activations(sess, data, layer)
		max_activations = np.squeeze(np.max(layer_activations, axis=1))
		active_indices = np.where(max_activations > 0)[0]
		active_indices = active_indices[np.argsort(max_activations[active_indices])[::-1]]
		saliency = []
		counts = []
		for neuron_index in active_indices:
			val, count = nntrainer.get_stochastic_saliency(sess, X, nnmodel.network[layer], class_index=neuron_index,
													num_average=num_average, threshold=threshold)
			saliency.append(val)
			counts.append(count)

	sess.close()
	tf.reset_default_graph()


	return np.vstack(saliency), np.array(counts)
