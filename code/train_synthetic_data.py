from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import helper
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit

#------------------------------------------------------------------------------------------------

all_models = ['cnn_2_50', 'cnn_4_25', 'cnn_10_10', 'cnn_25_4', 'cnn_50_2', 'cnn_100_1',
			  'cnn_2_50_invariant', 'scnn_4_25', 'scnn_25_4']

# save path
results_path = utils.make_directory('../results', 'synthetic')
params_path = utils.make_directory(results_path, 'model_params')

# load dataset
data_path = '../data/synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

# loop through models
for model_name in all_models:
	print('model: ' + model_name)
	tf.reset_default_graph()
	tf.set_random_seed(247)
	np.random.seed(247) # for reproducibility

	# load model parameters
	genome_model = helper.import_model(model_name)
	model_layers, optimization = genome_model.model(input_shape, output_shape)

	# build neural network class
	nnmodel = nn.NeuralNet(seed=247)
	nnmodel.build_layers(model_layers, optimization, supervised=True)
	nnmodel.inspect_layers()

	# create neural trainer
	file_path = os.path.join(params_path, model_name)
	nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

	# initialize session
	sess = utils.initialize_session()

	# set data in dictionary
	data = {'train': train, 'valid': valid, 'test': test}

	# fit model
	fit.train_minibatch(sess, nntrainer, data, batch_size=100, num_epochs=100,
		  patience=20, verbose=2, shuffle=True, save_all=False)
