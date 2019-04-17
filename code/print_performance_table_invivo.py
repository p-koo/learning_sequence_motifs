from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import helper
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, metrics

#------------------------------------------------------------------------------------------------


all_models = ['cnn_2', 'cnn_4', 'cnn_10', 'cnn_25', 'cnn_50', 'cnn_100',
			  'cnn_50_2', 'cnn9_4', 'cnn9_25']

# save path
results_path = utils.make_directory('../results', 'invivo')
params_path = utils.make_directory(results_path, 'model_params')
save_path = os.path.join(results_path, 'performance_summary.tsv')

# load dataset
data_path = '../data/invivo_dataset.h5'
train, valid, test = helper.load_invivo_dataset(data_path)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

# loop through models
num_params_all = []
roc_scores = []
pr_scores = []
roc_all = []
pr_all = []
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

	# create neural trainer
	file_path = os.path.join(params_path, model_name)
	nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=file_path)

	# initialize session
	sess = utils.initialize_session()

	# set the best parameters
	nntrainer.set_best_parameters(sess)#, nntrainer.file_path + '_last.ckpt')

	# count number of trainable parameters
	all_params = sess.run(nntrainer.nnmodel.get_trainable_parameters())
	num_params = 0
	for params in all_params:
	    if isinstance(params, list):
	        for param in params:
	            num_params += np.prod(param.shape)
	    else:
	        num_params += np.prod(params.shape)
	num_params_all.append(num_params)

	# get performance metrics
	predictions = nntrainer.get_activations(sess, test, 'output')
	roc, roc_curves = metrics.roc(test['targets'], predictions)
	pr, pr_curves = metrics.pr(test['targets'], predictions)

	roc_scores.append(roc)
	pr_scores.append(pr)
	roc_all.append(roc_curves)
	pr_all.append(pr_curves)
	sess.close()

# save results to file
with open(save_path, 'wb') as f:
	f.write("%s\t%s\t%s\t%s\n"%('model', 'num_params', 'ave roc', 'ave pr'))
	for i, model_name in enumerate(all_models):
		mean_roc = np.mean(roc_scores[i])
		std_roc = np.std(roc_scores[i])
		mean_pr = np.mean(pr_scores[i])
		std_pr = np.std(pr_scores[i])
		num_params = num_params_all[i]
		f.write("%s\t%d\t%.3f$\pm$%.3f\t%.3f$\pm$%.3f\n"%(model_name, num_params, mean_roc, std_roc, mean_pr, std_pr))
