from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import helper
import matplotlib.pyplot as plt
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, visualize

#------------------------------------------------------------------------------------------------


all_models = ['cnn_2', 'cnn_4', 'cnn_10', 'cnn_25', 'cnn_50', 'cnn_100',
			  'cnn_50_2', 'cnn9_4', 'cnn9_25']

# save path
results_path = utils.make_directory('../results', 'invivo')
params_path = utils.make_directory(results_path, 'model_params')
save_path = utils.make_directory(results_path, 'conv_filters')

# get data shapes
input_shape = [None, 1000, 1, 4]
output_shape = [None, 12]

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
	nntrainer.set_best_parameters(sess)

	# get 1st convolution layer filters
	W = nntrainer.get_parameters(sess, layer='conv1d_0')[0]

	# plot 1st convolution layer filters
	fig = visualize.plot_filter_logos(W, nt_width=50, height=100, norm_factor=3, num_rows=10)
	fig.set_size_inches(100, 100)
	outfile = os.path.join(save_path, model_name+'_conv_filters.pdf')
	fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
	plt.close()

	# save filters as a meme file for Tomtom 
	output_file = os.path.join(save_path, model_name+'.meme')
	utils.meme_generate(W, output_file, factor=3)

	# clip filters about motif to reduce false-positive Tomtom matches 
	W = np.squeeze(np.transpose(W, [3, 2, 0, 1]))
	W_clipped = helper.clip_filters(W, threshold=0.5, pad=3)
	
	# since W is different format, have to use a different function
	output_file = os.path.join(save_path, model_name+'_clip.meme')
	helper.meme_generate(W_clipped, output_file, factor=3) 
