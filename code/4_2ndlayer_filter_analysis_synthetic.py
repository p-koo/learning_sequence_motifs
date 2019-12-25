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


# plot 2nd layer filters of CNN-1
model_name = 'cnn_1'
num_trials = 5

for trial in range(num_trials):

    results_path = utils.make_directory('../results', 'synthetic_'+str(trial))
    params_path = utils.make_directory(results_path, 'model_params')
    save_path = utils.make_directory(results_path, 'conv_filters')

    # load dataset
    data_path = '../data/synthetic_dataset.h5'
    train, valid, test = helper.load_synthetic_dataset(data_path)

    # get data shapes
    input_shape = list(train['inputs'].shape)
    input_shape[0] = None
    output_shape = [None, train['targets'].shape[1]]

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

    # set the best parameters
    nntrainer.set_best_parameters(sess)

    # get 2nd convolution layer filters
    fmap = nntrainer.get_activations(sess, test, layer='conv1d_1_active')
    W = visualize.activation_pwm(fmap, X=test['inputs'], threshold=0.5, window=19)

    # plot 2nd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(0,30)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_2nd_layer_0.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()


    # plot 2nd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(30,60)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_2nd_layer_1.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()


    # plot 2nd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(60,90)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_2nd_layer_2.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()

    # plot 2nd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(90,120)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_2nd_layer_3.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()

    # save filters as a meme file for Tomtom 
    output_file = os.path.join(save_path, model_name+'_2nd_layer.meme')
    utils.meme_generate(W, output_file, factor=None)

    # clip filters about motif to reduce false-positive Tomtom matches 
    W = np.squeeze(np.transpose(W, [3, 2, 0, 1]))
    W_clipped = helper.clip_filters(W, threshold=0.5, pad=3)

    # since W is different format, have to use a different function
    output_file = os.path.join(save_path, model_name+'_2nd_layer_clip.meme')
    helper.meme_generate(W_clipped, output_file, factor=None) 




# plot 2nd layer and 3rd layer filters of CNN-1-3
model_name = 'cnn_1_3'
num_trials = 5

for trial in range(num_trials):

    results_path = utils.make_directory('../results', 'synthetic_'+str(trial))
    params_path = utils.make_directory(results_path, 'model_params')
    save_path = utils.make_directory(results_path, 'conv_filters')

    # load dataset
    data_path = '../data/synthetic_dataset.h5'
    train, valid, test = helper.load_synthetic_dataset(data_path)

    # get data shapes
    input_shape = list(train['inputs'].shape)
    input_shape[0] = None
    output_shape = [None, train['targets'].shape[1]]

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

    # set the best parameters
    nntrainer.set_best_parameters(sess)

    # get 1st convolution layer filters
    fmap = nntrainer.get_activations(sess, test, layer='conv1d_0_active')
    W = visualize.activation_pwm(fmap, X=test['inputs'], threshold=0.5, window=19)

    # plot 1st convolution layer filters
    fig = visualize.plot_filter_logos(W, nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_1st_layer.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()


    # save filters as a meme file for Tomtom 
    output_file = os.path.join(save_path, model_name+'_1st_layer.meme')
    utils.meme_generate(W, output_file, factor=None)

    # clip filters about motif to reduce false-positive Tomtom matches 
    W = np.squeeze(np.transpose(W, [3, 2, 0, 1]))
    W_clipped = helper.clip_filters(W, threshold=0.5, pad=3)

    # since W is different format, have to use a different function
    output_file = os.path.join(save_path, model_name+'_2st_layer_clip.meme')
    helper.meme_generate(W_clipped, output_file, factor=None) 

    
    #--------------------------------------------------------------------------------

    # get 2nd convolution layer filters
    fmap = nntrainer.get_activations(sess, test, layer='conv1d_1_active')
    W = visualize.activation_pwm(fmap, X=test['inputs'], threshold=0.5, window=19)

    # plot 2nd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(0,30)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_2nd_layer_0.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()


    # plot 2nd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(30,60)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_2nd_layer_1.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()


    # plot 2nd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(60,90)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_2nd_layer_2.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()

    # plot 2nd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(90,120)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_2nd_layer_3.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()

    # save filters as a meme file for Tomtom 
    output_file = os.path.join(save_path, model_name+'_2nd_layer.meme')
    utils.meme_generate(W, output_file, factor=None)

    # clip filters about motif to reduce false-positive Tomtom matches 
    W = np.squeeze(np.transpose(W, [3, 2, 0, 1]))
    W_clipped = helper.clip_filters(W, threshold=0.5, pad=3)

    # since W is different format, have to use a different function
    output_file = os.path.join(save_path, model_name+'_2nd_layer_clip.meme')
    helper.meme_generate(W_clipped, output_file, factor=None) 


    #--------------------------------------------------------------------------------
    
    # get 3rd convolution layer filters
    fmap = nntrainer.get_activations(sess, test, layer='conv1d_2_active')
    W = visualize.activation_pwm(fmap, X=test['inputs'], threshold=0.5, window=19)

    # plot 3rd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(0,30)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_3rd_layer_0.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()


    # plot 3rd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(30,60)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_3rd_layer_1.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()


    # plot 3rd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(60,90)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_3rd_layer_2.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()

    # plot 3rd convolution layer filters
    fig = visualize.plot_filter_logos(W[:,:,:,range(90,120)], nt_width=50, height=100, norm_factor=None, num_rows=10)
    fig.set_size_inches(100, 100)
    outfile = os.path.join(save_path, model_name+'_filters_3rd_layer_3.pdf')
    fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()

    # save filters as a meme file for Tomtom 
    output_file = os.path.join(save_path, model_name+'_3rd_layer.meme')
    utils.meme_generate(W, output_file, factor=None)

    # clip filters about motif to reduce false-positive Tomtom matches 
    W = np.squeeze(np.transpose(W, [3, 2, 0, 1]))
    W_clipped = helper.clip_filters(W, threshold=0.5, pad=3)

    # since W is different format, have to use a different function
    output_file = os.path.join(save_path, model_name+'_3rd_layer_clip.meme')
    helper.meme_generate(W_clipped, output_file, factor=None) 

