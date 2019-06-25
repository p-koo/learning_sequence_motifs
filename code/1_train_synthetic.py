from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import helper
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, metrics, fit, visualize

#------------------------------------------------------------------------------------------------


all_models = ['cnn_1', 'cnn_2', 'cnn_4', 'cnn_10', 'cnn_25', 'cnn_50', 'cnn_100',
              'cnn_50_2', 'cnn9_4', 'cnn9_25', 'cnn3_50', 'cnn3_2', 'cnn_2_1',
              'cnn_25_60', 'cnn_25_90', 'cnn_25_120', 'cnn_1_3']

# load dataset
data_path = '../data/synthetic_dataset.h5'
train, valid, test = helper.load_synthetic_dataset(data_path)

# get data shapes
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = [None, train['targets'].shape[1]]

num_trials = 5

for trial in range(num_trials):

    # save path
    results_path = utils.make_directory('../results', 'synthetic_'+str(trial))
    params_path = utils.make_directory(results_path, 'model_params')
    save_path = utils.make_directory(results_path, 'conv_filters')


    # loop through models
    num_params_all = []
    roc_scores = []
    pr_scores = []
    roc_all = []
    pr_all = []
    for model_name in all_models:
        print('model: ' + model_name)

        # load model parameters
        genome_model = helper.import_model(model_name)
        model_layers, optimization = genome_model.model(input_shape, output_shape)

        # build neural network class
        nnmodel = nn.NeuralNet()
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


        # set the best parameters
        nntrainer.set_best_parameters(sess)

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

        # get 1st convolution layer filters
        fmap = nntrainer.get_activations(sess, test, layer='conv1d_0_active')
        W = visualize.activation_pwm(fmap, X=test['inputs'], threshold=0.5, window=19)

        # plot 1st convolution layer filters
        fig = visualize.plot_filter_logos(W, nt_width=50, height=100, norm_factor=None, num_rows=10)
        fig.set_size_inches(100, 100)
        outfile = os.path.join(save_path, model_name+'_conv_filters.pdf')
        fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
        plt.close()

        # save filters as a meme file for Tomtom 
        output_file = os.path.join(save_path, model_name+'.meme')
        utils.meme_generate(W, output_file, factor=None)

        # clip filters about motif to reduce false-positive Tomtom matches 
        W = np.squeeze(np.transpose(W, [3, 2, 0, 1]))
        W_clipped = helper.clip_filters(W, threshold=0.5, pad=3)
        
        # since W is different format, have to use a different function
        output_file = os.path.join(save_path, model_name+'_clip.meme')
        helper.meme_generate(W_clipped, output_file, factor=None) 
        
        sess.close()


    # save results to file
    with open(os.path.join(results_path, 'results.tsv'), 'wb') as f:
        f.write("%s\t%s\t%s\t%s\n"%('model', 'num_params', 'ave roc', 'ave pr'))
        for i, model_name in enumerate(all_models):
            mean_roc = np.mean(roc_scores[i])
            std_roc = np.std(roc_scores[i])
            mean_pr = np.mean(pr_scores[i])
            std_pr = np.std(pr_scores[i])
            num_params = num_params_all[i]
            f.write("%s\t%d\t%.3f$\pm$%.3f\t%.3f$\pm$%.3f\n"%(model_name, num_params, mean_roc, std_roc, mean_pr, std_pr))


