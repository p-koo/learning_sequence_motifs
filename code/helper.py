from __future__ import print_function

import os, sys
import h5py
import numpy as np
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency, metrics



def load_invivo_dataset(filepath, verbose=True):
    # setup paths for file handling

    trainmat = h5py.File(filepath, 'r')

    if verbose:
        print("loading training data")
    X_train = np.array(trainmat['X_train']).astype(np.float32).transpose([0,2,3,1])
    y_train = np.array(trainmat['Y_train']).astype(np.float32)

    if verbose:
        print("loading cross-validation data")
    X_valid = np.array(trainmat['X_valid']).astype(np.float32).transpose([0,2,3,1])
    y_valid = np.array(trainmat['Y_valid']).astype(np.int32)

    if verbose:
        print("loading test data")
    X_test = np.array(trainmat['X_test']).astype(np.float32).transpose([0,2,3,1])
    y_test = np.array(trainmat['Y_test']).astype(np.int32)

    train = {'inputs': X_train, 'targets': y_train}
    valid = {'inputs': X_valid, 'targets': y_valid}
    test = {'inputs': X_test, 'targets': y_test}

    return train, valid, test


def load_synthetic_dataset(filepath, verbose=True):
    # setup paths for file handling

    trainmat = h5py.File(filepath, 'r')

    if verbose:
        print("loading training data")
    X_train = np.array(trainmat['X_train']).astype(np.float32)
    y_train = np.array(trainmat['Y_train']).astype(np.float32)

    if verbose:
        print("loading cross-validation data")
    X_valid = np.array(trainmat['X_valid']).astype(np.float32)
    y_valid = np.array(trainmat['Y_valid']).astype(np.int32)

    if verbose:
        print("loading test data")
    X_test = np.array(trainmat['X_test']).astype(np.float32)
    y_test = np.array(trainmat['Y_test']).astype(np.int32)


    X_train = np.expand_dims(X_train, axis=3).transpose([0,2,3,1])
    X_valid = np.expand_dims(X_valid, axis=3).transpose([0,2,3,1])
    X_test = np.expand_dims(X_test, axis=3).transpose([0,2,3,1])

    train = {'inputs': X_train, 'targets': y_train}
    valid = {'inputs': X_valid, 'targets': y_valid}
    test = {'inputs': X_test, 'targets': y_test}

    return train, valid, test


def load_synthetic_TF_models(filepath, dataset='test'):
    # setup paths for file handling

    trainmat = h5py.File(filepath, 'r')
    if dataset == 'train':
        return np.array(trainmat['model_train']).astype(np.float32)
    elif dataset == 'valid':
        return np.array(trainmat['model_valid']).astype(np.float32)
    elif dataset == 'test':
        return np.array(trainmat['model_test']).astype(np.float32)


def import_model(model_name):

    # get model
    if model_name == 'cnn_2':
        from models import cnn_2 as genome_model
    elif model_name == 'cnn_4':
        from models import cnn_4 as genome_model
    elif model_name == 'cnn_10':
        from models import cnn_10 as genome_model
    elif model_name == 'cnn_25':
        from models import cnn_25 as genome_model
    elif model_name == 'cnn_50':
        from models import cnn_50 as genome_model
    elif model_name == 'cnn_100':
        from models import cnn_100 as genome_model
    elif model_name == 'cnn_50_2':
        from models import cnn_50_2 as genome_model
    elif model_name == 'cnn9_4':
        from models import cnn9_4 as genome_model
    elif model_name == 'cnn9_25':
        from models import cnn9_25 as genome_model
    elif model_name == 'cnn3_2':
        from models import cnn3_2 as genome_model
    elif model_name == 'cnn3_50':
        from models import cnn3_50 as genome_model

    return genome_model



def backprop(X, params, layer='output', class_index=None, batch_size=128, method='guided', dataset='multitask'):
    tf.reset_default_graph()

    genome_model = helper.import_model(params['model_name'])
    model_layers, optimization = genome_model.model(params['input_shape'], params['output_shape'])


    nnmodel = nn.NeuralNet()
    nnmodel.build_layers(model_layers, optimization, method=method, use_scope=True)
    nntrainer = nn.NeuralTrainer(nnmodel, save='best', filepath=params['model_path'])

    # setup session and restore optimal parameters
    sess = utils.initialize_session(nnmodel.placeholders)
    nntrainer.set_best_parameters(sess, params['model_path'], verbose=0)

    # backprop saliency
    if layer == 'output':
        layer = list(nnmodel.network.keys())[-2]

    saliency = nntrainer.get_saliency(sess, X, nnmodel.network[layer], class_index=class_index, batch_size=batch_size)

    sess.close()
    tf.reset_default_graph()
    return saliency


def stochastic_backprop(X, params, layer='output', class_index=None, batch_size=128,
                       num_average=200, threshold=5.0, method='guided', dataset='multitask'):
    tf.reset_default_graph()

    # build new graph
    #g = tf.get_default_graph()
    #with g.gradient_override_map({'Relu': 'GuidedRelu'}):
    if dataset == 'multitask':
        model_layers, optimization, genome_model = load_multitask_model(params['model_name'], params['input_shape'], params['output_shape'],
                                                       params['dropout_status'], params['l2_status'], params['bn_status'])
    else:
        model_layers, optimization, genome_model = load_regulatory_code_model(params['model_name'], params['input_shape'], params['output_shape'],
                                                   params['dropout_status'], params['l2_status'], params['bn_status'])

    nnmodel = nn.NeuralNet()
    nnmodel.build_layers(model_layers, optimization, method=method, use_scope=True)
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

    return saliency, np.array(counts) #np.vstack(saliency), np.array(counts)



def entropy(X):
    information = np.log2(4) - np.sum(-X*np.log2(X+1e-10),axis=0)
    return information



def cosine_distance(X_norm, X_model):

    norm1 = np.sqrt(np.sum(X_norm**2, axis=0))
    norm2 = np.sqrt(np.sum(X_model**2, axis=0))

    dist = np.sum(X_norm*X_model, axis=0)/norm1/norm2
    return dist



def info_weighted_distance(X_saliency, X_model):

    X_norm = utils.normalize_pwm(X_saliency, factor=3)
    cd = cosine_distance(X_norm, X_model)
    model_info = entropy(X_model)
    good_info = np.sum(model_info*cd)/np.sum(model_info)

    inv_model_info = -(model_info-2)
    inv_cd = -(cd-1)
    bad_info = np.sum(inv_cd*inv_model_info)/np.sum(inv_model_info)
    return good_info, bad_info


def mean_info_distance(backprop_saliency, X_model):
    info = []
    for j, gs in enumerate(backprop_saliency):
        X_saliency = np.squeeze(gs).T
        good_info, bad_info = info_weighted_distance(X_saliency, X_model[j])
        info.append([good_info, bad_info])
    info = np.array(info)
    mean_info = np.nanmean(info, axis=0)
    return mean_info



def clip_filters(W, threshold=0.5, pad=3):
    num_filters, _, filter_length = W.shape

    W_clipped = []
    for i in range(num_filters):
        w = utils.normalize_pwm(W[i], factor=3)
        entropy = np.log2(4) + np.sum(w*np.log2(w+1e-7), axis=0)
        index = np.where(entropy > threshold)[0]
        if index.any():
            start = np.maximum(np.min(index)-pad, 0)
            end = np.minimum(np.max(index)+pad+1, filter_length)
            W_clipped.append(W[i,:,start:end])
        else:
            W_clipped.append(W[i,:,:])

    return W_clipped



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

    for j in range(len(W)):
        if factor:
            pwm = utils.normalize_pwm(W[j], factor=factor)
        else:
            pwm = W[j]
        f.write('MOTIF %s%d \n' % (prefix, j))
        f.write('letter-probability matrix: alength= 4 w= %d nsites= %d \n' % (pwm.shape[1], pwm.shape[1]))
        for i in range(pwm.shape[1]):
            f.write('%.4f %.4f %.4f %.4f \n' % tuple(pwm[:,i]))
        f.write('\n')

    f.close()

