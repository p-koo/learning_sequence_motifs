from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import h5py
import numpy as np
from deepomics import utils



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
    elif model_name == 'cnn_2_1':
        from models import cnn_2_1 as genome_model

    elif model_name == 'cnn_2_exp_b3':
        from models import cnn_2_exp_b3 as genome_model
    elif model_name == 'cnn_2_exp_b2_norm':
        from models import cnn_2_exp_b2_norm as genome_model
    elif model_name == 'cnn_2_exp_b_norm2':
        from models import cnn_2_exp_b_norm2 as genome_model
    elif model_name == 'cnn_2_exp_norm3':
        from models import cnn_2_exp_norm3 as genome_model
    elif model_name == 'cnn_2_exp_norm2_b':
        from models import cnn_2_exp_norm2_b as genome_model
    elif model_name == 'cnn_2_exp_norm_b2':
        from models import cnn_2_exp_norm_b2 as genome_model
    elif model_name == 'cnn_2_exp_none_norm2':
        from models import cnn_2_exp_none_norm2 as genome_model
    elif model_name == 'cnn_2_exp_none_b2':
        from models import cnn_2_exp_none_b2 as genome_model

    elif model_name == 'cnn_2_relu_b3':
        from models import cnn_2_relu_b3 as genome_model
    elif model_name == 'cnn_2_relu_b2_norm':
        from models import cnn_2_relu_b2_norm as genome_model
    elif model_name == 'cnn_2_relu_b_norm2':
        from models import cnn_2_relu_b_norm2 as genome_model
    elif model_name == 'cnn_2_relu_norm_b2':
        from models import cnn_2_relu_norm_b2 as genome_model
    elif model_name == 'cnn_2_relu_norm2_b':
        from models import cnn_2_relu_norm2_b as genome_model
    elif model_name == 'cnn_2_relu_norm3':
        from models import cnn_2_relu_norm3 as genome_model
    elif model_name == 'cnn_2_relu_none_norm2':
        from models import cnn_2_relu_none_norm2 as genome_model
    elif model_name == 'cnn_2_relu_none_b2':
        from models import cnn_2_relu_none_b2 as genome_model

    return genome_model


all_models = ['cnn_2_exp_b3', 'cnn_2_exp_b2_norm', 'cnn_2_exp_b_norm2', 'cnn_2_exp_norm3', 'cnn_2_exp_none_norm2', 'cnn_2_exp_none_b2',
              'cnn_2_relu_b3', 'cnn_2_relu_b2_norm', 'cnn_2_relu_b_norm2', 'cnn_2_relu_norm3', 'cnn_2_relu_none_norm2', 'cnn_2_relu_none_b2']



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

