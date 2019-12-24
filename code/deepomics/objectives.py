
import tensorflow as tf
import numpy as np


def binary_cross_entropy(targets, predictions):

	predictions = tf.clip_by_value(predictions,1e-7,1-1e-7)
	return tf.reduce_mean(targets*tf.log(predictions) + (1-targets)*tf.log(1-predictions), axis=get_reduce_axis(targets))


def weighted_binary_cross_entropy(targets, predictions, class_weights):

	predictions = tf.clip_by_value(predictions,1e-7,1-1e-7)
	return tf.reduce_mean(class_weights*(targets*tf.log(predictions) + (1-targets)*tf.log(1-predictions)), axis=get_reduce_axis(targets))

def categorical_cross_entropy(targets, predictions):

	predictions = tf.clip_by_value(predictions,1e-7,1-1e-7)
		#loss = -tf.reduce_mean(tf.reduce_sum(targets*tf.log(predictions), axis=1))
	return tf.reduce_sum(targets*tf.log(predictions), axis=get_reduce_axis(targets))


def squared_error(targets, predictions):

	return tf.reduce_sum(tf.square(targets - predictions), axis=get_reduce_axis(targets))


def categorical_cross_entropy2D(targets, predictions, shape):

	num_categories, num_classes = shape

	# reshape predictions
	predictions = tf.clip_by_value(predictions,1e-7,1-1e-7)
	predictions_reshape = tf.reshape(predictions, [-1, num_classes])

	# reshape targets
	targets_reshape = tf.reshape(targets, [-1, num_classes])

	# get cross engtorpy loss for each class followed by reshaping by categories
	loss_by_sample = tf.reshape(targets_reshape*tf.log(predictions_reshape), [-1, num_categories])

	# reduce sum over categories
	return tf.reduce_sum(loss_by_sample, axis=1)


def elbo_gaussian_gaussian(targets, X_mu, X_logvar, Z_mu, Z_logvar, KL_weight=None):

	# calculate kl divergence
	Z_sigma = tf.sqrt(tf.exp(Z_logvar)+1e-7)
	kl_divergence = 0.5*tf.reduce_sum(1 + 2*tf.log(Z_sigma) - tf.square(Z_mu) - tf.exp(2*tf.log(Z_sigma)), axis=1)

	# calculate reconstructed likelihood
	X_logvar = tf.log(tf.exp(X_logvar) + 1e-7)

	#log_likelihood = -tf.reduce_sum(tf.square(targets-X_mu), axis=1)
	const = tf.constant(-0.5*np.log(2*np.float32(np.pi)), dtype=tf.float32)
	log_likelihood = tf.reduce_sum(const - 0.5*X_logvar - 0.5*tf.divide(tf.square(targets-X_mu),tf.exp(X_logvar)), axis=get_reduce_axis(targets))

	if KL_weight is None:
		KL_weight = 1.0

	return log_likelihood + KL_weight*kl_divergence


def elbo_gaussian_binary(targets, X_mu, Z_mu, Z_logvar, KL_weight=None):

	# calculate kl divergence
	Z_sigma = tf.sqrt(tf.exp(Z_logvar)+1e-7)
	kl_divergence = 0.5*tf.reduce_sum(1 + 2*tf.log(Z_sigma) - tf.square(Z_mu) - tf.exp(2*tf.log(Z_sigma)), axis=1)

	# calculate reconstructed likelihood
	X_mu = tf.clip_by_value(X_mu, 1e-7, 1-1e-7)
	log_likelihood = tf.reduce_sum(targets*tf.log(X_mu) + (1.0-targets)*tf.log(1.0-X_mu), axis=get_reduce_axis(targets))

	if KL_weight is None:
		KL_weight = 1.0


	return log_likelihood + KL_weight*kl_divergence


def elbo_gaussian_softmax(targets, X, Z_mu, Z_logvar, X_shape, KL_weight=None):

	num_categories, num_classes = X_shape

	# calculate kl divergence
	Z_sigma = tf.sqrt(tf.exp(Z_logvar)+1e-7)
	kl_divergence = 0.5*tf.reduce_sum(1 + 2*tf.log(Z_sigma) - tf.square(Z_mu) - tf.exp(2*tf.log(Z_sigma)), axis=1)
	# calculate reconstructed likelihood
	# reshape
	X = tf.clip_by_value(X, 1e-7, 1-1e-7)
	predictions_reshape = tf.reshape(X, [-1, num_classes])
	targets_reshape = tf.reshape(targets, [-1, num_classes])

	# get categorical cross-entropy and reshape by data sample
	loss_by_sample = tf.reshape(tf.reduce_sum(targets_reshape*tf.log(predictions_reshape), axis=1), [-1, num_categories])
	log_likelihood = tf.reduce_sum(loss_by_sample, axis=1)

	if KL_weight is None:
		KL_weight = 1.0

	return log_likelihood + KL_weight*kl_divergence



def elbo_softmax_normal(targets, X, Z, Z_shape, KL_weight=None):

	num_categories, num_classes = Z_shape

	# calculate softmax-gumbel distribution --> approximate categorical distribution
	log_Z = tf.log(Z + 1e-7)
	kl_tmp = tf.reshape(Z*(log_Z - tf.log(1.0/num_classes)), [-1, num_categories, num_classes])
	kl_divergence = tf.reduce_sum(kl_tmp, [1,2])

	# calculate reconstructed likelihood
	log_likelihood = -tf.reduce_sum(tf.square(targets-X), axis=get_reduce_axis(targets))
	#log_likelihood = tf.reduce_sum(const - 0.5*X_logvar - 0.5*tf.divide(tf.square(targets-X_mu),tf.exp(X_logvar)), axis=1)

	if KL_weight is None:
		KL_weight = 1.0

	return log_likelihood + KL_weight*kl_divergence



def elbo_softmax_binary(targets, X, Z, Z_shape, KL_weight=None):

	num_categories, num_classes = Z_shape

	# calculate softmax-gumbel distribution --> approximate categorical distribution
	log_Z = tf.log(Z + 1e-7)
	kl_tmp = tf.reshape(Z*(log_Z - tf.log(1.0/num_classes)), [-1, num_categories, num_classes])
	kl_divergence = tf.reduce_sum(kl_tmp, [1,2])

	# calculate reconstructed likelihood
	X = tf.clip_by_value(X, 1e-7, 1-1e-7)
	log_likelihood = tf.reduce_sum(targets*tf.log(X) + (1.0-targets)*tf.log(1.0-X), axis=get_reduce_axis(targets))

	if KL_weight is None:
		KL_weight = 1.0

	return log_likelihood + KL_weight*kl_divergence

def elbo_softmax_softmax(targets, X, Z, X_shape, Z_shape, KL_weight=None):

	num_categories, num_classes = Z_shape

	# calculate softmax-gumbel distribution --> approximate categorical distribution
	log_Z = tf.log(Z + 1e-7)
	kl_tmp = tf.reshape(Z*(log_Z - tf.log(1.0/num_classes)), [-1, num_categories, num_classes])
	kl_divergence = tf.reduce_sum(kl_tmp, [1,2])

	# calculate reconstructed likelihood

	num_categories, num_classes = X_shape

	# reshape
	X = tf.clip_by_value(X, 1e-7, 1-1e-7)
	predictions_reshape = tf.reshape(X, [-1, num_classes])
	targets_reshape = tf.reshape(targets, [-1, num_classes])

	# get categorical cross-entropy and reshape by data sample
	loss_by_sample = tf.reshape(tf.reduce_sum(targets_reshape*tf.log(predictions_reshape), axis=1), [-1, num_categories])
	log_likelihood = tf.reduce_sum(loss_by_sample, axis=1)

	if KL_weight is None:
		KL_weight = 1.0

	return log_likelihood + KL_weight*kl_divergence



def get_reduce_axis(targets):

	dims = len(targets.get_shape())
	if dims == 2:
		axis = 1
	elif dims == 4:
		axis = [1,2,3]

	return axis
