import tensorflow as tf
import numpy as np

from deepomics import objectives

__all__ = [
	"build_updates",
	"build_loss",
	"cost_function"
]


def build_updates(optimization):
	"""Build updates"""

	if 'optimizer' in optimization.keys():
		optimizer = optimization['optimizer']
	else:
		optimizer = 'adam'
		optimization['learning_rate'] = 0.001

	if optimizer == 'sgd':
		learning_rate = 0.005
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'adadelta'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.GradientDescentOptimizer(learning_rate=learning_rate,
												 use_locking=use_locking,
												 name=name)

	elif optimizer == 'momentum':
		learning_rate = 0.005
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		momentum = 0.9
		if 'momentum' in optimization.keys():
			momentum = optimization['momentum']
		use_nesterov = True
		if 'use_nesterov' in optimization.keys():
			use_nesterov = optimization['use_nesterov']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'momenum'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.MomentumOptimizer(learning_rate=learning_rate,
										  momentum=momentum,
										  use_nesterov=use_nesterov,
										  use_locking=use_locking,
										  name=name)

	elif optimizer == 'adam':
		learning_rate = 0.001
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		beta1 = 0.95
		if 'beta1' in optimization.keys():
			beta1 = optimization['beta1']
		beta2 = 0.999
		if 'beta2' in optimization.keys():
			beta2 = optimization['beta2']
		epsilon = 1e-08
		if 'epsilon' in optimization.keys():
			epsilon = optimization['epsilon']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'adam'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.AdamOptimizer(learning_rate=learning_rate,
									  beta1=beta1,
									  beta2=beta2,
									  epsilon=epsilon,
									  use_locking=use_locking,
									  name=name)

	elif optimizer == 'rmsprop':
		learning_rate = 0.001
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		rho = 0.95
		if 'rho' in optimization.keys():
			rho = optimization['rho']
		epsilon = 1e-08
		if 'epsilon' in optimization.keys():
			epsilon = optimization['epsilon']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'rmsprop'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
										 rho=rho,
										 epsilon=epsilon,
										 use_locking=use_locking,
										 name=name)

	elif optimizer == 'adadelta':
		learning_rate = 0.001
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		rho = 0.95
		if 'rho' in optimization.keys():
			rho = optimization['rho']
		epsilon = 1e-08
		if 'epsilon' in optimization.keys():
			epsilon = optimization['epsilon']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'adadelta'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
										  rho=rho,
										  epsilon=epsilon,
										  use_locking=use_locking,
										  name=name)

	elif optimizer == 'adagrad':
		learning_rate = 0.001
		if 'learning_rate' in optimization.keys():
			leanring_rate = optimization['learning_rate']
		initial_accumulator_value = 0.95
		if 'initial_accumulator_value' in optimization.keys():
			initial_accumulator_value = optimization['initial_accumulator_value']
		use_locking = False
		if 'use_locking' in optimization.keys():
			use_locking = optimization['use_locking']
		name = 'adagrad'
		if 'name' in optimization.keys():
			name = optimization['name']
		return tf.train.AdagradOptimizer(learning_rate=learning_rate,
										 initial_accumulator_value=initial_accumulator_value,
										 use_locking=use_locking,
										 name=name)



def build_loss(network, predictions, targets, optimization):

	# build loss function
	if 'label_smoothing' not in optimization.keys():
		optimization['label_smoothing'] = 0
	loss = cost_function(network, targets, optimization)

	reg = 0
	if 'l1' in optimization.keys():
		l1 = get_l1_parameters(network)
		reg += tf.reduce_sum(tf.abs(l1)) * optimization['l1']

	if 'l2' in optimization.keys():
		l2 = get_l2_parameters(network)
		reg += tf.reduce_sum(tf.square(l2)) * optimization['l2']
	
	return loss, reg


def cost_function(network, targets, optimization):

	objective = optimization['objective']
	label_smoothing = optimization['label_smoothing']

	if objective == 'binary':

		if label_smoothing > 0:
			  targets = (targets*(1-label_smoothing) + 0.5*label_smoothing)

		predictions = network['output'].get_output()

		if 'class_weights' in optimization:
			if not optimization['class_weights']:
				loss = -objectives.weighted_binary_cross_entropy(targets, predictions, )
			else:
				loss = -objectives.binary_cross_entropy(targets, predictions)
		else:
			loss = -objectives.binary_cross_entropy(targets, predictions)


	elif objective == 'categorical':

		if label_smoothing > 0:
			num_classes = targets.get_shape()[-1].value
			smooth_positives = 1.0 - label_smoothing
			smooth_negatives = label_smoothing/num_classes
			targets = targets*smooth_positives + smooth_negatives

		predictions = network['output'].get_output()
		loss = -objectives.categorical_cross_entropy(targets, predictions)


	elif objective == 'squared_error':

		predictions = network['output'].get_output()
		loss = objectives.squared_error(targets, predictions)


	elif objective == 'categorical2D':

		if 'output' in network:
			predictions = network['output'].get_output()
		else:
			predictions = network['X'].get_output()

		loss = -objectives.categorical_cross_entropy2D(targets, predictions, optimization['softmax_shape'])


	elif objective == 'elbo_gaussian_gaussian':

		if 'KL_weight' in optimization:
			KL_weight = optimization['KL_weight']
		else:
			KL_weight = 1.0

		X = network['X'].get_output()
		X_logvar = network['X_logvar'].get_output()
		Z_mu = network['Z_mu'].get_output()
		Z_logvar = network['Z_logvar'].get_output()

		loss = -objectives.elbo_gaussian_gaussian(targets, X, X_logvar, Z_mu, Z_logvar, KL_weight=KL_weight)


	elif objective == 'elbo_gaussian_binary':

		if 'KL_weight' in optimization:
			KL_weight = optimization['KL_weight']
		else:
			KL_weight = 1.0

		if label_smoothing > 0:
			  targets = (targets*(1-label_smoothing) + 0.5*label_smoothing)

		X = network['X'].get_output()
		Z_mu = network['Z_mu'].get_output()
		Z_logvar = network['Z_logvar'].get_output()

		loss = -objectives.elbo_gaussian_binary(targets, X, Z_mu, Z_logvar, KL_weight=KL_weight)


	elif objective == 'elbo_gaussian_softmax':

		if 'KL_weight' in optimization:
			KL_weight = optimization['KL_weight']
		else:
			KL_weight = 1.0

		if label_smoothing > 0:
			num_classes = targets.get_shape()[-1].value
			smooth_positives = 1.0 - label_smoothing
			smooth_negatives = label_smoothing/num_classes
			targets = targets*smooth_positives + smooth_negatives

		X = network['X'].get_output()
		Z_mu = network['Z_mu'].get_output()
		Z_logvar = network['Z_logvar'].get_output()

		loss = -objectives.elbo_gaussian_softmax(targets, X, Z_mu, Z_logvar, optimization['softmax_shape'], KL_weight=KL_weight)


	elif objective == 'elbo_softmax_normal':

		if 'KL_weight' in optimization:
			KL_weight = optimization['KL_weight']
		else:
			KL_weight = 1.0

		X = network['X'].get_output()
		Z = network['Z'].get_output()

		loss = -objectives.elbo_softmax_normal(targets, X, Z, optimization['Z_shape'], KL_weight=KL_weight)


	elif objective == 'elbo_softmax_binary':

		if 'KL_weight' in optimization:
			KL_weight = optimization['KL_weight']
		else:
			KL_weight = 1.0

		if label_smoothing > 0:
			  targets = (targets*(1-label_smoothing) + 0.5*label_smoothing)

		X = network['X'].get_output()
		Z = network['Z'].get_output()

		loss = -objectives.elbo_softmax_binary(targets, X, Z, optimization['Z_shape'], KL_weight=KL_weight)


	elif objective == 'elbo_softmax_softmax':

		if 'KL_weight' in optimization:
			KL_weight = optimization['KL_weight']
		else:
			KL_weight = 1.0

		if label_smoothing > 0:
			num_classes = targets.get_shape()[-1].value
			smooth_positives = 1.0 - label_smoothing
			smooth_negatives = label_smoothing/num_classes
			targets = targets*smooth_positives + smooth_negatives

		X = network['X'].get_output()
		Z = network['Z'].get_output()
		loss = -objectives.elbo_softmax_softmax(targets, X, Z, optimization['softmax_shape'], optimization['Z_shape'], KL_weight=KL_weight)

	return loss


def get_l1_parameters(net):
	params = []
	for layer in net:
		if hasattr(net[layer], 'is_l1_regularize'):
			if net[layer].is_l1_regularize():
				variables = net[layer].get_variable()
				if isinstance(variables, list):
					for var in variables:
						params.append(var.get_variable())
				else:
					params.append(variables.get_variable())
	return merge_parameters(params)


def get_l2_parameters(net):
	params = []
	for layer in net:
		if hasattr(net[layer], 'is_l2_regularize'):
			if net[layer].is_l2_regularize():
				variables = net[layer].get_variable()
				if isinstance(variables, list):
					for var in variables:
						params.append(var.get_variable())
				else:
					params.append(variables.get_variable())
	return merge_parameters(params)



def merge_parameters(params):
	all_params = []
	for param in params:
		all_params = tf.concat([all_params, tf.reshape(param, [-1,])], axis=0)
	return all_params
