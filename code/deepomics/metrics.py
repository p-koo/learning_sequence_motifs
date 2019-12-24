import os, sys
import numpy as np
from six.moves import cPickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score
from scipy import stats


__all__ = [
	"pearsonr",
	"rsquare",
	"accuracy",
	"roc",
	"pr",
	"calculate_metrics"
]



def pearsonr(label, prediction):
	ndim = np.ndim(label)
	if ndim == 1:
		corr = [stats.pearsonr(label, prediction)]
	else:		
		num_labels = label.shape[1]
		corr = []
		for i in range(num_labels):
			#corr.append(np.corrcoef(label[:,i], prediction[:,i]))
			corr.append(stats.pearsonr(label[:,i], prediction[:,i])[0])
		
	return corr


def rsquare(label, prediction):
	ndim = np.ndim(label)
	if ndim == 1:
		y = label
		X = prediction
		m = np.dot(X,y)/np.dot(X, X)
		resid = y - m*X; 
		ym = y - np.mean(y); 
		rsqr2 = 1 - np.dot(resid.T,resid)/ np.dot(ym.T, ym);
		metric = [rsqr2]
		slope = [m]
	else:		
		num_labels = label.shape[1]
		metric = []
		slope = []
		for i in range(num_labels):
			y = label[:,i]
			X = prediction[:,i]
			m = np.dot(X,y)/np.dot(X, X)
			resid = y - m*X; 
			ym = y - np.mean(y); 
			rsqr2 = 1 - np.dot(resid.T,resid)/ np.dot(ym.T, ym);
			metric.append(rsqr2)
			slope.append(m)
	return metric, slope


def accuracy(label, prediction):
	ndim = np.ndim(label)
	if ndim == 1:
		metric = np.array(accuracy_score(label, np.round(prediction)))
	else:
		num_labels = label.shape[1]
		metric = np.zeros((num_labels))
		for i in range(num_labels):
			metric[i] = accuracy_score(label[:,i], np.round(prediction[:,i]))
	return metric


def roc(label, prediction):
	ndim = np.ndim(label)
	if ndim == 1:
		fpr, tpr, thresholds = roc_curve(label, prediction)
		score = auc(fpr, tpr)
		score = np.array(score)
		curves = [(fpr, tpr)]
	else:
		num_labels = label.shape[1]
		curves = []
		metric = np.zeros((num_labels))
		for i in range(num_labels):
			fpr, tpr, thresholds = roc_curve(label[:,i], prediction[:,i])
			score = auc(fpr, tpr)
			metric[i]= score
			curves.append((fpr, tpr))
	return metric, curves


def pr(label, prediction):
	ndim = np.ndim(label)
	if ndim == 1:
		precision, recall, thresholds = precision_recall_curve(label, prediction)
		score = auc(recall, precision)
		metric = np.array(score)
		curves = [(precision, recall)]
	else:
		num_labels = label.shape[1]
		curves = []
		metric = np.zeros((num_labels))
		for i in range(num_labels):
			precision, recall, thresholds = precision_recall_curve(label[:,i], prediction[:,i])
			score = auc(recall, precision)
			metric[i] = score
			curves.append((precision, recall))
	return metric, curves


def calculate_metrics(label, prediction, objective):
	"""calculate metrics for classification"""

	if (objective == "binary") | (objective == 'hinge'):
		ndim = np.ndim(label)
		if ndim == 1:
			label = one_hot_labels(label)
		correct = accuracy(label, prediction)
		auc_roc, roc_curves = roc(label, prediction)
		auc_pr, pr_curves = pr(label, prediction)
		mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr)]
		std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]

	elif objective == "categorical":
		correct = np.mean(np.equal(np.argmax(label, axis=1), np.argmax(prediction, axis=1)))
		auc_roc, roc_curves = roc(label, prediction)
		auc_pr, pr_curves = pr(label, prediction)
		mean = [np.nanmean(correct), np.nanmean(auc_roc), np.nanmean(auc_pr)]
		std = [np.nanstd(correct), np.nanstd(auc_roc), np.nanstd(auc_pr)]

	elif (objective == 'squared_error') | (objective == 'kl_divergence') | (objective == 'cdf'):
		corr = pearsonr(label,prediction)
		rsqr, slope = rsquare(label, prediction)
		mean = [np.nanmean(corr), np.nanmean(rsqr), np.nanmean(slope)]
		std = [np.nanstd(corr), np.nanstd(rsqr), np.nanstd(slope)]

	else:
		mean = 0
		std = 0

	return [mean, std]

