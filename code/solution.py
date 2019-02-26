import numpy as np 
from helper import *
'''
Homework2: logistic regression classifier
'''


def logistic_regression(data, label, max_iter, learning_rate):
	'''
	The logistic regression classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1).
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update

	Returns:
		w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
	'''
	w = np.zeros(np.shape(data)[1])
	x = data
	y = label
	n = len(data)
	for iteration in range(max_iter):
		score = np.dot(w, np.transpose(x))
		denominator = 1 / (1 + np.exp(np.dot(np.diag(y), score)))
		numerator = np.dot(np.diag(y), x)
		gradient_descent = -(1 / n) * sum(np.dot(np.diag(denominator), numerator))
		w -= learning_rate * gradient_descent
	return w


def thirdorder(data):
	'''
	This function is used for a 3rd order polynomial transform of the data.
	Args:
	data: input data with shape (:, 3) the first dimension represents
		  total samples (training: 1561; testing: 424) and the
		  second dimesion represents total features.

	Return:
		result: A numpy array format new data with shape (:,10), which using
		a 3rd order polynomial transformation to extend the feature numbers
		from 3 to 10.
		The first dimension represents total samples (training: 1561; testing: 424)
		and the second dimesion represents total features.
	'''
	ones = data[:, 0]
	x1 = data[:, 1]
	x2 = data[:, 2]
	transform_data = np.array([ones, x1, x2, x1 * x2, np.power(x1, 2),
		np.power(x2, 2), np.power(x1, 2) * x2,
		x1 * np.power(x2, 2),
		np.power(x1, 2) * np.power(x2, 2),
		np.power(x1, 3), np.power(x2, 3)])
	return transform_data.T


def accuracy(x, y, w):
	'''
		This function is used to compute accuracy of a logsitic regression model.

		Args:
		x: input data with shape (n, d), where n represents total data samples and d represents
			total feature numbers of a certain data sample.
		y: corresponding label of x with shape(n, 1), where n represents total data samples.
		w: the seperator learnt from logistic regression function with shape (d, 1),
			where d represents total feature numbers of a certain data sample.

		Return
			accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
			which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
	'''
	probability = 1 / (1 + np.exp(-np.dot(x, w)))
	probability.flat[probability > .5] = 1
	probability.flat[probability <= .5] = -1
	return (probability == y).mean()
