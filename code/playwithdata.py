from helper import *
import numpy as np
from solution import *


traindataloc, testdataloc = "../data/train.txt", "../data/test.txt"
train_data, train_label = load_features(traindataloc)
test_data, test_label = load_features(testdataloc)

max_iter = 10
learning_rate = 0.1
data = train_data
label = train_label

x = data  # predictors
y = label  # label

w = np.zeros(np.shape(data)[1])
n = len(data)
s = np.dot(w, np.transpose(x))
exponent = np.dot(np.diag(y), s)
denominator = 1 / (1 + np.exp(exponent))  # denominator of gradient equation
YnXn = np.dot(np.diag(y), x)  # numerator of gradient equation
insum = np.dot(np.diag(denominator), YnXn)
gt = -(1 / n) * sum(insum)  # gradient
w -= learning_rate * gt  # Update weights