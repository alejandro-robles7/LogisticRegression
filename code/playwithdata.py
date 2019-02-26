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

x2 = thirdorder(x)

w = logistic_regression(x, y,max_iter, learning_rate)
w2 = logistic_regression(x2, y,  max_iter, learning_rate)

probability = 1 / (1 + np.exp(-np.dot(x, w)))
probability.flat[probability > .5] = 1
probability.flat[probability <= .5] = -1
accuracy = (probability == y).mean()
accuracy