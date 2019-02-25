from helper import *
import numpy as np


data = load_features("../data/train.txt")

label = 2
max_iter = 4
learning_rate = 5

w = np.zeros(np.shape(data)[1])
for i in range(0, max_iter):
    x = data  # predictors
    y = label  # label
    WTXn = np.dot(w, np.transpose(x))
    exponent = np.dot(np.diag(y), WTXn)
    denominator = 1 / (1 + np.exp(exponent))  # denominator of gradient equation
    YnXn = np.dot(np.diag(y), x)  # numerator of gradient equation
    insum = np.dot(np.diag(denominator), YnXn)
    gt = -(1 / len(data)) * sum(insum)  # gradient
    w = w - (learning_rate * gt)