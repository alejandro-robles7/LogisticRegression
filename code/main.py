from helper import *
from solution import *

# Please Change if location is not working
traindataloc, testdataloc = "../data/train.txt", "../data/test.txt"


#Use for testing the training and testing processes of a model
def train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate):
	'''
	you should try various number of max_iter and learning_rate
	'''

	if modelname == 'thirdorder':
		train_data = thirdorder(train_data)
		test_data = thirdorder(test_data)

	w = logistic_regression(train_data, train_label, max_iter, learning_rate)
	train_acc = accuracy(train_data, train_label, w)
	test_acc = accuracy(test_data, test_label, w)

	return w, train_acc, test_acc




def test_logistic_regression():

	'''
	you should try various number of max_iter and learning_rate
	'''
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	modelname = 'regular'

	max_iter = [10, 30, 50, 100, 200]
	result_mi = [(0.973,0.948), (0.971, 0.950), (0.980, 0.962), (0.981, 0.959), (0.977, 0.948)]
	learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
	result_lr = [(0.981, 0.962), (0.981, 0.962), (0.981, 0.962), (0.981, 0.962), (0.981, 0.962)]
	for i, m_iter in enumerate(max_iter):
		_, train_acc, test_acc = train_test_a_model(modelname, train_data, train_label, test_data, test_label, m_iter, learning_rate[0])
		print("Case %d train accuracy:%f  test accuracy: %f"%(i+1, train_acc, test_acc))
	for i, l_rate in enumerate(learning_rate):
		_, train_acc, test_acc = train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter[4], l_rate)
		print("Case %d train accuracy:%f  test accuracy: %f"%(i+6,train_acc, test_acc))
	print("accuracy test done!")

def test_thirdorder_logistic_regression():
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	modelname = 'thirdorder'

	max_iter = [10, 30, 50, 100, 200]
	learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
	for i, m_iter in enumerate(max_iter):
		_, train_acc, test_acc = train_test_a_model(modelname, train_data, train_label, test_data, test_label, m_iter, learning_rate[0])
		print("Case %d train accuracy:%f  test accuracy: %f"%(i+1, train_acc, test_acc))
	for i, l_rate in enumerate(learning_rate):
		_, train_acc, test_acc = train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter[4], l_rate)
		print("Case %d train accuracy:%f  test accuracy: %f"%(i+6,train_acc, test_acc))
	print("accuracy test done!")



if __name__ == '__main__':
	test_logistic_regression()
	test_thirdorder_logistic_regression()
