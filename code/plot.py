from solution import *
from pandas import DataFrame
from matplotlib import pyplot


#Use for testing the training and testing processes of a model
def train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate):
    if modelname == 'thirdorder':
        train_data = thirdorder(train_data)
        test_data = thirdorder(test_data)

    w = logistic_regression(train_data, train_label, max_iter, learning_rate)
    train_acc = accuracy(train_data, train_label, w)
    test_acc = accuracy(test_data, test_label, w)
    return w, train_acc, test_acc


def testing(modelname):
    traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
    train_data,train_label = load_features(traindataloc)
    test_data, test_label = load_features(testdataloc)
    max_iter = [10, 30, 50, 100, 200]
    learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
    train_acc_list = list()
    test_acc_list = list()
    for i, m_iter in enumerate(max_iter):
        _, train_acc, test_acc = train_test_a_model(modelname, train_data, train_label, test_data, test_label, m_iter, learning_rate[0])
        print("Case %d train accuracy:%f  test accuracy: %f"%(i+1, train_acc, test_acc))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    for i, l_rate in enumerate(learning_rate):
        _, train_acc, test_acc = train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter[4], l_rate)
        print("Case %d train accuracy:%f  test accuracy: %f"%(i+6,train_acc, test_acc))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    print("accuracy test done!")
    return train_acc_list, test_acc_list




lr_train, lr_test = testing(None)
lr3_train, lr3_test = testing('thirdorder')

dat = {
    "Train": lr_train,
    "Test" : lr_test,
    "Train Third Order" : lr3_train,
    "Test Third Order": lr3_test
}

df = DataFrame.from_dict(dat)
ax = df.plot()
ax.set_xlabel("Iteration")
ax.set_ylabel("Accuracy")
pyplot.savefig("results.png")
