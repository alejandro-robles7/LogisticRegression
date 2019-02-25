from helper import *



def show_images(data):
    '''
    This function is used for plot image and save it.

    Args:
    data: Two images from train data with shape (2, 16, 16). The shape represents total 2
          images and each image has size 16 by 16.

    Returns:
        Do not return any arguments, just save the images you plot for your report.
    '''
    plt.imsave('five.png', data[0])
    plt.imsave('one.png', data[1])



def show_features(data, label):
    '''
    This function is used for plot a 2-D scatter plot of the features and save it.

    Args:
    data: train features with shape (1561, 2). The shape represents total 1561 samples and
          each sample has 2 features.
    label: train data's label with shape (1561,1).
           1 for digit number 1 and -1 for digit number 5.

    Returns:
    Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
    '''
    fig = plt.figure()
    alpha = 0.5

    for sample_label, sample in zip(label, data):
        if sample_label < 0:
            marker, color = "+", "blue"
        else:
            marker, color = "*", "red"
        plt.scatter(sample[0], sample[1], marker=marker, c=color, alpha=alpha)

    plt.xlabel("Symmetry")
    plt.ylabel("Average Intensity")
    plt.legend(('5', '1'))
    fig.savefig('features.png')


def play_with_data():
    # show the data
    traindataloc = "../data/train.txt"
    nums = 2
    data = load_data(traindataloc)[0:nums,1:]
    [n,d]=data.shape
    w= math.floor(math.sqrt(d))
    data = np.reshape(data, (nums, w, w))
    show_images(data)
    print("play with data done!")


def play_with_features():
    #get data
    traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
    train_data,train_label = load_features(traindataloc)
    test_data, test_label = load_features(testdataloc)
    show_features(train_data[:,1:3],train_label)
    print("play with features done!")


play_with_data()
## test question (b)
play_with_features()
## test question (c)
a = 2