import tensorflow as tf

import math
import numpy as np

def preprocessData(mnist, cand):
    X_train = mnist.train.images
    X_validation = mnist.validation.images
    X_test = mnist.test.images

    y_train = mnist.train.labels.astype("int")
    y_validation = mnist.validation.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    labels = []
    labels.append(y_train)
    labels.append(y_validation)
    labels.append(y_test)

    images = []
    images.append(X_train)
    images.append(X_validation)
    images.append(X_test)

    Data = [[None for x in range(2)] for y in range(3)]

    idx = 0
    for labl in labels:
        idx_prom8 = [idx for idx in range(len(labl)) if cand(labl[idx])]
        Data[idx][0] = images[idx][idx_prom8]
        Data[idx][1] = labl[idx_prom8]
        idx +=1

    return Data


def he_init(fan_in, fan_out):
    stddev = 2 / math.pow(fan_in + fan_out, math.sqrt(2))
    init = tf.truncated_normal(shape=(fan_in, fan_out), means=0, stddev = stddev)
    W = tf.Variable(init, name="kernel")


    return W


def neuron_layer(X, fan_out, kernel_initializer = None, activation=None):
    """

    :param X:
    :param fan_out:
    :param kernel_initializer:
    :param activation:
    :return:
    """

    W = None
    features = int(X.get_shape()[1])
    # features = X.shape[1]
    if kernel_initializer:
        W = kernel_initializer(features, fan_out)
    else:
        stddev = 2/math.pow((features + fan_out), 1/2)
        init = tf.truncated_normal(shape=(features, fan_out), means=0, stddev = stddev)
        W = tf.Variable(init, name="kernel")

    b = tf.Variable(tf.zeros([features]), name="bias")
    z = tf.matmul(X, W) + b

    if not activation:
        z = activation(z)

    return z





def next_batch(Training=None, epoch =0 , batch_idx=0, batch_size=0):
    """

    :param Training: Training하는 데이터
    :param batch_idx:  현재 배치 인덱스
    :param batch_size:  배치 사이즈
    :return:
    """
    if not Training:
        assert "error"

    training_size = len(Training[0])

    np.random.seed(epoch * batch_size + batch_idx )
    indices = np.random.randint(training_size, size=batch_size)
    X_batch = Training[0][indices]
    y_batch = Training[1][indices]
    return X_batch, y_batch



