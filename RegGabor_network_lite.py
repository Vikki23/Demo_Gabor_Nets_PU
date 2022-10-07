"""
Gabor-Nets Convolutions Lite

A simplified API for RegGabor_network_ops
"""

from RegGabor_network_ops import *


def conv2d(x, n_theta, n_omega, ksize, strides=(1,1,1,1), padding='VALID', stddev=0.4, even_initial = True, name='conv2d'):
    """
    2D regular Gabor Convolution lite: construct 2D Gabor kernels and convolve with inputs

    theta, omega and phase offset are set evenly
    the number of filters is set according to the numbers of theta, omega, and phase off
    Learn sigma, theta, omega, phase offset

    :param x: input tf tensor, shape [batchsize,height,width,channels]
    :param n_theta: number of theta initializations (int)
    :param n_omega: number of omegas initializations (int)
    :param ksize: size of square filters of 2D Gabor kernels (int)
    :param strides: stride size (4-tuple: default (1,1,1,1))
    :param padding: SAME or VALID (default: VALID)
    :param stddev: scale of variable initialization (default 0.2)
    :param even_initial: whether adopt even initialization strategy (True or False, default True)
    :param name: (default 'conv2d')

    :return R: outputs of Gabor kernels, shape [batchsize,height,width,channels]
    """
    xsh = x.get_shape().as_list()
    n_out = n_theta*n_omega
    theta = get_theta(xsh[3], n_theta, n_omega, even_initial=even_initial, name='Th'+name)
    omega = get_omega(xsh[3], n_theta, n_omega, even_initial=even_initial, name='W'+name)
    sigma = get_sigma(xsh[3], n_out, kernel_size=ksize, even_initial=even_initial, name='Sig'+name)
    phase = get_phase(xsh[3], n_out, even_initial=False, name='P'+name)
    W = get_2DGabor_kernels(theta, omega, sigma, phase, kernel_size=ksize)
    R = g_conv(x, W, strides=strides, padding=padding, name='gconv'+name)
    return R


def fullCon(x, n_out, stddev=0.2, name='fc'):
    """
    Fully connected lite

    :param x: input tf tensor, shape [batchsize,channels]
    :param n_out: number of channels of outputs (int)
    :param stddev: scale of variable initialization (default: 0.2)
    :param name: (default: fc)

    :return fc: outputs after of connected layer, shape [batchsize,channels]
    """
    xsh = x.get_shape().as_list()
    init_fw = tf.random_normal_initializer(stddev=stddev)
    fweight = tf.get_variable(name='fw'+name, shape=[xsh[1], n_out], initializer=init_fw)
    bias = tf.get_variable(name='fb'+name, shape=[n_out], initializer=tf.constant_initializer(1e-2))
    fc = tf.nn.bias_add(tf.matmul(x, fweight), bias)
    return fc


def fullCon_ReLU(x, n_out, stddev=0.2, alpha=0.2, name='fc_r'):
    """
    Fully connected lite with leaky ReLU layer

    :param x: input tf tensor, shape [batchsize,channels]
    :param n_out: number of channels of outputs (int)
    :param stddev: scale of variable initialization (default: 0.2)
    :param alpha: slope of active function when the value is less than 0 (default: 0.2)
    :param name: (default: fc_r)

    :return: outputs of fully connected layer and leaky ReLU layer, shape [batchsize,channels]
    """
    xsh = x.get_shape().as_list()
    init_fw = tf.random_normal_initializer(stddev=stddev)
    fweight = tf.get_variable(name='fw'+name, shape=[xsh[1], n_out], initializer=init_fw)
    bias = tf.get_variable(name='fb'+name, shape=[n_out], initializer=tf.constant_initializer(1e-2))
    fc = tf.nn.bias_add(tf.matmul(x, fweight), bias)
    return tf.nn.leaky_relu(fc,alpha=alpha)


def batch_norm(x, train_phase, decay=0.99, name='hbn'):
    """
    Batch normalization for the magnitudes of X

    :param x: input tf tensor, shape [batchsize,height,width,channels]
    :param train_phase: whether trainable (True or False)
    :param decay:
    :param name: (default:0.99)

    :return: outputs of batch normalization layer, shape [batchsize,,height,width,channels]
    """
    return g_batch_norm(x, train_phase, decay=decay, name=name)


def non_linearity(x, fnc=tf.nn.leaky_relu, alpha=0.2, name='nl'):
    """
    nonlinearity lite

    :param x: input tf tensor
    :param fnc: type of used nonlinearity function (default: leaky ReLU)
    :param alpha: for leaky ReLU (default: 0.2)
    :param name: (default: nl)

    :return: outputs of nonlinearity layer
    """
    xsh = x.get_shape()
    b = tf.get_variable('nlb' + name, shape=[xsh[-1]], initializer=tf.constant_initializer(0))
    Bx = tf.nn.bias_add(x, b)

    if 'leaky' in fnc.__name__:
        Rx = fnc(Bx,alpha=alpha,name=name)
    else:
        Rx = fnc(Bx,name=name)
    return Rx


def mean_pool(x, ksize=(1,1,1,1), strides=(1,1,1,1), name='mp'):
    """Mean pooling"""
    Y = tf.nn.avg_pool(x, ksize=ksize, strides=strides, padding='VALID',
                       name=name)
    return Y


def max_pool(x, ksize=(1,1,1,1), strides=(1,1,1,1), name='mp'):
    """Mean pooling"""
    Y = tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='VALID',
                       name=name)
    return Y

