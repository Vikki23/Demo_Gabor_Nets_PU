"""
Detailed Gabor-Nets Implementation
"""

import numpy as np
import tensorflow as tf
import scipy.io as sio


def g_conv(X, W, strides=(1, 1, 1, 1), padding='VALID', name='g_conv'):
    """
    Implement 2D Gabor convolutions

    :param X: input tensor, shape [mbatch,h,w,channels]
    :param W: kernel weights of Gabor kernels
    :param strides: stride size (4-tuple: default (1,1,1,1))
    :param padding: SAME or VALID (default: VALID)
    :param name: (default: g_conv)

    :return: convolution outputs
    """
    with tf.name_scope('gconv' + str(name)) as scope:
        # Convolve
        Y = tf.nn.conv2d(X, W, strides=strides, padding=padding, name=name)
        return Y


def g_batch_norm(X, train_phase, decay=0.99, name='hbn'):
    """
    Batch normalization for the magnitudes of X

    :param X: input tensor
    :param train_phase: whether trainable (True or False)
    :param decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
    :param name: (default: hbn)

    :return: outputs of batch normalization
    """
    with tf.name_scope(name) as scope:
        Rb = bn(X, train_phase, decay=decay, name=name)
        return Rb


def bn(X, train_phase, decay=0.99, name='batchNorm'):
    """Batch normalization module.

    :param X: tf tensor
    :param train_phase: boolean flag True: training mode, False: test mode
    :param decay: decay rate: 0 is memory-less, 1 no updates (default 0.99)
    :param name: (default batchNorm)

    Source: bgshi @ http://stackoverflow.com/questions/33949786/how-could-i-use-
    batch-normalization-in-tensorflow"""
    Xsh = X.get_shape().as_list()
    n_out = Xsh[-1]

    with tf.name_scope(name) as scope:
        beta = tf.get_variable(name + '_beta', dtype=tf.float32, shape=n_out,
                               initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable(name + '_gamma', dtype=tf.float32, shape=n_out,
                                initializer=tf.constant_initializer(1.0))
        pop_mean = tf.get_variable(name + '_pop_mean', dtype=tf.float32,
                                   shape=n_out, trainable=False)
        pop_var = tf.get_variable(name + '_pop_var', dtype=tf.float32,
                                  shape=n_out, trainable=False)
        batch_mean, batch_var = tf.nn.moments(X, list(range(len(Xsh) - 1)), name=name + 'moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        pop_mean_op = tf.assign(pop_mean, ema.average(batch_mean))
        pop_var_op = tf.assign(pop_var, ema.average(batch_var))

        with tf.control_dependencies([ema_apply_op, pop_mean_op, pop_var_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(train_phase, mean_var_with_update,
                        lambda: (pop_mean, pop_var))
    normed = tf.nn.batch_normalization(X, mean, var, beta, gamma, 1e-3)
    return normed


def get_2DGabor_kernels(theta, omega, sigma, phase, kernel_size, Gabor_type='Re'):
    """
    Create 2D Gabor kernels

    :param theta:
    :param omega:
    :param sigma:
    :param phase:
    :param kernel_size:
    :param Gabor_type:

    :return:
    """
    sh = theta.get_shape().as_list()
    k = kernel_size
    type = Gabor_type
    sig = sigma[np.newaxis, :, :] # [1,in,out]
    P = phase[np.newaxis,:,:] # [1,in,out]

    Freq_X = omega * tf.cos(theta)  # [in,out]
    Freq_Y = omega * tf.sin(theta)  # [in,out]
    Freq_X = Freq_X[np.newaxis, :, :]  # [1,in,out]
    Freq_Y = Freq_Y[np.newaxis, :, :]  # [1,in,out]

    # Create the Euclidean coordinates of the kernel
    # the first row is y-axis, the second row is x-axis
    coords = L2_grid(k)
    coords_X = coords[1, :]
    coords_Y = coords[0, :]
    coords_X = coords_X[:, np.newaxis, np.newaxis]  # (25,1,1)
    coords_Y = coords_Y[:, np.newaxis, np.newaxis]  # (25,1,1)

    # envelop construction k
    Envelop1 = (coords_X ** 2 + coords_Y ** 2) / (sig ** 2)
    Envelop = (tf.exp(-0.5 * Envelop1)) / (2 * np.pi * (sig ** 2))  # (25,in,out)

    Freq = Freq_X * coords_X + Freq_Y * coords_Y  # (25,in,out)
    # construct corresponding Gabor kernels
    if type == 'Re':
        kernels = Envelop * tf.cos(Freq+P)
    elif type == 'Im':
        kernels = Envelop * tf.sin(Freq+P)
    else:
        print('Error Gabor filter type')
        exit(0)
    kernels = tf.reshape(kernels, np.hstack([k, k, sh]))

    return kernels


def L2_grid(shape):
    """
    get the Euclidean coordinates of the kernel

    :param shape: size of the target kernel

    :return: coordinates of the target kernel
    """
    # Get neighbourhoods
    center = shape // 2
    lin = np.arange(shape)
    J, I = np.meshgrid(lin, lin)  # J-X I-Y
    I = I - center # Y
    J = J - center # X
    return np.vstack((np.reshape(I, -1), np.reshape(J, -1)))


def get_theta(n_in, n_theta, n_omega, even_initial=False, name='th'):
    """
    Get the list of theta

    :param n_in: number of the channels of inputs (int)
    :param n_theta: number of the initialization values of theta (int)
    :param n_omega: number of the initialization values of theta (int)
    :param even_initial: whether adopt the even initialization strategy (boolean)
    :param name: (default: th)

    :return: a list of theta
    """
    if even_initial:
        init = np.linspace(0, 1, num=n_theta, endpoint=False) * (np.pi)
        init = np.tile(init,(1, n_omega))
        init = np.tile(init,(n_in, 1))
    else:
        n_out = n_theta*n_omega
        init = np.random.rand(n_in, n_out) *(2*np.pi)

    init = np.float32(init)
    n_out = n_omega*n_theta
    theta = tf.get_variable(name, dtype=tf.float32, shape=[n_in, n_out],
                        initializer=tf.constant_initializer(init))
    return theta


def get_omega(n_in, n_theta, n_omega, even_initial=False, mean=0, name='omega'):
    """
    Get the list of omega

    :param n_in: number of the channels of inputs (int)
    :param n_theta: number of the initialization values of theta (int)
    :param n_omega: number of the initialization values of theta (int)
    :param even_initial: whether adopt the even initialization strategy (boolean)
    :param mean: the mean of normal distribution in the random initialization strategy
    :param name: (default: omega)

    :return: a list of omega
    """
    if even_initial:
        init = np.logspace(0,n_omega-1,num=n_omega,base=1/2)*(np.pi/2)
        init = init[:,np.newaxis]
        init = np.tile(init,[1,n_theta])
        init = np.reshape(init,[1,-1])
        init = np.tile(init,(n_in, 1))
        n_out = n_omega * n_theta
        omega = tf.get_variable(name, dtype=tf.float32, shape=[n_in, n_out],
                                initializer=tf.constant_initializer(init))
    else:
        n_out = n_omega * n_theta
        stddev = np.pi/8
        init = tf.random_normal_initializer(mean=mean,stddev=stddev)
        omega = tf.get_variable(name, dtype=tf.float32, shape=[n_in, n_out],
                                initializer=init)
    return omega


def get_sigma(n_in, n_out, kernel_size=5, even_initial=False, mean=0, name='sigma'):
    """
    Get the list of sigma

    :param n_in: number of the channels of inputs (int)
    :param n_out: number of the channels of outputs (int)
    :param kernel_size: size of kernels
    :param even_initial: whether adopt the even initialization strategy
    :param mean: the mean of normal distribution in the random initialization strategy
    :param name: (default: sigma)

    :return: a list of sigma
    """
    if even_initial:
        init = np.tile(kernel_size/8,(n_in, n_out))
        sigma = tf.get_variable(name, dtype=tf.float32, shape=[n_in, n_out],
                                initializer=tf.constant_initializer(init))
    else:
        stddev = (5/4) * (1/2)
        init = tf.random_normal_initializer(mean=mean, stddev=stddev)
        sigma = tf.get_variable(name, dtype=tf.float32, shape=[n_in, n_out],
                                initializer=init)
    return sigma


def get_phase(n_in, n_out, even_initial=False, name='P'):
    """
    Get the list of phase offsets

    :param n_in: number of the channels of inputs (int)
    :param n_out: number of the channels of outputs (int)
    :param even_initial: whether adopt the even initialization strategy
    :param name: (default: P)

    :return: a list of phase offsets
    """
    if even_initial:
        # initialization corresponding to each output
        init = np.zeros([n_in,n_out])
    else:
        init = np.random.rand(n_in,n_out) * (2*np.pi)
        init = np.float32(init)

    phase = tf.get_variable(name, dtype=tf.float32, shape=[n_in,n_out],
        initializer=tf.constant_initializer(init))

    return phase