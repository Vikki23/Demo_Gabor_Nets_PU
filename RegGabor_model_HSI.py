'''models for Gabor-Nets'''

import sys
sys.path.append('../')
import tensorflow as tf
import RegGabor_network_lite as rg_lite


def deep_model(args, x, train_phase):
   """
   deep Gabor-Nets model with two convolutional blocks (two convolutional layers in each block)
   and one fully connected block.
   """

   # settings
   n_th1 = args.n_theta
   n_omg1 = args.n_omega
   n_th2 = args.n_theta*args.gains
   n_omg2 = args.n_omega#*args.gains
   fs = args.filter_size
   ncl = args.n_classes
   sm = args.std_mult
   eveni = args.even_initial

   # Convolutional Layers with pooling
   with tf.name_scope('block1') as scope:
      cv1 = rg_lite.conv2d(x, n_th1, n_omg1, fs, padding='SAME', stddev=sm, even_initial=eveni, name='1')
      cv1_R = rg_lite.non_linearity(cv1, tf.nn.relu, alpha=0.2, name='1')

      cv2 = rg_lite.conv2d(cv1_R, n_th1, n_omg1, fs, padding='SAME', stddev=sm, even_initial=eveni, name='2')
      cv2_B = rg_lite.batch_norm(cv2, train_phase, name='2')

   with tf.name_scope('block2') as scope:
      cv3 = rg_lite.conv2d(cv2_B, n_th2, n_omg2, fs, padding='SAME', stddev=sm, even_initial=eveni, name='3')
      cv3_R = rg_lite.non_linearity(cv3,tf.nn.relu, alpha=0.2, name='3')

      cv4 = rg_lite.conv2d(cv3_R, n_th2, n_omg2, fs, padding='SAME', stddev=sm, even_initial=eveni, name='4')
      cv4_B = rg_lite.batch_norm(cv4, train_phase, name='4')

   # Final Layer
   with tf.name_scope('block3') as scope:
      rm = tf.reduce_mean(cv4_B, axis=[1, 2])
      rm_sh = rm.get_shape().as_list()
      fc1 = rg_lite.fullCon_ReLU(rm, rm_sh[1]*2, stddev=0.2, alpha=0.2, name='5')
      fc2 = rg_lite.fullCon(fc1, ncl, stddev=0.2, name='6')
   return fc2
   # end of deep_model


def deep_model_1block(args, x, train_phase):
   """
   deep Gabor-Nets model with one convolutional block and one fully connected block.
   """

   # settings
   n_th1 = args.n_theta
   n_omg1 = args.n_omega
   fs = args.filter_size
   ncl = args.n_classes
   sm = args.std_mult
   eveni = args.even_initial

   # Convolutional Layers with pooling
   with tf.name_scope('block1') as scope:
      cv1 = rg_lite.conv2d(x, n_th1, n_omg1, fs, padding='SAME', stddev=sm, even_initial=eveni, name='1')
      cv1_R = rg_lite.non_linearity(cv1, tf.nn.relu, alpha=0.2, name='1')

      cv2 = rg_lite.conv2d(cv1_R, n_th1, n_omg1, fs, padding='SAME', stddev=sm, even_initial=eveni, name='2')
      cv2_B = rg_lite.batch_norm(cv2, train_phase, name='2')

   # Final block
   with tf.name_scope('block3') as scope:
      rm = tf.reduce_mean(cv2_B, axis=[1, 2])
      rm_sh = rm.get_shape().as_list()
      fc1 = rg_lite.fullCon_ReLU(rm, rm_sh[1]*2, stddev=0.2, alpha=0.2, name='5')
      fc2 = rg_lite.fullCon(fc1, ncl, stddev=0.2, name='6')
   return fc2
   # end of deep_model_1block


def deep_model_3block(args, x, train_phase):
   """
   deep Gabor-Nets model with three convolutional blocks and one fully connected block.
   """

   # Settings
   n_th1 = args.n_theta
   n_omg1 = args.n_omega
   n_th2 = args.n_theta*args.gains
   n_omg2 = args.n_omega
   n_th3 = args.n_theta*args.gains #*(args.gains**2)
   n_omg3 = args.n_omega
   fs = args.filter_size
   ncl = args.n_classes
   sm = args.std_mult
   eveni = args.even_initial

   # Convolutional Layers with pooling
   with tf.name_scope('block1') as scope:
      cv1 = rg_lite.conv2d(x, n_th1, n_omg1, fs, padding='SAME', stddev=sm, even_initial=eveni, name='1')
      cv1_R = rg_lite.non_linearity(cv1, tf.nn.relu, alpha=0.2, name='1')

      cv2 = rg_lite.conv2d(cv1_R, n_th1, n_omg1, fs, padding='SAME', stddev=sm, even_initial=eveni, name='2')
      cv2_B = rg_lite.batch_norm(cv2, train_phase, name='2')

   with tf.name_scope('block2') as scope:
      cv3 = rg_lite.conv2d(cv2_B, n_th2, n_omg2, fs, padding='SAME', stddev=sm, even_initial=eveni, name='3')
      cv3_R = rg_lite.non_linearity(cv3,tf.nn.relu, alpha=0.2, name='3')

      cv4 = rg_lite.conv2d(cv3_R, n_th2, n_omg2, fs, padding='SAME', stddev=sm, even_initial=eveni, name='4')
      cv4_B = rg_lite.batch_norm(cv4, train_phase, name='4')

   with tf.name_scope('block3') as scope:
      cv5 = rg_lite.conv2d(cv4_B, n_th3, n_omg3, fs, padding='SAME', stddev=sm, even_initial=eveni, name='5')
      cv5_R = rg_lite.non_linearity(cv5,tf.nn.relu, alpha=0.2, name='5')

      cv6 = rg_lite.conv2d(cv5_R, n_th3, n_omg3, fs, padding='SAME', stddev=sm, even_initial=eveni, name='6')
      cv6_B = rg_lite.batch_norm(cv6, train_phase, name='6')

   # Final block
   with tf.name_scope('block4') as scope:
      rm = tf.reduce_mean(cv6_B, axis=[1, 2])
      rm_sh = rm.get_shape().as_list()
      fc1 = rg_lite.fullCon_ReLU(rm, rm_sh[1]*2, stddev=0.2, alpha=0.2, name='fc1')
      fc2 = rg_lite.fullCon(fc1, ncl, stddev=0.2, name='fc2')
   return fc2
   # end of deep_model_3block


def deep_pavia_4block(args, x, train_phase):
   """
   deep Gabor-Nets model with four convolutional blocks and one fully connected block.
   """

   # Settings
   n_th1 = args.n_theta
   n_omg1 = args.n_omega
   n_th2 = args.n_theta * args.gains
   n_omg2 = args.n_omega
   n_th3 = args.n_theta * args.gains #* (args.gains**2)
   n_omg3 = args.n_omega
   n_th4 = args.n_theta * args.gains #* (args.gains**3)
   n_omg4 = args.n_omega
   fs = args.filter_size
   ncl = args.n_classes
   sm = args.std_mult
   eveni = args.even_initial

   # Convolutional Layers with pooling
   with tf.name_scope('block1') as scope:
      cv1 = rg_lite.conv2d(x, n_th1, n_omg1, fs, padding='SAME', stddev=sm, even_initial=eveni, name='1')
      cv1_R = rg_lite.non_linearity(cv1, tf.nn.relu, alpha=0.2, name='1')

      cv2 = rg_lite.conv2d(cv1_R, n_th1, n_omg1, fs, padding='SAME', stddev=sm, even_initial=eveni, name='2')
      cv2_B = rg_lite.batch_norm(cv2, train_phase, name='2')

   with tf.name_scope('block2') as scope:
      cv3 = rg_lite.conv2d(cv2_B, n_th2, n_omg2, fs, padding='SAME', stddev=sm, even_initial=eveni, name='3')
      cv3_R = rg_lite.non_linearity(cv3,tf.nn.relu, alpha=0.2, name='3')

      cv4 = rg_lite.conv2d(cv3_R, n_th2, n_omg2, fs, padding='SAME', stddev=sm, even_initial=eveni, name='4')
      cv4_B = rg_lite.batch_norm(cv4, train_phase, name='4')

   with tf.name_scope('block3') as scope:
      cv5 = rg_lite.conv2d(cv4_B, n_th3, n_omg3, fs, padding='SAME', stddev=sm, even_initial=eveni, name='5')
      cv5_R = rg_lite.non_linearity(cv5,tf.nn.relu, alpha=0.2, name='5')

      cv6 = rg_lite.conv2d(cv5_R, n_th3, n_omg3, fs, padding='SAME', stddev=sm, even_initial=eveni, name='6')
      cv6_B = rg_lite.batch_norm(cv6, train_phase, name='6')

   with tf.name_scope('block4') as scope:
      cv7 = rg_lite.conv2d(cv6_B, n_th4, n_omg4, fs, padding='SAME', stddev=sm, even_initial=eveni, name='7')
      cv7_R = rg_lite.non_linearity(cv7,tf.nn.relu, alpha=0.2, name='7')

      cv8 = rg_lite.conv2d(cv7_R, n_th4, n_omg4, fs, padding='SAME', stddev=sm, even_initial=eveni, name='8')
      cv8_B = rg_lite.batch_norm(cv8, train_phase, name='8')

   # Final block
   with tf.name_scope('block5') as scope:
      rm = tf.reduce_mean(cv8_B, axis=[1, 2])
      rm_sh = rm.get_shape().as_list()
      fc1 = rg_lite.fullCon_ReLU(rm, rm_sh[1]*2, stddev=0.2, alpha=0.2, name='fc1')
      fc2 = rg_lite.fullCon(fc1, ncl, stddev=0.2, name='fc2')
   return fc2
   # end of deep_model_4block