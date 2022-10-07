"""
Run Gabor-Nets for the ROSIS Pavia University dataset

C. Liu, J. Li, L. He, A. Plaza, S. Li and B. Li, "Naive Gabor Networks for Hyperspectral Image Classification," in IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 1, pp. 376-390, Jan. 2021.

"""

import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf
import scipy.io as sio
sys.path.append('../')

from RegGabor_model_HSI import deep_model
from sample_saveID import Sample


def load_mat_data(args):
   """
   load the data of the Pavia University scene and arrange the training and test sets.

   :param args: settings

   :return: data
   """

   # the names of the loaded data
   imfile = 'PaviaU_im.mat'
   gtfile = 'PaviaU_gt.mat'

   # load data
   c = Sample(args.n_cols, args.n_rows, args.n_channels, args.dim, args.n_classes, 1, 1)
   c.load(imfile, 'im', gtfile, 'map', 1, 1)
   c.normalization()
   c.init_sam(args.n_perclass, args.n_validate, args.id_set, args.dataset_path, args.data_name, mk=args.mk)

   # arrange the training and test sets
   data = {}
   data['test_x'] = c.test['sample']
   data['test_y'] = c.test['label']
   if args.n_validate > 0:
      if args.combine_train_val:
         data['train_x'] = np.vstack((c.train['sample'], c.validate['sample']))
         data['train_y'] = np.hstack((c.train['label'], c.validate['label']))
      else:
         data['train_x'] = c.train['sample']
         data['train_y'] = c.train['label']
         data['valid_x'] = c.validate['sample']
         data['valid_y'] = c.validate['label']
   return data
   # end of load_mat_data


def settings(args):
   args.data_name = 'PU'
   args.dataset_path = 'DataSets/'
   args.n_cols = 340        # number of columns
   args.n_rows = 610        # number of rows
   args.n_channels = 103    # number of bands
   args.dim = 15            # patch size
   args.filter_size = 5     # filter size
   args.n_classes = 9       # number of predefined classes
   args.n_perclass = 100    # number of training samples per class
   args.n_validate = 50     # number of validation samples per class
   args.mk = 0              # whether make data augmentation
   args.id_set = 1          # the index of training sets
   args.bnum = 2            # number of convolutional blocks

   data = load_mat_data(args)
   data['test_x'] = np.vstack((data['test_x'], data['valid_x']))
   data['test_y'] = np.hstack((data['test_y'], data['valid_y']))
   ###########################################################################
   
   # Other options
   if args.default_settings:
      args.n_epochs = 100
      args.batch_size = 50
      args.learning_rate = 0.0076
      args.std_mult = 0.4
      args.delay = 12
      args.n_theta = 4
      args.n_omega = 4
      args.gains = 2
      args.lr_div = 10
      args.even_initial = True

   # options for naming
   if args.mk:
      args.name_aug = '_Aug'
   else:
      args.name_aug = '_NoAug'
   if args.even_initial:
      args.name_init = ''
   else:
      args.name_init = '_RandInit'

   check_file_name = args.data_name + args.name_aug + args.name_init + '_P' + str(args.dim) +\
                     '_t' + str(args.n_perclass) + '_epo' + str(args.n_epochs) + '_model_' + str(args.id_set)
   args.log_path = add_folder('./logs')
   args.checkpoint_path = add_folder('./checkpoints') + '/' + check_file_name + '.ckpt'
   args.checkpoint_path_valid = add_folder('./checkpoints') + '/' + check_file_name +'_valid.ckpt'
   args.checkpoint_path_loss = add_folder('./checkpoints') + '/' + check_file_name +'_loss.ckpt'
   return args, data
   # end of settings


def add_folder(folder_name):
   if not os.path.exists(folder_name):
      os.mkdir(folder_name)
      print('Created {:s}'.format(folder_name))
   return folder_name
   # end of add_folder


def minibatcher(inputs, targets, batchsize, shuffle=False):
   """
   Generate batches.

   :param inputs: features
   :param targets: labels
   :param batchsize: the size of each batch
   :param shuffle: whether shuffle the batch.
   """

   assert len(inputs) == len(targets)
   if shuffle:
      indices = np.arange(len(inputs))
      np.random.shuffle(indices)
   for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
      if shuffle:
         excerpt = indices[start_idx:start_idx + batchsize]
      else:
         excerpt = slice(start_idx, start_idx + batchsize)
      yield inputs[excerpt], targets[excerpt]
   # end of minibatcher


def get_learning_rate(args, current, best, counter, learning_rate):
   """If have not seen accuracy improvement in delay epochs, then divide 
   learning rate by 10
   """
   if current > best:
      best = current
      counter = 0
   elif counter > args.delay:
      learning_rate = learning_rate / args.lr_div
      counter = 0
   else:
      counter += 1
   return (best, counter, learning_rate)


def get_vars(sess, args, save_path):
   """
   Get the values of variables and save them.

   :param sess: session
   :param args: settings
   :param save_path: the path for saving
   """
   Th = {}
   ti = 0
   W = {}
   wi = 0
   Sig = {}
   si = 0
   P = {}
   i = 0
   for var in tf.trainable_variables():
      if var.name[0] == 'T':
         Th[ti] = sess.run(var.name)
         ti = ti + 1
      if var.name[0] == 'W':
         W[wi] = sess.run(var.name)
         wi = wi + 1
      if var.name[0] == 'S':
         Sig[si] = sess.run(var.name)
         si = si + 1
      if var.name[0] == 'P':
         P[i] = sess.run(var.name)
         i = i + 1
   sio.savemat(save_path + '_Th', {'Th%d' % i: Th[i] for i in range(args.bnum)})
   sio.savemat(save_path + '_W', {'W%d' % i: W[i] for i in range(args.bnum)})
   sio.savemat(save_path + '_Sig', {'Sig%d' % i: Sig[i] for i in range(args.bnum)})
   sio.savemat(save_path + '_P', {'P%d' % i: P[i] for i in range(args.bnum)})
   # end of get_vars


def main(args):
   """Main body."""
   tf.reset_default_graph()
   ##### SETUP AND LOAD DATA #####
   # args.combine_train_val = 1
   args, data = settings(args)
   
   ##### BUILD MODEL #####
   ## Placeholders
   x = tf.placeholder(tf.float32, [args.batch_size,args.dim,args.dim,args.n_channels], name='x')
   y = tf.placeholder(tf.int64, [args.batch_size], name='y')
   learning_rate = tf.placeholder(tf.float32, name='learning_rate')
   train_phase = tf.placeholder(tf.bool, name='train_phase')

   # Construct model and optimizer
   pred = deep_model(args, x, train_phase)
   loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))

   # Evaluation criteria
   correct_pred = tf.equal(tf.argmax(pred, 1), y)
   accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

   # Optimizer
   optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
   grads_and_vars = optim.compute_gradients(loss)
   train_op = optim.apply_gradients(grads_and_vars)
   
   ##### TRAIN ####
   # Configure tensorflow session
   init_global = tf.global_variables_initializer()
   init_local = tf.local_variables_initializer()
   config = tf.ConfigProto()
   config.gpu_options.allow_growth = True
   config.log_device_placement = False
   
   lr = args.learning_rate
   saver = tf.train.Saver()
   sess = tf.Session(config=config)
   sess.run([init_global, init_local], feed_dict={train_phase : True})

   # save initializations
   init_path = add_folder('./Initialization') + '/Init_' + args.data_name + '_Reg2DGabor' + args.name_aug + \
               args.name_init + '_ks_' + str(args.filter_size) + '_Patch_' + str(args.dim) + \
               '_t' + str(args.n_perclass) + '_v' + str(args.n_validate) + '_perC' + '_bs' + \
               str(args.batch_size) + '_epo' + str(args.n_epochs) + '_' + str(args.id_set)
   get_vars(sess, args, init_path)

   # preparation before training
   start = time.time()
   epoch = 0
   tsh = [args.n_epochs, np.int(np.floor(data['train_y'].size / args.batch_size))]
   vsh = [args.n_epochs, np.int(np.floor(data['valid_y'].size / args.batch_size))]
   train_loss_all = np.zeros(tsh)
   train_acc_all = np.zeros(tsh)
   valid_acc_all = np.zeros(vsh)
   train_loss = np.zeros(args.n_epochs)
   train_acc = np.zeros(args.n_epochs)
   valid_acc = np.zeros(args.n_epochs)
   train_time = np.zeros(args.n_epochs)

   # start training loop
   print('Starting training loop...')
   valid_acc_best = 0.60
   train_loss_best = 0.50
   while epoch < args.n_epochs:
      # Training steps
      batcher = minibatcher(data['train_x'], data['train_y'], args.batch_size, shuffle=True)
      train_loss_ = 0.
      train_acc_ = 0.
      for i, (X, Y) in enumerate(batcher):
         feed_dict = {x: X, y: Y, learning_rate: lr, train_phase: True}
         __, loss_, accuracy_ = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
         train_loss_ += loss_
         train_acc_ += accuracy_
         train_loss_all[epoch][i] = loss_
         train_acc_all[epoch][i] = accuracy_
      train_acc_ /= (i + 1.)
      train_loss_ /= (i + 1.)
      train_loss[epoch] = train_loss_
      train_acc[epoch] = train_acc_
      
      if not args.combine_train_val:
         batcher = minibatcher(data['valid_x'], data['valid_y'], args.batch_size)
         valid_acc_ = 0.
         for i, (X, Y) in enumerate(batcher):
            feed_dict = {x: X, y: Y, train_phase: False}
            accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
            valid_acc_ += accuracy_
            valid_acc_all[epoch][i] = accuracy_
         valid_acc_ /= (i+1.)
         valid_acc[epoch] = valid_acc_
         train_time[epoch] = time.time()-start
         print('[{:04d} | {:0.1f}] Loss: {:04f}, Train Acc.: {:04f}, Validation Acc.: {:04f}, Learning rate: {:.2e}'.format(epoch,
            time.time()-start, train_loss_, train_acc_, valid_acc_, lr))
         # Save model
         if valid_acc_ > valid_acc_best:
            valid_acc_best = valid_acc_
            saver.save(sess, args.checkpoint_path_valid)
            print('Valid-best model saved')
         if train_loss_ < train_loss_best:
            train_loss_best = train_loss_
            saver.save(sess, args.checkpoint_path_loss)
            print('Loss-best model saved')
      else:
         train_time[epoch] = time.time() - start
         print('[{:04d} | {:0.1f}] Loss: {:04f}, Train Acc.: {:04f}, Learning rate: {:.2e}'.format(epoch,
            time.time()-start, train_loss_, train_acc_, lr))

      
      # Updates to the training scheme
      lr = args.learning_rate * np.power(0.10, epoch / 50)
      epoch += 1

   saver.save(sess, args.checkpoint_path)
   print('Model saved')

   # save learned parameters
   retrive_file = 'Results/Retrieve_' + args.data_name + '_Reg2DGabor' + args.name_aug + args.name_init + \
                  '_SaveBest_lr_auto_ks_' + str(args.filter_size) + '_Patch_' + str(args.dim) + \
                  '_t' + str(args.n_perclass) + '_v' + str(args.n_validate) + '_perC' + '_bs' + \
                  str(args.batch_size) + '_epo' + str(args.n_epochs) + '_' + str(args.id_set)
   get_vars(sess, args, retrive_file)

   #########################################################
   # TEST
   ntotal = data['test_y'].size
   testlg = [1,np.int(np.ceil(ntotal / args.batch_size))]
   test_acc_all = np.zeros(testlg)
   batcher = minibatcher(data['test_x'], data['test_y'], args.batch_size)
   test_acc = 0
   for i, (X, Y) in enumerate(batcher):
      feed_dict = {x: X, y: Y, train_phase: False}
      accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
      test_acc += accuracy_ * args.batch_size
      test_acc_all[0][i] = accuracy_
      sys.stdout.write('Testing\r')
      sys.stdout.flush()

   ntest = (i + 1) * args.batch_size
   if ntest < ntotal:
      nremain = ntotal - ntest
      feed_dict = {x: data['test_x'][-args.batch_size:, :, :, :], train_phase: False}
      pred_labels_ = sess.run(tf.argmax(pred, 1), feed_dict=feed_dict)
      correct_pred_ = np.equal(pred_labels_[-nremain:], data['test_y'][-nremain:]).astype('float32')
      accuracy_ = np.mean(correct_pred_)
      test_acc += accuracy_ * nremain
      test_acc_all[0][i+1] = accuracy_
      sys.stdout.write('Testing\r')
      sys.stdout.flush()

   test_acc /= ntotal
   print('Test Acc.: {:04f}'.format(test_acc))

   #########################################################
   # TEST for valid-best
   saver.restore(sess, args.checkpoint_path_valid)
   test_acc_valid_all = np.zeros(testlg)
   batcher = minibatcher(data['test_x'], data['test_y'], args.batch_size)
   test_acc_valid = 0.
   for i, (X, Y) in enumerate(batcher):
      feed_dict = {x: X, y: Y, train_phase: False}
      accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
      test_acc_valid += accuracy_ * args.batch_size
      test_acc_valid_all[0][i] = accuracy_
      sys.stdout.write('Testing\r')
      sys.stdout.flush()

   ntest = (i + 1) * args.batch_size
   if ntest < ntotal:
      nremain = ntotal - ntest
      feed_dict = {x: data['test_x'][-args.batch_size:, :, :, :], train_phase: False}
      pred_labels_ = sess.run(tf.argmax(pred, 1), feed_dict=feed_dict)
      correct_pred_ = np.equal(pred_labels_[-nremain:], data['test_y'][-nremain:]).astype('float32')
      accuracy_ = np.mean(correct_pred_)
      test_acc_valid += accuracy_ * nremain
      test_acc_valid_all[0][i + 1] = accuracy_
      sys.stdout.write('Testing\r')
      sys.stdout.flush()

   test_acc_valid /= ntotal
   print('Valid-best Test Acc.: {:04f}'.format(test_acc_valid))

   #########################################################
   # TEST for loss-best
   saver.restore(sess, args.checkpoint_path_loss)
   test_acc_loss_all = np.zeros(testlg)
   batcher = minibatcher(data['test_x'], data['test_y'], args.batch_size)
   test_acc_loss = 0.
   for i, (X, Y) in enumerate(batcher):
      feed_dict = {x: X, y: Y, train_phase: False}
      accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
      test_acc_loss += accuracy_ * args.batch_size
      test_acc_loss_all[0][i] = accuracy_
      sys.stdout.write('Testing\r')
      sys.stdout.flush()

   ntest = (i + 1) * args.batch_size
   if ntest < ntotal:
      nremain = ntotal - ntest
      feed_dict = {x: data['test_x'][-args.batch_size:, :, :, :], train_phase: False}
      pred_labels_ = sess.run(tf.argmax(pred, 1), feed_dict=feed_dict)
      correct_pred_ = np.equal(pred_labels_[-nremain:], data['test_y'][-nremain:]).astype('float32')
      accuracy_ = np.mean(correct_pred_)
      test_acc_loss += accuracy_ * nremain
      test_acc_loss_all[0][i + 1] = accuracy_
      sys.stdout.write('Testing\r')
      sys.stdout.flush()

   test_acc_loss /= ntotal
   print('Loss-best Test Acc.: {:04f}'.format(test_acc_loss))

   results_file = 'Results/'+ args.data_name + '_Reg2DGabor' + args.name_aug + args.name_init +'_SaveBest_lr_auto_ks_'+str(args.filter_size)+'_Patch_'+str(args.dim)+\
                  '_t'+str(args.n_perclass)+'_v'+str(args.n_validate)+'_perC'+'_bs'+str(args.batch_size)+\
                  '_epo'+str(args.n_epochs)+'_'+str(args.id_set)+'.mat'
   sio.savemat(results_file,
               {'train_acc_all': train_acc_all, 'train_loss_all': train_loss_all,'valid_acc_all': valid_acc_all,
                'test_acc_all':test_acc_all,'test_acc_valid_all': test_acc_valid_all, 'test_acc_loss_all': test_acc_loss_all,
                'train_acc': train_acc, 'train_loss': train_loss,'valid_acc': valid_acc,
                'test_acc': test_acc,'test_acc_valid': test_acc_valid, 'test_acc_loss': test_acc_loss,
                'train_time': train_time}  )
   sess.close()
      

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--data_dir", help="data directory", default='./data')
   parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
   parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool, default=False)
   main(parser.parse_args())
