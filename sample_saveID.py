"""
class Sample
Created for manipulating data

"""

import os
import random
import h5py
import scipy
import numpy as np

class Sample:
    
    def __init__(self, IMGX, IMGY, IMGB, SAMN, CLAN, SEPN, RSEED):
        self.img = {'im': [],
               'gt': [],
               'add': [],
              }
        self.all = {'sample': [],
                    'XY': []
                    }
        self.train = {'sample': [],
                 'label': [],
                 'XY': []
                }
        self.test = {'sample': [],
                 'label': [],
                 'XY': []
                }
        self.validate = {'sample': [],
                 'label': [],
                 'XY': []
                }
        self.IMGX = IMGX
        self.IMGY = IMGY
        self.IMGB = IMGB
        self.SAMD = SAMN // 2   # padding size: half of the patch size
        self.SAMN = SAMN        # patch size
        self.CLAN = CLAN        # the number of predefined classes
        self.SEPN = SEPN        # the total number of batches when loading all the samples
        if RSEED != 1:          # seed for shuffle
            random.seed(RSEED)
    
    def __del__(self):
        self.img.clear()
        self.all.clear()
        self.train.clear()
        self.test.clear()
        self.validate.clear()
        del self.img
        del self.all
        del self.train
        del self.test
        del self.validate
        try:
            self.self_sam.clear()
            self.flush.clear()
            del self.self_sam
            del self.flush
        except:
            pass
        # print('del samples!')
        

    def load(self, img_file, img_id, gt_file, gt_id, resh, bg_bias):
        """
        Load the feature and ground-truth map;
        prepare the features as patches.

        :param img_file: the name of feature file
        :param img_id: the name of the target variable in img_file
        :param gt_file: the name of the ground-truth file
        :param gt_id: the name of the target variable in gt_file
        :param resh: whether reshape the data
        :param bg_bias: background bias (the index of the first class annotated in ground-truth)

        :return: self
        """

        self.img['gt'] = scipy.io.loadmat(gt_file)[gt_id]
        self.img['im'] = scipy.io.loadmat(img_file)[img_id]
        self.img['gt'] = self.img['gt'] - bg_bias
        self.img['gt'] = self.img['gt'].astype('float32')
        self.img['im'] = self.img['im'].astype('float32')
        if resh == 1:
            self.img['gt'] = self.img['gt'].T
            self.img['im'] = np.reshape(self.img['im'].T,
                                        [self.IMGX, self.IMGY, self.IMGB])
        # extend the image at edges
        img_im = self.img['im']
        n = self.SAMD
        r1 = np.repeat([img_im[0,:,:]], n, axis=0)
        r2 = np.repeat([img_im[-1,:,:]], n, axis=0)
        img_add = np.concatenate((r1, img_im, r2))
        r1 = np.reshape(img_add[:,0,:],[self.IMGX + 2 * n, 1, self.IMGB])
        r2 = np.reshape(img_add[:,-1,:],[self.IMGX + 2 * n, 1, self.IMGB])
        r1 = np.repeat(r1, self.SAMD, axis=1)
        r2 = np.repeat(r2, self.SAMD, axis=1)
        self.img['add'] = np.concatenate((r1, img_add, r2), axis=1)
        self.img['add'] = self.img['add'].astype('float32')
        # end of load


    def normalization(self):
        """
        Normalization: make data values within [-1,1]
        """

        img_add = self.img['add'].astype('float32')
        for i in range(self.IMGB):
            img_add[:,:,i] = img_add[:,:,i] - img_add[:,:,i].min()
            img_add[:,:,i] = img_add[:,:,i] / img_add[:,:,i].max()
            img_add[:,:,i] = img_add[:,:,i] * 2 - 1
        self.img['add'] = img_add
        # end of normalization


    def make_sample(self, origin):
        """
        data augmentation (flip and mirror)

        :param origin: input data for augmentation
        """

        sample = origin['sample']  # features
        label = origin['label']    # labels
        XY = origin['XY']          # locations
        a = np.flip(sample,1)
        b = np.flip(sample,2)
        c = np.flip(b,1)
        new = {}
        new['sample'] = np.concatenate((a,b,c,sample),axis=0)
        new['label'] = np.concatenate((label,label,label,label),axis=0)
        new['XY'] = np.concatenate((XY,XY,XY,XY),axis=0)
        return new
        # end of make_sample


    def init_sam(self, n_perclass, n_validate, train_set_id, data_path, data_name, mk=0):
        """
        Generate training set according to the desired number of training samples per class
        If the required training set has existed, load it;
        otherwise, randomly select training samples as required, and save the training set.
        and save the training set.

        :param n_perclass: number of training samples per class
        :param n_validate: number of validation samples per class
        :param train_set_id: index of the training set
        :param data_path: path of data
        :param data_name: name of data
        :param mk: whether implement data augmentation
        """

        # generate the data path and fetch data
        if data_path[-1] is not '/':
            data_path = data_path + '/'
        if n_validate > 0:
            id_file = data_path + data_name + '_Train_Valid_Test_ID_Num_' + str(n_perclass) + '_' + str(train_set_id) + '.mat'
        else:
            id_file = data_path + data_name + '_Train_Test_ID_Num_' + str(n_perclass) + '_' + str(train_set_id) + '.mat'
        fetch = os.path.exists(id_file)

        # fetch the locations of samples from the existing training set;
        # otherwise, randomly select training samples as required.
        if not fetch: # the required training set does not exist
            for i in range(self.CLAN):
                # fetch all the samples belonging to class i
                c_label = list(np.array(np.where(self.img['gt'] == i)).T)
                random.shuffle(c_label)
                self.train['XY'].extend(c_label[:n_perclass])
                if n_validate > 0:
                    self.validate['XY'].extend(c_label[n_perclass:(n_perclass+n_validate)])
                    self.test['XY'].extend(c_label[(n_perclass + n_validate ):])
                else:
                    self.test['XY'].extend(c_label[n_perclass:])
            # save the training set (save ids)
            if n_validate > 0:
                train_id = np.array(self.train['XY'])
                validate_id = np.array(self.validate['XY'])
                test_id = np.array(self.test['XY'])
                scipy.io.savemat(id_file, {'train_id': train_id, 'test_id': test_id, 'validate_id': validate_id})
            else:
                train_id = np.array(self.train['XY'])
                test_id = np.array(self.test['XY'])
                scipy.io.savemat(id_file, {'train_id': train_id, 'test_id': test_id})
        else: # the required training set exists
            self.train['XY'] = scipy.io.loadmat(id_file)['train_id']
            self.train['XY'] = self.train['XY'].tolist()
            self.test['XY'] = scipy.io.loadmat(id_file)['test_id']
            self.test['XY'] = self.test['XY'].tolist()
            if n_validate > 0:
                self.validate['XY'] = scipy.io.loadmat(id_file)['validate_id']
                self.validate['XY'] = self.validate['XY'].tolist()

        # shuffle the training set
        # fetch the corresponding features for the training, test, and validation sets
        random.shuffle(self.train['XY'])
        for i in self.train['XY']:
            self.train['sample'].append(self.get_sample(i))
            self.train['label'].append(self.get_label(i))
        for i in self.test['XY']:
            self.test['sample'].append(self.get_sample(i))
            self.test['label'].append(self.get_label(i))
        if self.validate['XY']:
            random.shuffle(self.validate['XY'])
            for i in self.validate['XY']:
                self.validate['sample'].append(self.get_sample(i))
                self.validate['label'].append(self.get_label(i))

        # convert each property to the form of array
        for i in self.train:
            self.train[i] = np.array(self.train[i])
            self.test[i] = np.array(self.test[i])
            if self.validate:
                self.validate[i] = np.array(self.validate[i])

        # flip and mirror the samples
        if mk:
            self.train = self.make_sample(self.train)
        # end of init_sam


    def init_sam_fixNum(self, n_perclass, n_validate, train_set_id, data_path, data_name, fix_name, mk=0):
        """
        Generate training set according to the fixed number set ahead for each class
        If the required training set has existed, load it;
        otherwise, randomly select training samples as required, and save the training set.

        :param n_perclass: [array] the fixed numbers of training samples for each class
        :param n_validate: [array] the fixed numbers of validation samples for each class
        :param train_set_id: index of the training set
        :param data_path: path of data
        :param data_name: name of data
        :param fix_name: annotation for the training set
        :param mk: whether implement data augmentation
        """

        # generate the data path and fetch data
        if data_path[-1] is not '/':
            data_path = data_path + '/'
        id_file = data_path + data_name + '_Train_Valid_Test_ID_fixNum_' + fix_name + '_' + str(train_set_id) + '.mat'
        fetch = os.path.exists(id_file)

        # fetch the locations of samples from the existing training set;
        # otherwise, randomly select training samples as required.
        if not fetch:  # the required training set does not exist
            for i in range(self.CLAN):
                # fetch all the samples belonging to class i
                c_label = list(np.array(np.where(self.img['gt'] == i)).T)
                random.shuffle(c_label)
                self.train['XY'].extend(c_label[:n_perclass[i]])
                if not n_validate is None:
                    if n_validate[i] > 0:
                        self.validate['XY'].extend(c_label[n_perclass[i]:(n_perclass[i]+n_validate[i])])
                        self.test['XY'].extend(c_label[(n_perclass[i] + n_validate[i]):])
                    else:
                        self.test['XY'].extend(c_label[n_perclass[i]:])
            # save the training set (save ids)
            if not self.validate['XY'] is None:
                train_id = np.array(self.train['XY'])
                validate_id = np.array(self.validate['XY'])
                test_id = np.array(self.test['XY'])
                scipy.io.savemat(id_file, {'train_id': train_id, 'test_id': test_id, 'validate_id': validate_id})
            else:
                train_id = np.array(self.train['XY'])
                test_id = np.array(self.test['XY'])
                scipy.io.savemat(id_file, {'train_id': train_id, 'test_id': test_id})
        else: # the required training set exists
            self.train['XY'] = scipy.io.loadmat(id_file)['train_id']
            self.train['XY'] = self.train['XY'].tolist()
            self.test['XY'] = scipy.io.loadmat(id_file)['test_id']
            self.test['XY'] = self.test['XY'].tolist()
            if not n_validate is None:
                self.validate['XY'] = scipy.io.loadmat(id_file)['validate_id']
                self.validate['XY'] = self.validate['XY'].tolist()

        # shuffle the training set
        # fetch the corresponding features for the training, test, and validation sets
        random.shuffle(self.train['XY'])
        for i in self.train['XY']:
            self.train['sample'].append(self.get_sample(i))
            self.train['label'].append(self.get_label(i))
        for i in self.test['XY']:
            self.test['sample'].append(self.get_sample(i))
            self.test['label'].append(self.get_label(i))
        if self.validate['XY']:
            random.shuffle(self.validate['XY'])
            for i in self.validate['XY']:
                self.validate['sample'].append(self.get_sample(i))
                self.validate['label'].append(self.get_label(i))

        # convert each property to the form of array
        for i in self.train:
            self.train[i] = np.array(self.train[i])
            self.test[i] = np.array(self.test[i])
            if self.validate:
                self.validate[i] = np.array(self.validate[i])

        # data augmentation
        if mk:
            self.train = self.make_sample(self.train)
        # end of init_sam_fixNum


    def init_sam_percent(self, t_perc, v_perc, train_set_id, data_path, data_name, mk=0):
        """
        Generate training set according to the desired percent of training samples per class
        If the required training set has existed, load it;
        otherwise, randomly select training samples as required, and save the training set.

        :param t_perc: the percent of training samples in each class
        :param v_perc: the percent of validation samples in each class
        :param train_set_id: index of the training set
        :param data_path: path of data
        :param data_name: name of data
        :param mk: whether implement data augmentation
        """

        # generate the data path and fetch data
        if data_path[-1] is not '/':
            data_path = data_path + '/'
        if v_perc > 0:
            id_file = data_path + data_name + '_Train_Valid_Test_ID_percent_' + str(int(t_perc*100)) + \
                      '_' + str(train_set_id) + '.mat'
        else:
            id_file = data_path + data_name + '_Train_Test_ID_Num_' + str(int(t_perc*100)) +\
                      '_' + str(train_set_id) + '.mat'
        fetch = os.path.exists(id_file)

        # fetch the locations of samples from the existing training set;
        # otherwise, randomly select training samples as required.
        if not fetch:  # the required training set does not exist
            for i in range(self.CLAN):
                # fetch all the samples belonging to class i
                c_label = list(np.array(np.where(self.img['gt'] == i)).T)
                n_perclass = np.ceil(len(c_label)*t_perc).astype('int32')
                random.shuffle(c_label)
                self.train['XY'].extend(c_label[:n_perclass])
                if v_perc > 0:
                    n_validate = np.ceil(len(c_label)*v_perc).astype('int32')
                    self.validate['XY'].extend(c_label[n_perclass:(n_perclass+n_validate)])
                    self.test['XY'].extend(c_label[(n_perclass + n_validate):])
                else:
                    self.test['XY'].extend(c_label[n_perclass:])
            # save the training set (save ids)
            if v_perc > 0:
                train_id = np.array(self.train['XY'])
                validate_id = np.array(self.validate['XY'])
                test_id = np.array(self.test['XY'])
                scipy.io.savemat(id_file, {'train_id': train_id, 'test_id': test_id, 'validate_id': validate_id})
            else:
                train_id = np.array(self.train['XY'])
                test_id = np.array(self.test['XY'])
                scipy.io.savemat(id_file, {'train_id': train_id, 'test_id': test_id})
        else:
            self.train['XY'] = scipy.io.loadmat(id_file)['train_id']
            self.train['XY'] = self.train['XY'].tolist()
            self.test['XY'] = scipy.io.loadmat(id_file)['test_id']
            self.test['XY'] = self.test['XY'].tolist()
            if v_perc > 0:
                self.validate['XY'] = scipy.io.loadmat(id_file)['validate_id']
                self.validate['XY'] = self.validate['XY'].tolist()

        # shuffle the training set
        # fetch the corresponding features for the training, test, and validation sets
        random.shuffle(self.train['XY'])
        for i in self.train['XY']:
            self.train['sample'].append(self.get_sample(i))
            self.train['label'].append(self.get_label(i))
        for i in self.test['XY']:
            self.test['sample'].append(self.get_sample(i))
            self.test['label'].append(self.get_label(i))
        if self.validate['XY']:
            random.shuffle(self.validate['XY'])
            for i in self.validate['XY']:
                self.validate['sample'].append(self.get_sample(i))
                self.validate['label'].append(self.get_label(i))

        # convert each property to the form of array
        for i in self.train:
            self.train[i] = np.array(self.train[i])
            self.test[i] = np.array(self.test[i])
            if self.validate:
                self.validate[i] = np.array(self.validate[i])

        # data augmentation
        if mk:
            self.train = self.make_sample(self.train)
        # end of init_sam_percent


    def load_predefine_train(self, gt_tfile, gt_tid, mk=0,resh=1,bg_bias=1):
        """
        Load predefined training set.

        :param gt_tfile: groundtruth of the predefined training set
        :param gt_tid: the name of the target variable in gt_tfile
        :param mk: whether make the data augmentation
        :param resh: whether reshape the data
        :param bg_bias: background bias (the index of the first class annotated in ground-truth)
        """

        self.img['gt2'] = scipy.io.loadmat(gt_tfile)[gt_tid]
        self.img['gt2'] = self.img['gt2'] - bg_bias
        if resh == 1:
            self.img['gt2'] = self.img['gt2'].T
        for i in range(self.CLAN):
            c_label = list(np.array(np.where(self.img['gt2'] == i)).T)
            random.shuffle(c_label)
            self.train['XY'].extend(c_label[:])
        random.shuffle(self.train['XY'])
        for i in self.train['XY']:
            self.train['sample'].append(self.get_sample(i))
            self.train['label'].append(self.get_label(i))
        for i in self.train:
            self.train[i] = np.array(self.train[i])

        # data augmentation
        if mk:
            self.train = self.make_sample(self.train)
        # end of load_predefine_train


    def load_predefine_test(self, gt2_file, gt2_id, resh=1, bg_bias=1, n_validation = 0, shuffle_opt = True):
        """
        Load predefined test set.

        :param gt2_file:  ground-truth of the predefined test set
        :param gt2_id: the name of the target variable in gt2_file
        :param resh: whether reshape the data
        :param bg_bias: background bias (the index of the first class annotated in ground-truth)
        :param n_validation: number of validation samples per class
        :param shuffle_opt: option for shuffle
        """

        self.img['gt3'] = scipy.io.loadmat(gt2_file)[gt2_id]
        self.img['gt3'] = self.img['gt3'] - bg_bias
        if resh == 1:
            self.img['gt3'] = self.img['gt3'].T
        for i in range(self.CLAN):
            c_label = list(np.array(np.where(self.img['gt3'] == i)).T)
            if shuffle_opt:
                random.shuffle(c_label)
            if n_validation>0:
                self.validate['XY'].extend(c_label[:n_validation])
            self.test['XY'].extend(c_label[:])
        # test shuffle and format convert
        if shuffle_opt:
            random.shuffle(self.test['XY'])
        for i in self.test['XY']:
            self.test['sample'].append(self.get_sample(i))
            self.test['label'].append(self.get_label(i))
        for i in self.test:
            self.test[i] = np.array(self.test[i])
        # validation shuffle and format convert
        if n_validation > 0:
            if shuffle_opt:
                random.shuffle(self.validate['XY'])
            for i in self.validate['XY']:
                self.validate['sample'].append(self.get_sample(i))
                self.validate['label'].append(self.get_label(i))
            for i in self.validate:
                self.validate[i] = np.array(self.validate[i])
        # end of load_predefine_test
        

    def load_all_sam(self, SepId = 0):
        """
        Load all the samples in batch.
        If SepId = 0, load all the samples once.

        :param SepId: batch id
        """

        # setting the Y coordinates
        JS = 0
        JE = self.IMGY
        # setting the X coordinates
        if SepId == 0:  # load all the samples of the given image
            IS = 0
            IE = self.IMGX
            print('Load all the samples of the given image')
        else:  # load partial samples of the given image
            if self.SEPN < 2:  # refine the total number of batches for loading data
                self.SEPN = 2
            bs = int(np.ceil(self.IMGX // self.SEPN))
            IS = bs * (SepId-1)
            if SepId < self.SEPN:
                IE = bs * SepId
            else:
                IE = self.IMGX
            print('Load all the samples in batch.')
            print('X: [{0:d},{1:d}]    Y: [{2:d},{3:d}]'.format(IS + 1, IE, JS + 1, JE))
        # load samples
        self.all['XY'] = np.zeros([(IE-IS) * (JE-JS), 2])
        id = 0
        for i in range(IS,IE):
            for j in range(JS,JE):
                xy = np.array([i, j])
                self.all['XY'][id, :] = xy
                self.all['sample'].append(self.get_sample(xy))
                id += 1
        for i in self.all:
            self.all[i] = np.array(self.all[i])
        # end of load_all_sam


    def st_merge(self, predict):
        indexs = np.where(self.flush['label'] != predict)
        indexs = list(indexs[0])
        for i in self.flush:
            self.flush[i] = np.delete(self.flush[i], indexs, axis=0)
            try:
                self.candidate[i] = np.append(
                        self.candidate[i],
                        self.flush[i],
                        0)
            except ValueError:
                self.candidate[i] = self.flush[i]
        self.candidate['label_v'] = to_categorical(
                self.candidate['label'],
                self.CLAN)
        for i in ['XY', 'label']:
            self.self_sam[i] = np.append(self.self_sam[i], self.flush[i], 0)
        for i in self.flush['XY']:
            self.self_sam['map'][i[0]][i[1]] = 1


    def get_sample(self, xy):
        d = self.SAMD
        x = xy[0]
        y = xy[1]
        try:
            self.img['im'][x][y]
        except IndexError:
            return []
        x += d
        y += d
        sam = self.img['add'][(x - d): (x + d + 1), (y - d): (y + d + 1)]
        return np.array(sam)


    def get_label(self, xy):
        return self.img['gt'][xy[0]][xy[1]]


    def get_label_train(self, xy):
        return self.img['gt2'][xy[0]][xy[1]]


    def get_label_test(self, xy):
        return self.img['gt3'][xy[0]][xy[1]]


    def get_seg(self, xy):
        return self.img['seg'][xy[0]][xy[1]]


    def add_sample(self, indexs):
        new = {}
        for i in self.train:
             new[i] = self.candidate[i][indexs]
             self.candidate[i] = np.delete(self.candidate[i], indexs, axis=0)
        new = self.make_sample(new)
        for i in self.train:
            self.train[i] = np.append(self.train[i], new[i], 0)
