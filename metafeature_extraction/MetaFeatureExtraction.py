from utils import shannon_entropy, read_nifti, mutual_information, correlation_coefficient, normalize
import numpy as np
from scipy.stats import kurtosis, skew, pearsonr
from scipy.ndimage import zoom
from tqdm import tqdm
import itertools
import cv2
import random
import os

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

class MetaFeatureExtraction():
    def __init__(self, task, subset_size, fe = '', model = None, nr_of_filters = None):
        # self.id = id
        self.task = task
        self.subset_size = subset_size
        self.fe = fe
        if model:
            self.load_model(model.feature_extractor)
        if nr_of_filters:
            self.nr_of_filters = nr_of_filters
    def load_model(self, model):
        self.model = model
    def im2features(self, im):
        im = im.astype('float32')
        im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        im = cv2.resize(im, (224,224))
        im = np.expand_dims(im, axis=0)
        im = preprocess_input(im)
        return self.model.predict(im)
    def sum_and_log_meta_features(self):
        if len(self.meta_features.shape) == 5:
            self.meta_features = np.log(np.sum(self.meta_features, axis = 1))
            self.meta_features[self.meta_features==-np.inf]=0
            self.meta_features = np.reshape(self.meta_features, (self.meta_features.shape[0],self.meta_features.shape[1]*self.meta_features.shape[2],self.meta_features.shape[3]))
    def gather_random_addresses(self):
        self.addresses = random.sample(os.listdir(r'/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(self.task)), self.subset_size)
        for i in range(self.subset_size):
            self.addresses[i] = os.path.join(r'/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(self.task), self.addresses[i])
    def gather_address(self, nr):
        self.addresses = [os.listdir(r'/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(self.task))[nr]]
        self.addresses[0] = os.path.join(r'/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(self.task), self.addresses[0])
    def convert_address(self, index):
        if self.addresses[index][-10] == '_':
            return self.addresses[index][-9:-7]
        else:
            return self.addresses[index][-10:-7]
    def gather_all_addresses(self):
        self.addresses = os.listdir('/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(self.task))
        for i in range(len(self.addresses)):
            self.addresses[i] = os.path.join('/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(self.task), self.addresses[i])
    def gather_mean_meta_labels(self):
        self.meta_labels = np.array(np.zeros(19))
        for address in self.addresses:
            nr = address[-10:-7]
            if nr[0] == '_':
                nr = nr[1:]
            elif nr[1] == '_':
                nr = nr[2:]
            label = np.load('/home/tjvsonsbeek/featureExtractorUnet/metadata/{}/{}.npy'.format(self.task, nr))
            for lab in range(len(label)):
                self.meta_labels[lab] += label[lab]/len(self.addresses)

    def gather_list_meta_labels(self):
        self.meta_labels = np.zeros((self.subset_size,19))
        for i, address in enumerate(self.addresses):
            nr = address[-10:-7]
            if nr[0] == '_':
                nr = nr[1:]
            elif nr[1] == '_':
                nr = nr[2:]
            label = np.load(r'/home/tjvsonsbeek/featureExtractorUnet/metadata/{}/{}.npy'.format(self.task, nr))
            for lab in range(len(label)):
                self.meta_labels[i, lab] = label[lab]
    def save_meta_features(self):
        np.save('metadata/meta_regressor_features_{}_{}.npy'.format(self.task, self.type), self.meta_features)
    def save_meta_labels(self):
        np.save('metadata/meta_regressor_labels_{}_{}.npy'.format(self.task, self.type), self.meta_labels)
    def load_meta_features(self):
            if self.fe == 'STAT':
                if self.task == 'Task11_CHAOSLiver':
                    self.meta_features = np.load('metadata/statistical/meta_regressor_features_{}_{}.npy'.format(self.task, self.fe))
                else:
                    self.meta_features = np.load('metadata/statistical/meta_regressor_features_{}_{}_{}.npy'.format(self.subset_size, self.task, self.fe))
            else:
                if self.task == 'Task11_CHAOSLiver':
                    self.meta_features = np.load('metadata/deeplearning/meta_regressor_features_{}_{}.npy'.format(self.task, self.fe))
                else:
                    self.meta_features = np.load('metadata/deeplearning/meta_regressor_features_{}_{}_{}.npy'.format(self.subset_size, self.task, self.fe))
    def load_meta_labels(self):
        if self.fe == 'STAT':
            if self.task == 'Task11_CHAOSLiver':
                self.meta_labels = np.load('metadata/statistical/meta_regressor_labels_{}.npy'.format(self.task))
            else:
                self.meta_labels = np.load('metadata/statistical/meta_regressor_labels_{}_{}_{}.npy'.format(self.subset_size,self.task, self.fe))
        else:
            if self.task == 'Task11_CHAOSLiver':
                self.meta_labels = np.load('metadata/statistical//meta_regressor_labels_{}.npy'.format(self.task))
            else:
                self.meta_labels = np.load('metadata/deeplearning/meta_regressor_labels_{}_{}_{}.npy'.format(self.subset_size,self.task, self.fe))
    def gather_histograms(self, bins = 100):
        for index in tqdm(range(len(self.addresses))):

            im = normalize(read_nifti(self.addresses[index]))
            nr_of_histograms = np.prod(im.shape[2:])
            hist_array  = np.zeros((nr_of_histograms, bins))

            if len(im.shape)==3:
                for nr in range(im.shape[2]):
                    hist, edges = np.histogram(im[:,:,nr], bins = bins)
                    hist_array[nr,:] = hist
            elif len(im.shape)==4:
                for nr1 in range(im.shape[2]):
                    for nr2 in range(im.shape[3]):
                        hist, edges = np.histogram(im[:,:,nr1,nr2], bins = bins)
                        hist_array[nr2*(nr1+1),:] = hist
            else:
                print("Shouldnt happen")
            if self.addresses[index][-10] == '_':
                name = self.addresses[index][-9:-7]
            else:
                name = self.addresses[index][-10:-7]
            if not os.path.isdir("/home/tjvsonsbeek/featureExtractorUnet/metadata/histograms/{}".format(self.task)):
                    os.mkdir("/home/tjvsonsbeek/featureExtractorUnet/metadata/histograms/{}".format(self.task))
            np.save('/home/tjvsonsbeek/featureExtractorUnet/metadata/histograms/{}/{}.npy'.format(self.task, name), hist_array)
    def gather_meta_features_DL(self):
        global_features = np.zeros((self.subset_size,1000))
        for index in range(len(self.addresses)):
            im = normalize(read_nifti(self.addresses[index]))

            if len(im.shape) == 3:
                for i in range(im.shape[2]):
                    sub_im = im[:,:,i]
                    local_features = self.im2features(sub_im)
                    global_features[index,:] += local_features[0,:]
            elif len(im.shape) == 4:
                for i in range(im.shape[2]):
                    for j in range(im.shape[3]):
                        sub_im = im[:,:,i,j]
                        local_features = self.im2features(sub_im)
                        global_features[index,:] += local_features[0,:]
        # global_features = global_features
        self.meta_features = global_features
    def gather_meta_features(self):
        if self.fe == 'STAT':
            self.gather_meta_features_stat()
        else:
            self.gather_meta_features_DL_NoTop()
    def gather_meta_features_DL_NoTop(self):
        global_features = np.zeros((self.subset_size,7,7, self.nr_of_filters))
        for index in range(len(self.addresses)):
            im = normalize(read_nifti(self.addresses[index]))

            if len(im.shape) == 3:
                for i in [int(im.shape[2]/2)]:
                    sub_im = im[:,:,i]
                    local_features = self.im2features(sub_im)
                    global_features[index,:,:,:] += local_features[0,:,:,:]
            elif len(im.shape) == 4:
                for i in [int(im.shape[2]/2)]:
                    for j in range(im.shape[3]):
                        sub_im = im[:,:,i,j]
                        local_features = self.im2features(sub_im)
                        global_features[index,:,:,:] += local_features[0,:,:,:]
        self.meta_features = global_features
    def gather_meta_features_stat(self):
        meta_features = np.array(np.zeros((33)))
        # lists for features are made to get mean and std of dataset later
        meta_skew = []
        meta_kurtosis = []
        meta_entropy = []
        meta_median = []
        meta_std = []
        meta_mutinf = []
        meta_mean = []
        meta_corr = []
        meta_sparsity = []
        meta_xy_axis = []
        meta_z_axis = []

        for index in range(len(self.addresses)):
            im = normalize(read_nifti(self.addresses[index]))
            # mean skewness


            meta_skew.append(skew(im,axis = None))
            # mean kurtosis
            meta_kurtosis.append(kurtosis(im, axis = None))
            # normalized class entropy
            meta_entropy.append(shannon_entropy(im, base = len(im.shape)))
            #median pixel value
            meta_median.append(np.median(im))
            # mean pixel value
            meta_mean.append(np.mean(im))
            # sparsity
            meta_sparsity.append(np.count_nonzero(im)/np.prod(im.shape))
            # xy axis
            meta_xy_axis.append((im.shape[0]+im.shape[1])/2)
            # z axis
            meta_z_axis.append(im.shape[2])
            pairs = list(range(index+1,len(self.addresses)))
            # comparison of images wihtin dataset with each other
            for pair in pairs:
                im2 = normalize(read_nifti(self.addresses[pair]))
                im_adj = im
                im2_adj = im2
                print(im2.shape)
                print(im.shape)
                if im_adj.shape != im2_adj.shape:
                    if len(im_adj.shape) == 3:
                        print("over here")
                        for axis in range(len(im_adj.shape)):
                                if axis == 0:
                                    print("cp0")
                                    if im_adj.shape[axis]>im2_adj.shape[axis]:
                                        im_adj = zoom(im_adj, (im2_adj.shape[axis]/im_adj.shape[axis], 1, 1))
                                        print("c1")
                                    elif im_adj.shape[axis]<im2_adj.shape[axis]:
                                        im2_adj = zoom(im2_adj, (im_adj.shape[axis]/im2_adj.shape[axis], 1, 1))
                                        print("c2")
                                if axis == 1:
                                    print("cp1")
                                    if im_adj.shape[axis]>im2_adj.shape[axis]:
                                        im_adj = zoom(im_adj, (1,im2_adj.shape[axis]/im_adj.shape[axis], 1))
                                        print("c1")
                                    elif im_adj.shape[axis]<im2_adj.shape[axis]:
                                        im2_adj = zoom(im2_adj, (1,im_adj.shape[axis]/im2_adj.shape[axis], 1))
                                        print("c2")
                                if axis == 2:
                                    print("cp2")
                                    if im_adj.shape[axis]>im2_adj.shape[axis]:
                                        im_adj = zoom(im_adj, (1,1,im2_adj.shape[axis]/im_adj.shape[axis]))
                                        print("c1")
                                    elif im_adj.shape[axis]<im2_adj.shape[axis]:
                                        im2_adj = zoom(im2_adj, (1,1,im_adj.shape[axis]/im2_adj.shape[axis]))
                                        print("c2")
                    else:
                        for axis in range(len(im.shape)):
                                if axis == 0:
                                    if im_adj.shape[axis]>=im2_adj.shape[axis]:
                                        im_adj = im_adj[0:im2_adj.shape[axis],:,:,:]
                                    else:
                                        im2_adj = im2_adj[0:im_adj.shape[axis],:,:,:]
                                if axis == 1:
                                    if im_adj.shape[axis]>=im2_adj.shape[axis]:
                                        im_adj = im_adj[:,0:im2_adj.shape[axis],:,:]
                                    else:
                                        im2_adj = im2_adj[:,0:im_adj.shape[axis],:,:]
                                if axis == 2:
                                    if im_adj.shape[axis]>=im2_adj.shape[axis]:
                                        im_adj = im_adj[:,:,0:im2_adj.shape[axis],:]
                                    else:
                                        im2_adj = im2_adj[:,:,0:im_adj.shape[axis],:]
                                if axis == 3:
                                    if im_adj.shape[axis]>=im2_adj.shape[axis]:
                                        im_adj = im_adj[:,:,:,0:im2_adj.shape[axis]]
                                    else:
                                        im2_adj = im2_adj[:,:,:,0:im_adj.shape[axis]]

                print(im2_adj.shape)
                print(im_adj.shape)

                corr = correlation_coefficient(im_adj, im2_adj)
                meta_corr.append(corr)
                # mean mutual information of class and attirbute
                hist_2d, x_edges, y_edges = np.histogram2d(im_adj.ravel(), im2_adj.ravel(), bins=255)
                meta_mutinf.append(mutual_information(hist_2d))
        # nr of instances
        meta_features[0] = len(self.addresses)

        # mean pixel value
        meta_features[1] = np.mean(meta_mean)
        # std pixel value
        meta_features[2] = np.std(meta_mean)
        # coefficient of variation
        meta_features[3] = np.std(meta_mean)/np.mean(meta_mean)

        # mean skew value
        meta_features[4] = np.mean(meta_skew)
        # std of skew value
        meta_features[5] = np.std(meta_skew)
        # coefficient of variation of skew
        meta_features[6] = np.std(meta_skew)/np.mean(meta_skew)

        # mean kurtosis value
        meta_features[7] = np.mean(meta_kurtosis)
        # std of kurtosis value
        meta_features[8] = np.std(meta_kurtosis)
        # coefficient of variation of kurtosis
        meta_features[9] = np.std(meta_kurtosis)/np.mean(meta_kurtosis)

        # mean entropy value
        meta_features[10] = np.mean(meta_entropy)
        # std of entropy value
        meta_features[11] = np.std(meta_entropy)
        # coefficient of variation of entropy
        meta_features[12] = np.std(meta_entropy)/np.mean(meta_entropy)

        # mean median value
        meta_features[13] = np.mean(meta_median)
        # std of median value
        meta_features[14] = np.std(meta_median)
        # coefficient of variation of median # filtered out

        # mean mutinf value
        meta_features[15] = np.mean(meta_mutinf)
        # std of mutinf value
        meta_features[16] = np.std(meta_mutinf)
        # coefficient of variation of mutinf
        meta_features[17] = np.std(meta_mutinf)/np.mean(meta_mutinf)
        # max mutinf value
        meta_features[18] = np.max(meta_mutinf)

        # mean corr value
        meta_features[19] = np.mean(meta_corr)
        # std of corr value
        meta_features[20] = np.std(meta_corr)
        # coefficient of variation of corr
        meta_features[21] = np.std(meta_corr)/np.mean(meta_corr)

        # mean sparsity value
        meta_features[22] = np.mean(meta_sparsity)
        # std of sparsity value
        meta_features[23] = np.std(meta_sparsity)
        # coefficient of variation of sparsity
        meta_features[24] = np.std(meta_sparsity)/np.mean(meta_sparsity)

        # mean xy_axis value
        meta_features[25] = np.mean(meta_xy_axis)
        # std of xy_axis value
        meta_features[26] = np.std(meta_xy_axis)
        # coefficient of variation of xy_axis
        meta_features[27] = np.std(meta_xy_axis)/np.mean(meta_xy_axis)

        # mean z_axis value
        meta_features[28] = np.mean(meta_z_axis)
        # std of z_axis value
        meta_features[29] = np.std(meta_z_axis)
        # coefficient of variation of z_axis
        meta_features[30] = np.std(meta_z_axis)/np.mean(meta_z_axis)

        # equivalent number of features
        meta_features[31] = meta_features[10]/meta_features[15]
        # noise signal ratio
        meta_features[32] = (meta_features[10] - meta_features[15])/meta_features[15]

        self.meta_features = meta_features
        # np.save("metadata/meta_regression_{}.npy".format(self.id), meta_features)
def load_histograms(self):
    a=1
# def gather_meta_features_from_histogram(self):
    # histogram_dict = {}
    # for index, address in enumerate(self.addresses):
    #     histogram_dict[convert_address(index)] = np.load("/home/tjvsonsbeek/featureExtractorUnet/metadata/histograms/{}/{}.npy".format(self.task, convert_address(index)))
    #
    #
    #
    #
    # meta_features = np.array(np.zeros((37)))
    # # lists for features are made to get mean and std of dataset later
    # meta_skew = []
    # meta_kurtosis = []
    # meta_entropy = []
    # meta_median = []
    # meta_std = []
    # meta_mutinf = []
    # meta_mean = []
    # meta_corr = []
    # meta_sparsity = []
    # meta_xy_axis = []
    # meta_z_axis = []
    #
    # for histogram in histogram_dict.keys():
    #     for slice in range(histogram.shape[0])
    #     # mean skewness
    #     meta_skew.append(skew(im,axis = None))
    #     # mean kurtosis
    #     meta_kurtosis.append(kurtosis(im, axis = None))
    #     # normalized class entropy
    #     meta_entropy.append(shannon_entropy(im, base = len(im.shape)))
    #     #median pixel value
    #     meta_median.append(np.median(im))
    #     # mean pixel value
    #     meta_mean.append(np.mean(im))
    #     # sparsity
    #     meta_sparsity.append(np.count_nonzero(im)/np.prod(im.shape))
    #     # xy axis
    #     meta_xy_axis.append((im.shape[0]+im.shape[1])/2)
    #     # z axis
    #     meta_z_axis.append(im.shape[2])
    #     pairs = list(range(index+1,len(self.addresses)))
    #     # comparison of images wihtin dataset with each other
    #     for pair in pairs:
    #         im2 = normalize(read_nifti(self.addresses[pair]))
    #         im_adj = im
    #         im2_adj = im2
    #         print(im2.shape)
    #         print(im.shape)
    #         if im_adj.shape != im2_adj.shape:
    #             if len(im_adj.shape) == 3:
    #                 print("over here")
    #                 for axis in range(len(im_adj.shape)):
    #                         if axis == 0:
    #                             print("cp0")
    #                             if im_adj.shape[axis]>im2_adj.shape[axis]:
    #                                 im_adj = zoom(im_adj, (im2_adj.shape[axis]/im_adj.shape[axis], 1, 1))
    #                                 print("c1")
    #                             elif im_adj.shape[axis]<im2_adj.shape[axis]:
    #                                 im2_adj = zoom(im2_adj, (im_adj.shape[axis]/im2_adj.shape[axis], 1, 1))
    #                                 print("c2")
    #                         if axis == 1:
    #                             print("cp1")
    #                             if im_adj.shape[axis]>im2_adj.shape[axis]:
    #                                 im_adj = zoom(im_adj, (1,im2_adj.shape[axis]/im_adj.shape[axis], 1))
    #                                 print("c1")
    #                             elif im_adj.shape[axis]<im2_adj.shape[axis]:
    #                                 im2_adj = zoom(im2_adj, (1,im_adj.shape[axis]/im2_adj.shape[axis], 1))
    #                                 print("c2")
    #                         if axis == 2:
    #                             print("cp2")
    #                             if im_adj.shape[axis]>im2_adj.shape[axis]:
    #                                 im_adj = zoom(im_adj, (1,1,im2_adj.shape[axis]/im_adj.shape[axis]))
    #                                 print("c1")
    #                             elif im_adj.shape[axis]<im2_adj.shape[axis]:
    #                                 im2_adj = zoom(im2_adj, (1,1,im_adj.shape[axis]/im2_adj.shape[axis]))
    #                                 print("c2")
    #             else:
    #                 for axis in range(len(im.shape)):
    #                         if axis == 0:
    #                             if im_adj.shape[axis]>=im2_adj.shape[axis]:
    #                                 im_adj = im_adj[0:im2_adj.shape[axis],:,:,:]
    #                             else:
    #                                 im2_adj = im2_adj[0:im_adj.shape[axis],:,:,:]
    #                         if axis == 1:
    #                             if im_adj.shape[axis]>=im2_adj.shape[axis]:
    #                                 im_adj = im_adj[:,0:im2_adj.shape[axis],:,:]
    #                             else:
    #                                 im2_adj = im2_adj[:,0:im_adj.shape[axis],:,:]
    #                         if axis == 2:
    #                             if im_adj.shape[axis]>=im2_adj.shape[axis]:
    #                                 im_adj = im_adj[:,:,0:im2_adj.shape[axis],:]
    #                             else:
    #                                 im2_adj = im2_adj[:,:,0:im_adj.shape[axis],:]
    #                         if axis == 3:
    #                             if im_adj.shape[axis]>=im2_adj.shape[axis]:
    #                                 im_adj = im_adj[:,:,:,0:im2_adj.shape[axis]]
    #                             else:
    #                                 im2_adj = im2_adj[:,:,:,0:im_adj.shape[axis]]
    #
    #         print(im2_adj.shape)
    #         print(im_adj.shape)
    #
    #         corr = correlation_coefficient(im_adj, im2_adj)
    #         meta_corr.append(corr)
    #         # mean mutual information of class and attirbute
    #         hist_2d, x_edges, y_edges = np.histogram2d(im_adj.ravel(), im2_adj.ravel(), bins=255)
    #         meta_mutinf.append(mutual_information(hist_2d))
    # # nr of instances
    # meta_features[0] = len(self.addresses)
    #
    # # mean pixel value
    # meta_features[1] = np.mean(meta_mean)
    # # std pixel value
    # meta_features[2] = np.std(meta_mean)
    # # coefficient of variation
    # meta_features[3] = np.std(meta_mean)/np.mean(meta_mean)
    #
    # # mean std value
    # meta_features[4] = np.mean(meta_std)
    # # std of std value
    # meta_features[5] = np.std(meta_std)
    # # coefficient of variation of std
    # meta_features[6] = np.std(meta_std)/np.mean(meta_std)
    #
    # # mean skew value
    # meta_features[7] = np.mean(meta_skew)
    # # std of skew value
    # meta_features[8] = np.std(meta_skew)
    # # coefficient of variation of skew
    # meta_features[9] = np.std(meta_skew)/np.mean(meta_skew)
    #
    # # mean kurtosis value
    # meta_features[10] = np.mean(meta_kurtosis)
    # # std of kurtosis value
    # meta_features[11] = np.std(meta_kurtosis)
    # # coefficient of variation of kurtosis
    # meta_features[12] = np.std(meta_kurtosis)/np.mean(meta_kurtosis)
    #
    # # mean entropy value
    # meta_features[13] = np.mean(meta_entropy)
    # # std of entropy value
    # meta_features[14] = np.std(meta_entropy)
    # # coefficient of variation of entropy
    # meta_features[15] = np.std(meta_entropy)/np.mean(meta_entropy)
    #
    # # mean median value
    # meta_features[16] = np.mean(meta_median)
    # # std of median value
    # meta_features[17] = np.std(meta_median)
    # # coefficient of variation of median
    # meta_features[18] = np.std(meta_median)/np.mean(meta_median)
    #
    # # mean mutinf value
    # meta_features[19] = np.mean(meta_mutinf)
    # # std of mutinf value
    # meta_features[20] = np.std(meta_mutinf)
    # # coefficient of variation of mutinf
    # meta_features[21] = np.std(meta_mutinf)/np.mean(meta_mutinf)
    # # max mutinf value
    # meta_features[22] = np.max(meta_mutinf)
    #
    # # mean corr value
    # meta_features[23] = np.mean(meta_corr)
    # # std of corr value
    # meta_features[24] = np.std(meta_corr)
    # # coefficient of variation of corr
    # meta_features[25] = np.std(meta_corr)/np.mean(meta_corr)
    #
    # # mean sparsity value
    # meta_features[26] = np.mean(meta_sparsity)
    # # std of sparsity value
    # meta_features[27] = np.std(meta_sparsity)
    # # coefficient of variation of sparsity
    # meta_features[28] = np.std(meta_sparsity)/np.mean(meta_sparsity)
    #
    # # mean xy_axis value
    # meta_features[29] = np.mean(meta_xy_axis)
    # # std of xy_axis value
    # meta_features[30] = np.std(meta_xy_axis)
    # # coefficient of variation of xy_axis
    # meta_features[31] = np.std(meta_xy_axis)/np.mean(meta_xy_axis)
    #
    # # mean z_axis value
    # meta_features[32] = np.mean(meta_z_axis)
    # # std of z_axis value
    # meta_features[33] = np.std(meta_z_axis)
    # # coefficient of variation of z_axis
    # meta_features[34] = np.std(meta_z_axis)/np.mean(meta_z_axis)
    #
    # # equivalent number of features
    # meta_features[35] = meta_features[13]/meta_features[19]
    # # noise signal ratio
    # meta_features[36] = (meta_features[13] - meta_features[19])/meta_features[19]
    #
    # self.meta_features = meta_features
    # # np.save("metadata/meta_regression_{}.npy".format(self.id), meta_features)
