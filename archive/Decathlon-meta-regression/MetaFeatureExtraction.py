from utils import meta_generator_OM, shannon_entropy, read_nifti, mutual_information, correlation_coefficient, normalize
import numpy as np
from scipy.stats import kurtosis, skew, pearsonr
from scipy.ndimage import zoom
import itertools
import cv2
import random
import os
class MetaFeatureExtraction():
    def __init__(self,task):
        # self.id = id
        self.task = task

    def gather_addresses(self):
        self.addresses = random.sample(os.listdir('/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(self.task)), 5)
        for i in range(5):
            self.addresses[i] = os.path.join('/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(self.task), self.addresses[i])

    def gather_meta_labels(self):
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
    def save_meta_features(self):
        np.save('metadata/meta_regressor_features_{}.npy'.format(self.task), self.meta_features)
    def save_meta_labels(self):
        np.save('metadata/meta_regressor_labels_{}.npy'.format(self.task), self.meta_labels)
    def load_meta_features(self):
        self.meta_features = np.load('metadata/meta_regressor_features_{}.npy'.format(self.task))
    def load_meta_labels(self):
        self.meta_labels = np.load('metadata/meta_regressor_labels_{}.npy'.format(self.task))

    def gather_meta_features(self):
        meta_features = np.array(np.zeros((13)))
        mutualInformation = []
        meanPixelValue    = []
        # nr of instances
        meta_features[0] = len(self.addresses)
        # number of features
        # meta_features[1] = 1
        # number of target concept values
        # meta_features[2] = 1#float(np.count_nonzero(label)/(label.shape[0]*label.shape[2]))
        # dataset dimensionality
        # meta_features[3] = meta_features[0]/meta_features[1]

        for index in range(len(self.addresses)):
            print(self.addresses[index])
            im = normalize(read_nifti(self.addresses[index]))
            # mean skewness
            meta_features[5] += skew(im,axis = None)/len(self.addresses)
            # mean kurtosis
            meta_features[6] += kurtosis(im, axis = None)/len(self.addresses)

            # normalized class entropy
            meta_features[7] += shannon_entropy(im, base = len(im.shape))/len(self.addresses)
            # mean normalized feature entropy
            meta_features[8] += shannon_entropy(im, base = len(im.shape))/len(self.addresses)

            # mean pixel value
            meanPixelValue.append(np.mean(im))

            pairs = list(range(index+1,len(self.addresses)))

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
                # mean absolute linear coefficient ofall possible pairs of features
                # [corr, plch] = pearsonr(im, im2)
                corr = correlation_coefficient(im_adj, im2_adj)
                meta_features[4] += corr/len(self.addresses)
                # mean mutual information of class and attirbute
                hist_2d, x_edges, y_edges = np.histogram2d(im_adj.ravel(), im2_adj.ravel(), bins=255)
                mutualInformation.append(mutual_information(hist_2d))

        # mean mutual information of class and attirbute
        meta_features[9] = np.mean(mutualInformation)
        # max mutual information of class and attirbute
        meta_features[10] = np.max(mutualInformation)
        # equivalent number of features
        meta_features[11] = meta_features[7]/meta_features[9]
        # noise signal ratio
        meta_features[12] = (meta_features[8] - meta_features[9])/meta_features[9]
        # mean pixel values
        meta_features[1] = np.mean(meanPixelValue)
        # std pixel value
        meta_features[2] = np.std(meanPixelValue)
        # coefficient of variation
        meta_features[3] = np.std(meanPixelValue)/np.mean(meanPixelValue)
        self.meta_features = meta_features
        # np.save("metadata/meta_regression_{}.npy".format(self.id), meta_features)
