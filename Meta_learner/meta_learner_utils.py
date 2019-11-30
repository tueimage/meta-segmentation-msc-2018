import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np
import nibabel as nib
import random
import cv2
from matplotlib import pyplot as plt
import keras.backend as K
import scipy.misc
import seaborn as sns
from itertools import cycle
from sklearn.manifold import TSNE
import csv
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression

def feature_selection(regression_features_raw, regression_labels, frac):
    ff = np.zeros((regression_features_raw.shape[1], 19))
    for p in range(19):
        ff[:,p], _ = f_regression(regression_features_raw, regression_labels[:,p])
    ff = np.nanmean(ff, axis = 1)

    features_to_keep = np.argsort(ff)[-int(ff.shape[0]*frac):]

    regression_features = np.zeros((regression_features_raw.shape[0], len(features_to_keep)))

    for i, f in enumerate(features_to_keep):
        regression_features[:,i] = regression_features_raw[:,f]
    return regression_features, features_to_keep

def feature_selection_test_set(regression_features_raw, features_to_keep):
    regression_features = np.zeros((regression_features_raw.shape[0], len(features_to_keep)))

    for i, f in enumerate(features_to_keep):
        regression_features[:,i] = regression_features_raw[:,f]
    return regression_features

def add_task_specific_metafeatures(regression_features_raw, task_specific_features, tasks_list, sample_size):
    regression_features_raw[regression_features_raw==-np.inf] = 0
    regression_features_raw = np.concatenate([regression_features_raw, np.zeros((len(tasks_list)*sample_size, 5))],axis=1)
    for task_id, task in enumerate(tasks_list):
        for nr in range(sample_size):
            regression_features_raw[task_id*sample_size+nr,-5:] = task_specific_features[task_id,:]
    return regression_features_raw
def joint_normalize_metafeatures(metafeatures1, metafeatures2):
    """
    Normalization of two sets of metafeatures. Input is two set of metafeatures, output is the same set but normalized. Normalized metafeatures have mean 0 and standard deviation 1
    """
    normalized_metafeatures1 = np.zeros_like(metafeatures1)
    normalized_metafeatures2 = np.zeros_like(metafeatures2)
    metafeatures_joint = np.concatenate([metafeatures1, metafeatures2], axis = 0)
    for c in range(metafeatures_joint.shape[1]):
        std = np.std(metafeatures_joint[:,c])
        if std == 0:
            std = 1
        normalized_metafeatures1[:,c] = (metafeatures1[:,c]-np.mean(metafeatures_joint[:,c]))/std
        normalized_metafeatures2[:,c] = (metafeatures2[:,c]-np.mean(metafeatures_joint[:,c]))/std
    return normalized_metafeatures1, normalized_metafeatures2

def normalize_metafeatures(metafeatures):
    """
    Normalization of metafeatures. Normalized metafeatures have mean 0 and standard deviation 1
    """
    normalized_metafeatures = np.zeros_like(metafeatures)
    for c in range(metafeatures.shape[1]):
        std = np.std(metafeatures[:,c])
        if std == 0:
            std = 1
        normalized_metafeatures[:,c] = (metafeatures[:,c]-np.mean(metafeatures[:,c]))/std

    return normalized_metafeatures

def simple_single_model(nr_of_features = 1):
    model = Sequential()
    model.add(Dense(50, input_dim=nr_of_features, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(30, activation='relu'))#,activity_regularizer=l1(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)
    return model


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)[0]

def regression_feature_selection(train_features, train_labels, test_features, percent):
    # print(train_features[0,:])
    # print(train_labels)
    ff = np.zeros((train_features.shape[1], train_labels.shape[1]))
    for p in range(train_labels.shape[1]):
        ff[:,p], _ = f_regression(train_features, train_labels[:,p])
        # print(ff)
    ff = np.nanmean(ff, axis = 1)
    # print(ff)
    features_to_keep = np.argsort(ff)[-int(ff.shape[0]/percent):]

    print(len(features_to_keep))
    print(len(features_to_keep))
    # threshold = int(np.nanmean(ff)*2)
    # features_to_keep = largest_indices(ff,10)

    new_train_features = np.zeros((train_features.shape[0], len(features_to_keep)))
    new_test_features = np.zeros((test_features.shape[0], len(features_to_keep)))


    for i, f in enumerate(features_to_keep):
        new_train_features[:,i] = train_features[:,f]
        new_test_features[:,i] = test_features[:,f]
    return new_train_features, new_test_features
def preprocess_metafeatures(metafeatures, tasks_list,sample_size):
    nr_of_subsets = sample_size
    nr_of_filters = metafeatures[tasks_list[0]].shape[2]
    fmaps = np.zeros((len(tasks_list)*nr_of_subsets, nr_of_filters))
    for fmap in tqdm(range(nr_of_filters)):
        for i_id, i in enumerate(tasks_list):
            for p in range(nr_of_subsets):
                for j_id, j in enumerate(tasks_list):
                    k = (metafeatures[i][p,:,fmap]>0)*1
                    l = (metafeatures[j][p,:,fmap]>0)*1
                    if np.count_nonzero(k) == 0 and np.count_nonzero(l) == 0:
                        continue
                    overlap = (49-np.sum(np.absolute(k-l)))/49
                    if overlap>0.80 and i!=j:
                        fmaps[i_id*nr_of_subsets+p,fmap]=1
                        fmaps[j_id*nr_of_subsets+p,fmap]=1
    usable_filters = np.zeros(nr_of_filters)

    for f in range(nr_of_filters):
        if np.count_nonzero(fmaps[:,nr_of_filters-f-1])==0:
            usable_filters[nr_of_filters-f-1]=1
            fmaps = np.delete(fmaps, nr_of_filters-f-1,axis=1)
    new_metafeatures = np.zeros((len(tasks_list)*nr_of_subsets,np.count_nonzero(usable_filters)*49))
    count = 0
    for filter in np.nonzero(usable_filters)[0]:
        for i_id, i in enumerate(tasks_list):
            for p in range(nr_of_subsets):
                new_metafeatures[i_id*nr_of_subsets + p, count*49:(count+1)*49] = metafeatures[i][p,:,filter]
        count+=1
    return new_metafeatures.astype(np.float64), fmaps.astype(np.float64), usable_filters
def preprocess_metalabels(metalabels, tasks_list, sample_size):
    nr_of_subsets = sample_size
    new_metalabels = np.zeros((len(tasks_list)*nr_of_subsets,19))

    for id, task in enumerate(tasks_list):
        new_metalabels[id*nr_of_subsets:(id+1)*nr_of_subsets,:] =  (np.sum(metalabels[task], axis = 1)/metalabels[task].shape[1])[:sample_size,:]
    return new_metalabels
def preprocess_metafeatures_test_set(metafeatures, tasks_list,sample_size, usable_filters):
    nr_of_subsets = sample_size
    nr_of_filters = metafeatures[tasks_list[0]].shape[2]
    fmaps = np.zeros((len(tasks_list[10:])*sample_size, nr_of_filters))

    for fmap in tqdm(range(nr_of_filters)):
        for i_id, i in enumerate(tasks_list[:10]):
            for p in range(nr_of_subsets):
                for j_id, j in enumerate(tasks_list[10:]):
                    k = (metafeatures[i][p,:,fmap]>0)*1
                    l = (metafeatures[j][p,:,fmap]>0)*1
                    if np.count_nonzero(k) == 0 and np.count_nonzero(l) == 0:
                        continue
                    overlap = (49-np.sum(np.absolute(k-l)))/49
                    if overlap>0.80:
                        fmaps[j_id*nr_of_subsets+p,fmap]=1

    for f in range(nr_of_filters):
        if usable_filters[nr_of_filters-f-1] == 1:
            fmaps = np.delete(fmaps, nr_of_filters-f-1,axis=1)

    new_metafeatures = np.zeros((sample_size*len(tasks_list[10:]),np.count_nonzero(usable_filters)*49))
    # count = 0
    # for filter in np.nonzero(usable_filters)[0]:
    #     new_metafeatures[0, count*49:(count+1)*49] = metafeatures[i][p,:,filter]
    #     count+=1
    return new_metafeatures.astype(np.float64), fmaps.astype(np.float64)

def subset_index_to_address(subset_indices, train_addresses):
    addresses = []
    for i in subset_indices:
        addresses.append(train_addresses[i])
    return addresses



def historyPlot(history, name):
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    #     summarize history for accuracy
    plt.figure(np.random.randint(1,10))
    plt.subplot(221)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

#     summarize history for loss

    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.subplot(223)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model AUC')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')

    plt.subplot(224)
    plt.plot(history.history['mean_iou'])
    plt.plot(history.history['val_mean_iou'])
    plt.title('model IoU')
    plt.ylabel('IoU')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.tight_layout()
    plt.savefig("{}.png".format(name))


def create_data_subsets(addresses, size):
    """
    create random subsets
    """
    new_addresses = []
    indices = list(range(len(addresses)))
    random.shuffle(indices, random.random)
    indices = indices[:size]

    for index in indices:
        new_addresses.append(addresses[index])
    return new_addresses

def read_decathlon_data():
    """
    convert csv file to num[y array
    """

    with open('/home/tjvsonsbeek/decathlonData/decathlon_csv.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        decathlon_results = {}
        for row in reader:
            key = '{}_{}_{}_{}'.format(row[0], row[1], row[3], row[6])
            if key not in decathlon_results.keys():
                decathlon_results[key] = row[4]
    np.save('decathlon_results.npy', decathlon_results)
    csvFile.close()

def get_metalabels_decathlon():
    """
    convert excel file with decathlon results to numpy arrays containing the results
    """
    dataset_xlsx = ['braintumor', 'heart', 'liver', 'hippocampus', 'prostate', 'lung', 'pancreas', 'hepaticvessel', 'spleen', 'colon']
    dataset      = ['Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon']
    participants = ['BCVuniandes', 'beomheep', 'CerebriuDIKU', 'EdwardMa12593', 'ildoo', 'iorism82', 'isarasua', 'Isensee', 'jiafucang', 'lesswire1', 'lupin', 'oldrich.kodym', 'ORippler', 'phil666', 'rzchen_xmu', 'ubilearn', 'whale', '17111010008', 'allan.kim01']
    decathlon_results = np.load('decathlon_results.npy')
    decathlon_results = decathlon_results.item()
    print(decathlon_results)
    prev_id = 0
    for dataset_nr in range(len(dataset)):
        print('/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(dataset[dataset_nr]))
        dataset_ids = os.listdir('/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(dataset[dataset_nr]))
        dataset_ids.sort()
        if dataset_nr == 7:
            prev_id = 1000
        try:
            for im_nr in range(prev_id,len(dataset_ids)+prev_id):
                print(im_nr+1)
                meta_label = np.zeros(len(participants))
                for ptcpt_nr in range(len(participants)):


                    meta_label[ptcpt_nr] = decathlon_results['{}_{}_DCS_{}'.format(im_nr+1, dataset_xlsx[dataset_nr], participants[ptcpt_nr])]
                nr = dataset_ids[im_nr-prev_id][-10:-7]
                if nr[0] == '_':
                    nr = nr[1:]
                if not os.path.isdir("/home/tjvsonsbeek/featureExtractorUnet/metadata/{}".format(dataset[dataset_nr])):
                    os.mkdir("/home/tjvsonsbeek/featureExtractorUnet/metadata/{}".format(dataset[dataset_nr]))
                np.save("/home/tjvsonsbeek/featureExtractorUnet/metadata/{}/{}.npy".format(dataset[dataset_nr], nr), meta_label)
            prev_id += len(dataset_ids)
        except:
            prev_id = im_nr

task_specific_features = np.array([ [0, 0, 0.39, 1.0, 1],
                                    [1, 0, 0.10, 1.0, 0],
                                    [1, 1, 1.00, 0.7, 0],
                                    [1, 0, 1.00, 1.0, 0],
                                    [1, 0, 0.51, 1.0, 0],
                                    [0, 1, 0.06, 0.1, 1],
                                    [1, 1, 0.10, 0.4, 0],
                                    [0, 1, 0.13, 0.1, 1],
                                    [1, 1, 0.21, 0.4, 0],
                                    [1, 1, 0.07, 0.4, 0],
                                    [1, 1, 1.00 ,0.7, 0],
                                    [1, 1, 1.00 ,0.7, 0],
                                    [1, 0, 1.00 ,1.0, 1]])
