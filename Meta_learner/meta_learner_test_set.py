from sklearn.decomposition import PCA
from sklearn.manifold import MDS
# from meta_network import meta_learner

# from utils import subset_index_to_address
from utils import preprocess_metalabels, preprocess_metafeatures, preprocess_metafeatures_test_set, regression_feature_selection, largest_indices
from visualization import visualize_meta_labels, visualize_meta_features, visualize_regression_result, visualize_features_tSNE, visualize_features_MDS, visualize_confusion_matrix, visualize_features_PCA, visualize_overall_result
from keras.optimizers import Adam
from MetaFeatureExtraction import MetaFeatureExtraction
from tqdm import tqdm
import random
import math
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense,Input,Dropout
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    normalized_metafeatures = np.zeros_like(metafeatures)
    for c in range(metafeatures.shape[1]):
        std = np.std(metafeatures[:,c])
        if std == 0:
            std = 1
        normalized_metafeatures[:,c] = (metafeatures[:,c]-np.mean(metafeatures[:,c]))/std

    return normalized_metafeatures

def simple_model(nr_of_features = 1):
    model = Sequential()
    model.add(Dense(1024, input_dim=3577, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(19, activation='relu'))
    model.summary()
    return model
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

def main():
    tasks_list=  ['Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon','Task11_CHAOSLiver', 'Task12_LITSLiver','Task13_ACDCHeart']
    nr_of_methods = 19
    feature_extractors = ['STAT']#,'VGG16','ResNet50','MobileNetV1']
    regression_method = 'singleDeep'
    best_frac = {'STAT': 0.37,'VGG16': 0.23,'ResNet50': 0.49,'MobileNetV1': 0.49}
    meta_subset_size = 20
    sample_size = 100
    visualization = False
# load the meta_features and load_meta_labels
    meta_features = {}
    meta_labels   = {}
    min_train_sets = 1
    max_train_sets = 9


    for fe in feature_extractors:
        print(fe)
        for task_id, task in enumerate(tasks_list):

            m = MetaFeatureExtraction(task, meta_subset_size, fe)
            m.load_meta_labels()
            m.load_meta_features()
            if fe != 'STAT': m.sum_and_log_meta_features()
            if task_id < 10 : meta_labels[task] = m.meta_labels
            meta_features[task] = m.meta_features

        if fe == 'STAT':
            regression_features_raw = np.zeros((len(tasks_list[:10])*sample_size, 38))
            regression_labels   = np.zeros((len(tasks_list[:10])*sample_size, 19))
            regression_features_raw_test_set = np.zeros((len(tasks_list[10:]), 38))

            for task_id, task in enumerate(tasks_list[:10]):
                    for nr in range(sample_size):
                        if meta_subset_size == 20:
                            regression_features_raw[task_id*sample_size+nr,:33] = meta_features[task][nr,:]
                            regression_features_raw[task_id*sample_size+nr,33:] = task_specific_features[task_id,:]
                            regression_labels[task_id*sample_size+nr,:] = meta_labels[task][nr,:]
                        else:
                            regression_features_raw[task_id*sample_size+nr,:33] = np.sum(meta_features[task][nr,:],axis=0)/meta_features[task][nr,:].shape[0]
                            regression_features_raw[task_id*sample_size+nr,33:] = task_specific_features[task_id,:]
                            regression_labels[task_id*sample_size+nr,:] = np.sum(meta_labels[task][nr,:], axis=0)/meta_features[task][nr,:].shape[0]
            for task_id, task in enumerate(tasks_list[10:]):
                if task == 'Task11_CHAOSLiver':
                    for nr in range(1):
                        regression_features_raw_test_set[task_id+nr,:33] = np.sum(meta_features[task][nr,:],axis=0)/5
                        regression_features_raw_test_set[task_id+nr,33:] = task_specific_features[task_id+10,:]
                        print(np.sum(meta_features[task][nr,:],axis=0)/5)
                else:
                    for nr in range(1):
                        regression_features_raw_test_set[task_id+nr,:33] = np.sum(meta_features[task][nr,:],axis=0)/20
                        regression_features_raw_test_set[task_id+nr,33:] = task_specific_features[task_id+10,:]
                        print(np.sum(meta_features[task][nr,:],axis=0)/20)
        else:
            # compose the regression features and labels
            _, regression_features_raw, usable_filters = preprocess_metafeatures(meta_features, tasks_list[:10], sample_size)
            _, regression_features_raw_test_set = preprocess_metafeatures_test_set(meta_features, tasks_list, 1, usable_filters)
            regression_labels   = preprocess_metalabels(meta_labels, tasks_list[:10], sample_size)
            print(np.count_nonzero(regression_features_raw_test_set[0,:]))
            print(np.count_nonzero(regression_features_raw_test_set[1,:]))
            print(np.count_nonzero(regression_features_raw_test_set[2,:]))
            regression_features_raw = add_task_specific_metafeatures(regression_features_raw, task_specific_features, tasks_list[:10], sample_size)
            regression_features_raw_test_set = add_task_specific_metafeatures(regression_features_raw_test_set, task_specific_features[10:], tasks_list[10:], 1)

        regression_features_raw2 =  normalize_metafeatures(regression_features_raw)
        regression_features_raw, regression_features_raw_test_set =  joint_normalize_metafeatures(regression_features_raw, regression_features_raw_test_set)

        # regression_features_raw_test_set =  normalize_metafeatures(regression_features_raw_test_set)
        result= np.zeros((2,10,2,19))
        for frac_id, frc in enumerate(['once']):
            frac = best_frac[fe]

            # print(frac)
            for do_feature_selection in [True, False]:
                if do_feature_selection:
                    # print('Feature selection')
                    regression_features, features_to_keep = feature_selection(regression_features_raw, regression_labels, frac)
                    regression_features_test_set = feature_selection_test_set(regression_features_raw_test_set, features_to_keep)
                    regression_features2, features_to_keep2 = feature_selection(regression_features_raw2, regression_labels, frac)
                else:
                    frac_id+=1
                    regression_features = regression_features_raw
                    regression_features_test_set = regression_features_raw_test_set
                    regression_features2 = regression_features_raw2
                    # print('Nofeature selection')
                print(regression_features_test_set.shape)
                pred = np.zeros((10,2,19))
                res = np.zeros((10))
                gt_labels = np.load('metadata/decathlon_avgstd_results.npy')
                for task_id, task in tqdm(enumerate(tasks_list[:10])):
                        nr_of_metafeatures = regression_features2.shape[1]
                        test_set = [task_id]
                        train_set = list(set(range(10))-set(test_set))


                        test_set.sort()
                        train_set.sort()
                        train_regression_features = np.zeros((9 * sample_size, nr_of_metafeatures))
                        train_regression_labels   = np.zeros((9 * sample_size, nr_of_methods))

                        test_regression_features = np.zeros((1 * sample_size, nr_of_metafeatures))
                        test_regression_labels   = np.zeros((1 * sample_size, nr_of_methods))


                        for i, task in enumerate(train_set):
                            train_regression_features[i*sample_size:(i+1)*sample_size,:] = regression_features[task*sample_size:(task+1)*sample_size,:]
                            train_regression_labels[i*sample_size:(i+1)*sample_size,:] = regression_labels[task*sample_size:(task+1)*sample_size,:]


                        for i, task in enumerate(test_set):
                            test_regression_features[i*sample_size:(i+1)*sample_size,:] = regression_features[task*sample_size:(task+1)*sample_size,:]
                            test_regression_labels[i*sample_size:(i+1)*sample_size,:] = regression_labels[task*sample_size:(task+1)*sample_size,:]
                        for p in range(19):

                            K.clear_session()
                            model = simple_single_model(train_regression_features.shape[1])
                            np.random.seed(42)
                            np.random.shuffle(train_regression_features)
                            np.random.seed(42)
                            np.random.shuffle(train_regression_labels)
                            model = reset_weights(model)
                            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
                            model.fit(train_regression_features, train_regression_labels[:,p], epochs=10, verbose=0, shuffle=True)
                            test_result = model.predict(test_regression_features)
                #
                #
                #         #
                #         for p in range(19):
                #             regression_model = SVR()
                #
                #             regression_model.fit(train_regression_features, train_regression_labels[:,p])
                #             test_result = regression_model.predict(test_regression_features)
                #
                            pred[task_id,0,p] = np.mean(test_result)
                            pred[task_id,1,p] = np.std(test_result)
                #
                #         res[task_id] = mean_absolute_error(gt_labels[task_id, 0, :],pred[task_id, 0, :])
                # print(np.mean(res))
                result[frac_id,:,:,:] = pred
                print('Decarhlon')

                for task_id, task in enumerate(tasks_list[10:]):

                    # K.clear_session()
                    # model = simple_single_model(regression_features.shape[1])
                    # np.random.seed(42)
                    # np.random.shuffle(regression_features)
                    # np.random.seed(42)
                    # np.random.shuffle(regression_labels)
                    # model = reset_weights(model)
                    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
                    # model.fit(regression_features, regression_labels[:,7], epochs=10, verbose=0, shuffle=True)
                    # test_result = model.predict(test_regression_features)
                    regression_model = SVR()


                    regression_model.fit(regression_features, regression_labels[:,7])
                    test_result = regression_model.predict(regression_features_test_set[task_id:(task_id+1),:])
                    # test_result = model.predict(regression_features_test_set[task_id:(task_id+1),:])
                    print(task)
                    print(np.mean(test_result))
        np.save('dl_results/frac_dl_{}_{}.npy'.format(fe, regression_method),result)


if __name__ == '__main__':
    main()
