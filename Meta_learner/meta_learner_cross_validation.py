from sklearn.decomposition import PCA
from sklearn.manifold import MDS
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
np.random.seed(42)

def main():
    tasks_list=  ['Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon','Task11_CHAOSLiver', 'Task12_LITSLiver','Task13_ACDCHeart']

    participants = ['BCVuniandes', 'beomheep', 'CerebriuDIKU', 'EdwardMa12593', 'ildoo', 'iorism82', 'isarasua', 'Isensee', 'jiafucang', 'lesswire1', 'lupin', 'oldrich.kodym', 'ORippler', 'phil666', 'rzchen_xmu', 'ubilearn', 'whale', '17111010008', 'allan.kim01']
    nr_of_methods = len(participants)
    feature_extractors = ['STAT','VGG16','ResNet50','MobileNetV1']
    regression_method = ['SVR','DNN']
    best_frac = {'STAT': 0.37,'VGG16': 0.23,'ResNet50': 0.49,'MobileNetV1': 0.49}
    meta_subset_size = 20
    sample_size = 100
    visualization = False
    gt_labels = np.load('metadata/decathlon_avgstd_results.npy')

# load the meta_features and load_meta_labels
    meta_features = {}
    meta_labels   = {}
    for fe_id, fe in enumerate(feature_extractors):
        pred = np.zeros((len(feature_extractors), len(regression_methods), len(tasks_list[:10]),2,19))
        result = np.zeros((len(feature_extractors), len(regression_methods), len(tasks_list[:10]),3))
        result = np.zeros((len(feature_extractors), len(regression_methods), nr_of_methods,3))
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
        else:
            # compose the regression features and labels
            _, regression_features_raw, usable_filters = preprocess_metafeatures(meta_features, tasks_list[:10], sample_size)
            regression_labels   = preprocess_metalabels(meta_labels, tasks_list[:10], sample_size)

            regression_features_raw = add_task_specific_metafeatures(regression_features_raw, task_specific_features, tasks_list[:10], sample_size)

        regression_features_raw =  normalize_metafeatures(regression_features_raw)




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

                if do_feature_selection:
                    train_regression_features, features_to_keep = feature_selection(train_regression_features, train_regression_labels, frac)
                    test_regression_features = feature_selection_test_set(test_regression_features, features_to_keep)

                for reg_id, regression_method in enumerate(regression_methods):
                    for p in range(19):
                        if regression_method == 'DNN':
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
                        elif regression_method == 'SVR':
                            regression_model = SVR()
                            regression_model.fit(train_regression_features, train_regression_labels[:,p])
                            test_result = regression_model.predict(test_regression_features)
                        else:
                            print('Regression method not known')

                        pred[fe_id, reg_id, task_id,0,p] = np.mean(test_result)
                        pred[fe_id, reg_id, task_id,1,p] = np.std(test_result)
                # MAE
                result_method[fe_id, res_id, task_id,0] = mean_absolute_error(gt_labels[task_id, 0, :],pred[task_id, 0, :])
                # Spearman task rank
                temp_gt, temp_pred = zip(*sorted(zip(gt_labels[task_id,0,:], pred)))
                results_per_mthd[fe_id, reg_id, task_id, c_id, 3] = np.round(spearmanr(temp_gt, temp_pred)[0],2)
                result_method[fe_id, res_id, task_id,1] = np.std(gt_labels[task_id, 0, :],pred[task_id, 0, :])
                result_method[fe_id, res_id, task_id,1] = mean_absolute_error(gt_labels[task_id, 0, :],pred[task_id, 0, :])
        # print(np.mean(res))
        result[frac_id,:,:,:] = pred
        print('Decathlon')



if __name__ == '__main__':
    main()
