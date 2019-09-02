# from meta_network import meta_learner
from data import Data, MetaData
# from utils import subset_index_to_address
from utils import preprocess_metalabels, preprocess_metafeatures, preprocess_metafeatures_test_set
from visualization import visualize_meta_labels, visualize_meta_features, visualize_regression_result, visualize_features_tSNE, visualize_features_MDS, visualize_confusion_matrix, visualize_features_PCA, visualize_overall_result
from keras.optimizers import Adam
from MetaFeatureExtraction import MetaFeatureExtraction
from tqdm import tqdm
import random
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Input
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def simple_model(nr_of_features = 1):
    model = Sequential()
    model.add(Dense(1024, input_dim=3577, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(19, activation='relu'))
    model.summary()
    return model

task_specific_features = np.array([[0, 0, 0.39, 1, 1],
                    [1, 0, 0.10, 1.0, 0],
                    [1, 1, 1.00, 0.7, 0],
                    [1, 0, 1.00, 1.0, 0],
                    [1, 0, 0.51, 1.0, 0],
                    [0, 1, 0.06, 0.1, 1],
                    [1, 1, 0.10, 0.4, 0],
                    [0, 1, 0.13, 0.1, 1],
                    [1, 1, 0.21, 0.4, 0],
                    [1, 1, 0.07, 0.4, 0],
                    [1, 1, 1.00 ,1.0 ,0]])

def main():
    tasks_list=  ['Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon']
    dataset_xlsx = ['Braintumor', 'Heart', 'Liver', 'Hippocampus', 'Prostate', 'Lung', 'Pancreas', 'Hepaticvessel', 'Spleen', 'Colon']
    meta_feature_names = ['nr of instances', 'mean pixel value', 'std pixel value', 'coefficient of variation', 'mean correlation coefficient', 'mean skewness', 'mean kurtosis', 'nomalized class entropy', 'mean normalized feature entropy', 'mean mutual inf', 'max mutual inf', 'equivalent nr of features', 'noise signal ratio']
    participants = ['BCVuniandes', 'beomheep', 'CerebriuDIKU', 'EdwardMa12593', 'ildoo', 'iorism82', 'isarasua', 'Isensee', 'jiafucang', 'lesswire1', 'lupin', 'oldrich.kodym', 'ORippler', 'phil666', 'rzchen_xmu', 'ubilearn', 'whale', '17111010008', 'allan.kim01']
    metalearning = False
    multi_learning = False
    dl_learning = False
    meta_subset_size = 10
    nr_of_subsets = 100
    nr_of_methods = len(participants)
    feature_extractors = ['STAT','VGG16','ResNet50','MobileNetV1']
    meta_subset_sizes = [20]
    meta_sample_sizes = [100]
    visualization = True
# load the meta_features and load_meta_labels
    meta_features = {}
    meta_labels   = {}
    for fe in feature_extractors:
        for ms_id, meta_subset_size in enumerate(meta_subset_sizes):
            for ss_id, sample_size in enumerate(meta_sample_sizes):
                for task in tasks_list:
                    m = MetaFeatureExtraction(task, meta_subset_size, fe)
                    m.load_meta_labels()
                    m.load_meta_features()
                    if fe != 'STAT': m.sum_and_log_meta_features()
                    meta_labels[task] = m.meta_labels
                    meta_features[task] = m.meta_features
                if fe == 'STAT':
                    regression_features = np.zeros((10*sample_size, 38))
                    regression_labels   = np.zeros((10*sample_size, 19))

                    regression_features_chaos = np.zeros((1,38))
                    regression_labels_chaos   = np.zeros(1)

                    for task_id, task in enumerate(tasks_list):
                        if task == 'Task11_CHAOSLiver':
                            regression_features_chaos[0,:33] = np.sum(meta_features[task][0,:],axis=0)/5
                            regression_features_chaos[0,33:] = task_specific_features[task_id,:]
                            regression_labels_chaos = np.sum(meta_labels[task][0,:],axis=0)/5
                            print(regression_features_chaos)
                            print(regression_labels_chaos)
                        else:
                            for nr in range(sample_size):
                                if meta_subset_size == 20:
                                    regression_features[task_id*sample_size+nr,:33] = meta_features[task][nr,:]#np.sum(meta_features[task][nr,:],axis=0)/meta_subset_size
                                    regression_features[task_id*sample_size+nr,33:] = task_specific_features[task_id,:]
                                    regression_labels[task_id*sample_size+nr,:] = meta_labels[task][nr,:]#np.sum(meta_labels[task][nr,:], axis=0)/meta_subset_size
                                else:
                                    regression_features[task_id*sample_size+nr,:33] = np.sum(meta_features[task][nr,:],axis=0)/meta_subset_size
                                    regression_features[task_id*sample_size+nr,33:] = task_specific_features[task_id,:]
                                    regression_labels[task_id*sample_size+nr,:] = np.sum(meta_labels[task][nr,:], axis=0)/meta_subset_size

                else:
                    # compose the regression features and labels
                    _,regression_features,usable_filters = preprocess_metafeatures(meta_features, tasks_list[:10], sample_size)
                    regression_labels   = preprocess_metalabels(meta_labels, tasks_list[:10], sample_size)
                    regression_features[regression_features==-np.inf] = 0

                    # compose the regression features and labels CHAOS
                    _,regression_features_chaos = preprocess_metafeatures_test_set(meta_features, tasks_list, sample_size,usable_filters)
                    regression_labels_chaos   = np.sum(meta_labels[task][0,:],axis=0)/5
                    regression_features_chaos[regression_features_chaos==-np.inf] = 0

                for i in tqdm(range(regression_features.shape[1])):
                    std = np.std(regression_features[:,i])
                    if std == 0:
                        std = 1
                    regression_features[:,i] = (regression_features[:,i]-np.mean(regression_features[:,i]))/std

                for i in tqdm(range(regression_features_chaos.shape[1])):
                    std = np.std(regression_features_chaos[:,i])
                    if std == 0:
                        std = 1
                    regression_features_chaos[:,i] = (regression_features_chaos[:,i]-np.mean(regression_features_chaos[:,i]))/std

                if visualization:
                    visualize_confusion_matrix(regression_features, dataset_xlsx, fe)
                    visualize_features_tSNE(regression_features, dataset_xlsx, list(range(10)), fe+'nolegend')
                    visualize_features_MDS(regression_features, dataset_xlsx, list(range(10)), fe+'nolegend')
                    visualize_features_PCA(regression_features, dataset_xlsx, list(range(10)), fe+'nolegend')
                if metalearning:
                    nr_of_metafeatures = regression_features.shape[1]

                    pred_labels = np.zeros((10,2,19))
                    nr_of_times_in_test = np.zeros(10)
                    chaos_test_result = []
                    # do a k-fold regression until every dataset has been in text three times
                    # random.seed(9001)
                    print(np.count_nonzero(regression_features))
                    print(np.count_nonzero(regression_labels))
                    while np.min(nr_of_times_in_test)<5:
                        test_set = random.sample(list(range(10)),3)
                        train_set = list(set(range(10))-set(test_set))
                        for s in test_set:
                            nr_of_times_in_test[s]+=1
                        test_set.sort()
                        train_set.sort()
                        train_regression_features = np.zeros((7*sample_size, nr_of_metafeatures))
                        train_regression_labels   = np.zeros((7*sample_size, nr_of_methods))

                        test_regression_features = np.zeros((3*sample_size, nr_of_metafeatures))
                        test_regression_labels   = np.zeros((3*sample_size, nr_of_methods))
                        for i, task in enumerate(train_set):
                            train_regression_features[i*sample_size:(i+1)*sample_size,:] = regression_features[task*sample_size:(task+1)*sample_size,:]
                            train_regression_labels[i*sample_size:(i+1)*sample_size,:] = regression_labels[task*sample_size:(task+1)*sample_size,:]

                        for i, task in enumerate(test_set):
                            test_regression_features[i*sample_size:(i+1)*sample_size,:] = regression_features[task*sample_size:(task+1)*sample_size,:]
                            test_regression_labels[i*sample_size:(i+1)*sample_size,:] = regression_labels[task*sample_size:(task+1)*sample_size,:]
                        if multi_learning:
                            multiple_regression_model = MultiOutputRegressor(SVR(kernel='poly', C=100, gamma='auto', degree=2, epsilon=.1, coef0=1, tol=1e-7,verbose = True), n_jobs=-1)
                            multiple_regression_model.fit(train_regression_features, train_regression_labels)

                            train_result = multiple_regression_model.predict(train_regression_features)
                            train_score = multiple_regression_model.score(train_regression_features, train_regression_labels)

                            test_result = multiple_regression_model.predict(test_regression_features)
                            test_score = multiple_regression_model.score(test_regression_features, test_regression_labels)


                            for s_id, s in enumerate(test_set):
                                if nr_of_times_in_test[s]<=5:
                                    for p in range(nr_of_methods):
                                        pred_labels[s,0,p] += np.mean(test_result[sample_size*s_id:sample_size*(s_id+1),p])/5
                                        pred_labels[s,1,p] += np.std(test_result[sample_size*s_id:sample_size*(s_id+1),p])/5
                            print(nr_of_times_in_test)
                            if visualization:
                                visualize_regression_result(test_result, test_isensee, dataset_xlsx, fe, test_set, meta_subset_size, participants[i])
                                visualize_regression_result(train_result, train_isensee, dataset_xlsx, fe, train_set, meta_subset_size, participants[i])
                        elif dl_learning:
                            # train_regression_features = np.expand_dims(train_regression_features,-1)
                            np.random.seed(42)
                            np.random.shuffle(train_regression_features)
                            np.random.seed(42)
                            np.random.shuffle(train_regression_labels)
                            # test_regression_features = np.expand_dims(test_regression_features,-1)
                            for p in range(nr_of_methods):
                                model = simple_model(train_regression_features.shape[1])
                                model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
                                model.fit(train_regression_features, train_regression_labels[:,:], epochs=1000, verbose=1, shuffle=True)
                                test_result = model.predict(test_regression_features)
                                for s_id, s in enumerate(test_set):
                                    if nr_of_times_in_test[s]<=5:
                                        pred_labels[s,0,p] += np.mean(test_result[sample_size*s_id:sample_size*(s_id+1),0])/5
                                        pred_labels[s,1,p] += np.std(test_result[sample_size*s_id:sample_size*(s_id+1),0])/5
                        else:
                            for i in range(19):
                                svr_model = SVR(kernel='poly', C=100, gamma='auto', degree=2, epsilon=.1, coef0=1, tol=1e-7,verbose = True)
                                svr_model.fit(train_regression_features, train_regression_labels[:,i])

                                train_result = svr_model.predict(train_regression_features)
                                train_score = svr_model.score(train_regression_features, train_regression_labels[:,i])

                                test_result = svr_model.predict(test_regression_features)
                                test_score = svr_model.score(test_regression_features, test_regression_labels[:,i])
                                if i==7: #This is the Isensee nnunet
                                    chaos_result = svr_model.predict(regression_features_chaos)
                                    chaos_test_result.append(chaos_result[0])

                                for s_id, s in enumerate(test_set):
                                    if nr_of_times_in_test[s]<=5:
                                        pred_labels[s,0,i] += np.mean(test_result[sample_size*s_id:sample_size*(s_id+1)])/5
                                        pred_labels[s,1,i] += np.std(test_result[sample_size*s_id:sample_size*(s_id+1)])/5
                                if visualization:
                                    visualize_regression_result(test_result, test_isensee, dataset_xlsx, fe, test_set, meta_subset_size, participants[i])
                                    visualize_regression_result(train_result, train_isensee, dataset_xlsx, fe, train_set, meta_subset_size, participants[i])

                    np.save('results/pred_labels_{}_{}_{}.npy'.format(sample_size, meta_subset_size, fe), pred_labels)
                    np.save('results/pred_chaos_{}_{}_{}.npy'.format(sample_size, meta_subset_size, fe), chaos_test_result)
if __name__ == '__main__':
    main()
