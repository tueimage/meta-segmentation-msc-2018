# from meta_network import meta_learner
from data import Data, MetaData
# from utils import subset_index_to_address
# from utils import meta_pred_generator, historyPlot, create_data_subsets, dice_coef_loss, auc, mean_iou, dice_coef
from visualization import visualize_meta_labels, visualize_meta_features, visualize_regression_result, visualize_features_tSNE, visualize_features_MDS, visualize_confusion_matrix, visualize_features_PCA

from keras.optimizers import Adam
from MetaFeatureExtraction import MetaFeatureExtraction
from tqdm import tqdm
import random
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.svm import SVR
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():
    tasks_list=  ['Task01_BrainTumour', 'Task02_Heart', 'Task03_Liver', 'Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon']
    dataset_xlsx = ['braintumor', 'heart', 'liver', 'hippocampus', 'prostate', 'lung', 'pancreas', 'hepaticvessel', 'spleen', 'colon']
    meta_feature_names = ['nr of instances', 'mean pixel value', 'std pixel value', 'coefficient of variation', 'mean correlation coefficient', 'mean skewness', 'mean kurtosis', 'nomalized class entropy', 'mean normalized feature entropy', 'mean mutual inf', 'max mutual inf', 'equivalent nr of features', 'noise signal ratio']
    participants = ['BCVuniandes', 'beomheep', 'CerebriuDIKU', 'EdwardMa12593', 'ildoo', 'iorism82', 'isarasua', 'Isensee', 'jiafucang', 'lesswire1', 'lupin', 'oldrich.kodym', 'ORippler', 'phil666', 'rzchen_xmu', 'ubilearn', 'whale', '17111010008', 'allan.kim01']

    meta_subset_size = 8
    nr_of_subsets = 100
## get the meta_features and meta_labels
    # for task in tasks_list:
    #     features = np.zeros((nr_of_subsets, 13))
    #     labels = np.zeros((nr_of_subsets, len(participants)))
    #     print(task)
    #     for subset in tqdm(range(nr_of_subsets)):
    #         print(subset)
    #         m = meta_regressor(task)
    #         m.gather_addresses()
    #         m.gather_meta_labels()
    #         m.gather_meta_features()
    #         features[subset,:] = m.meta_features
    #         labels[subset,:]   = m.meta_labels
    #     np.save('meta_regressor_labels_{}.npy'.format(task), labels)
    #     np.save('meta_regressor_features_{}.npy'.format(task), features)
## load the meta_features and load_meta_labels
    meta_features = {}
    meta_labels   = {}
    meta_test_features = {}
    meta_test_labels   = {}
    for task in tasks_list:
        m = MetaFeatureExtraction(task)
        m.load_meta_labels()
        m.load_meta_features()

        meta_labels[task] = m.meta_labels
        meta_features[task] = m.meta_features
# visualise the meta_features and meta_labels
    # visualize_meta_features(meta_features, tasks_list, meta_feature_names)
    # visualize_meta_labels(meta_labels, tasks_list, participants)
# do the regression
# compose the regression features and labels
    regression_features = np.zeros((10*nr_of_subsets, 13))
    regression_labels   = np.zeros((10*nr_of_subsets, 19))

    for task in range(len(tasks_list)):#len(tasks_list)):
        for nr in range(nr_of_subsets):
            regression_features[task*nr_of_subsets+nr,:] = meta_features[tasks_list[task]][nr,:]
            regression_labels[task*nr_of_subsets+nr,:] = meta_labels[tasks_list[task]][nr,:]
    for i in range(regression_features.shape[1]):
        std = np.std(regression_features[:,i])
        if std == 0:
            std = 1
        regression_features[:,i] = (regression_features[:,i]-np.mean(regression_features[:,i]))/std
    # do a 10-fold k regression
    # visualize_confusion_matrix(regression_labels, dataset_xlsx, 'labels', 1)
    # visualize_confusion_matrix(regression_features, dataset_xlsx, 'features', 1)
    #
    # visualize_features_tSNE(regression_labels, dataset_xlsx, 'labels_together', list(range(10)), 1)
    # visualize_features_tSNE(regression_features, dataset_xlsx, 'features_together', list(range(10)), 1)
    #
    # visualize_features_MDS(regression_labels, dataset_xlsx, 'labels_together', list(range(10)), 1)
    # visualize_features_MDS(regression_features, dataset_xlsx, 'features_together', list(range(10)), 1)
    #
    # visualize_features_PCA(regression_labels, dataset_xlsx, 'labels_together', list(range(10)), 1)
    # visualize_features_PCA(regression_features, dataset_xlsx, 'features_together', list(range(10)), 1)

    for kfold in range(10):
        test_set = random.sample(list(range(10)),3)
        train_set = list(set(range(10))-set(test_set))
        print(test_set)
        test_set.sort()
        train_set.sort()
        train_regression_features = np.zeros((7*nr_of_subsets, 13))
        train_regression_labels   = np.zeros((7*nr_of_subsets, 19))

        test_regression_features = np.zeros((3*nr_of_subsets, 13))
        test_regression_labels   = np.zeros((3*nr_of_subsets, 19))
        for i in range(len(train_set)):
            task = train_set[i]
            train_regression_features[i*100:i*100+100,:] = regression_features[task*100:task*100+100,:]
            train_regression_labels[i*100:i*100+100,:] = regression_labels[task*100:task*100+100,:]

        for i in range(len(test_set)):
            task = test_set[i]
            test_regression_features[i*100:i*100+100,:] = regression_features[task*100:task*100+100,:]
            test_regression_labels[i*100:i*100+100,:] = regression_labels[task*100:task*100+100,:]





        train_regression_features_noshuffle = train_regression_features.copy()
        train_regression_labels_noshuffle = train_regression_labels.copy()
        np.random.seed(42)
        np.random.shuffle(train_regression_features)
        np.random.seed(42)
        np.random.shuffle(train_regression_labels)

        # np.random.seed(51)
        # np.random.shuffle(test_regression_features)
        # np.random.seed(51)
        # np.random.shuffle(test_regression_labels)
        # np.random.seed(51)
        # np.random.shuffle(test_regression_labels_dataset_nr)

        train_isensee = train_regression_labels[:,7]
        test_isensee = test_regression_labels[:,7]

        svr_model = SVR(kernel='poly', C=100, gamma='auto', degree=1, epsilon=.1, coef0=1, tol=1e-7,verbose = False)
        #svr_model = SVR(kernel='rbf', C=100, gamma='auto', degree=1, epsilon=.1, coef0=1, verbose = True)
        svr_model.fit(train_regression_features, train_isensee)

        train_result = svr_model.predict(train_regression_features)
        train_result_noshuffle = svr_model.predict(train_regression_features_noshuffle)
        train_score = svr_model.score(train_regression_features, train_isensee)

        test_result = svr_model.predict(test_regression_features)
        test_score = svr_model.score(test_regression_features, test_isensee)


        print("train score: {}".format(train_score))
        print("test  score: {}".format(test_score))
        # coefficients = predictor.coef_
        # pred_labels = isensee[300:]
        # for row in range(X.shape[0]):
        #     print('{}  {}  {}   {}'.format(y[row],y_rbf[row], y_lin[row], y_poly[row]))
        # from utils import visualize_regression_result
        print(visualize_regression_result)
        # visualize_regression_result(test_result, test_isensee, dataset_xlsx,test_set,  nr_of_subsets, 'test', kfold)
        # visualize_features_tSNE(train_result_noshuffle, dataset_xlsx, 'train', train_set, kfold)
        # visualize_features_MDS(train_result_noshuffle, dataset_xlsx, 'train', train_set, kfold)
        # visualize_features_PCA(train_result_noshuffle, dataset_xlsx, 'train', train_set, kfold)

        # visualize_regression_result(train_result, train_isensee, train_regression_labels_dataset_nr, 'train', kfold)
        # print(test_result)
        # print(test_isensee.shape)
        a=np.resize(np.array(test_result),(300,1))
        b=np.resize(np.array(test_isensee),(300,1))
        print(a.shape)
        print(b.shape)
        a=np.concatenate([a,b],axis =1)

        visualize_features_tSNE(a, dataset_xlsx, 'test', test_set, kfold)
        visualize_features_MDS(a, dataset_xlsx, 'test', test_set, kfold)
        visualize_features_PCA(a, dataset_xlsx, 'test', test_set, kfold)

        visualize_regression_result(test_result, test_isensee, dataset_xlsx, 'test', test_set, kfold)

            # print(X[row,:])
        # for i in range(outcome.shape[0]):
        #     print(regression_labels[i,:])
        #     print(outcome[i,:])
        # ptcp, outcm = zip(*sorted(zip(list(outcome[i, :]), participants)))
        # for i in range(len(ptcp)):
        #     print("{}: {}   {}".format(i+1, ptcp[-i-1], outcm[-i-1]))
if __name__ == '__main__':
    main()
