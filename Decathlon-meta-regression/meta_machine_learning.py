# from meta_network import meta_learner
from data import Data, MetaData
# from utils import subset_index_to_address
# from utils import meta_pred_generator, historyPlot, create_data_subsets, dice_coef_loss, auc, mean_iou, dice_coef
from utils import visualize_meta_labels, visualize_meta_features, visualize_regression_one_method
from keras.optimizers import Adam
from meta_network import meta_learner, meta_regressor
from networks import EncoderDecoderNetwork
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
    tasks_list=  [ 'Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus','Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas','Task08_HepaticVessel','Task09_Spleen','Task10_Colon']
#,
    meta_feature_names = ['nr of instances', 'nr of features', 'nr of target concept values', 'dataset dimensionality', 'mean correlation coefficient', 'mean skewness', 'mean kurtosis', 'nomalized class entropy', 'mean normalized feature entropy', 'mean mutual inf', 'max mutual inf', 'equivalent nr of features', 'noise signal ratio']
    participants = ['BCVuniandes', 'beomheep', 'CerebriuDIKU', 'EdwardMa12593', 'ildoo', 'iorism82', 'isarasua', 'Isensee', 'jiafucang', 'lesswire1', 'lupin', 'oldrich.kodym', 'ORippler', 'phil666', 'rzchen_xmu', 'ubilearn', 'whale', '17111010008', 'allan.kim01']
    meta_features_names = ['']
    meta_subset_size = 8
    nr_of_subsets = 100
    n = 350
## get the meta_features and meta_labels
    for task in tasks_list:
        features = np.zeros((nr_of_subsets, 13))
        labels = np.zeros((nr_of_subsets, len(participants)))
        print(task)
        for subset in tqdm(range(nr_of_subsets)):
            print(subset)
            m = meta_regressor(task)
            m.gather_addresses()
            m.gather_meta_labels()
            m.gather_meta_features()
            features[subset,:] = m.meta_features
            labels[subset,:]   = m.meta_labels
        np.save('meta_regressor_labels_{}.npy'.format(task), labels)
        np.save('meta_regressor_features_{}.npy'.format(task), features)
## load the meta_features and load_meta_labels
    # meta_features = {}
    # meta_labels   = {}
    # meta_test_features = {}
    # meta_test_labels   = {}
    # for task in tasks_list:
    #     m = meta_regressor(task)
    #     m.load_meta_labels()
    #     m.load_meta_features()
    #     if task == 'Task01_BrainTumour' or task =='Task02_Heart' or task =='Task03_Liver':
    #         meta_test_labels[task] = m.meta_labels
    #         meta_test_features[task] = m.meta_features
    #     else:
    #         meta_labels[task] = m.meta_labels
    #         meta_features[task] = m.meta_features
## visualise the meta_features and meta_labels
    # visualize_meta_features(meta_features, tasks_list, meta_feature_names)
    # visualize_meta_labels(meta_labels, tasks_list, participants)
## do the regression
## compose the regression features and labels
    # regression_features = np.zeros((n, 13))
    # regression_labels   = np.zeros((n, 19))
    #
    # regression_test_features = np.zeros((150, 13))
    # regression_test_labels   = np.zeros((150, 19))
    #
    # for task in range(7):#len(tasks_list)):
    #     for nr in range(nr_of_subsets):
    #         regression_features[task*nr_of_subsets+nr,:] = meta_features[nr,:]
    #         regression_labels[task*nr_of_subsets+nr,:] = meta_labels[nr,:]
    # for task in range(3):
    #     for nr in range(nr_of_subsets):
    #         regression_test_features[task*nr_of_subsets+nr,:] = meta_test_features[nr,:]
    #         regression_test_labels[task*nr_of_subsets+nr,:] = meta_test_labels[nr,:]
    #
    # isensee = regression_labels[:,7]
    #
    # isensee_test = regression_test_labels[:,7]
    # for i in range(regression_features.shape[1]):
    #     std = np.std(regression_features[:,i])
    #     if std == 0:
    #         std = 1
    #     regression_features[:,i] = (regression_features[:,i]-np.mean(regression_features[:,i]))/std
    #
    # for i in range(regression_test_features.shape[1]):
    #     std = np.std(regression_test_features[:,i])
    #     if std == 0:
    #         std = 1
    #     regression_test_features[:,i] = (regression_test_features[:,i]-np.mean(regression_test_features[:,i]))/std
    #
    # print(regression_features)
    # # for label in range(len(isensee)):
    # #     isensee[label] = int(isensee[label]*100)
    # np.random.seed(42)
    # np.random.shuffle(regression_features)
    # np.random.seed(42)
    # np.random.shuffle(regression_labels)
    #
    # np.random.seed(42)
    # np.random.shuffle(regression_test_features)
    # np.random.seed(42)
    # np.random.shuffle(regression_test_labels)
    # X = regression_features[:300,:]
    # X2 = regression_test_features[:25,:]
    # y = isensee[:300]
    # y2 = isensee_test[:25]
    # predictor = LinearRegression(n_jobs=-1)
    # predictor.fit(X=regression_features[:300,:], y=regression_labels[:300,:])
    # outcome = predictor.predict(X = regression_features[:300,:])
    # # predictor = SVC(kernel = 'poly',degree = 6)
    # # predictor.fit(regression_features[:300,:], isensee[:300])
    # # outcome = predictor.predict(regression_features[300:,:])
    # svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1, verbose = True)
    # svr_lin = SVR(kernel='linear', C=100, gamma='auto', verbose = True)
    # svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
    #                coef0=1, verbose = True)
    # y_rbf_model = svr_rbf.fit(X, y)
    # y_lin_model = svr_lin.fit(X, y)
    # y_poly_model = svr_poly.fit(X, y)
    # y_rbf = y_rbf_model.predict(X)
    # y_lin = y_lin_model.predict(X)
    # y_poly = y_poly_model.predict(X)
    # y_rbf_test = y_rbf_model.predict(X2)
    # y_lin_test = y_lin_model.predict(X2)
    # y_poly_test = y_poly_model.predict(X2)
    # y_rbf_score_train = y_rbf_model.score(X, y)
    # y_lin_score_train = y_lin_model.score(X, y)
    # y_poly_score_train = y_poly_model.score(X, y)
    # y_rbf_score_test = y_rbf_model.score(X2, y2)
    # y_lin_score_test = y_lin_model.score(X2, y2)
    # y_poly_score_test = y_poly_model.score(X2, y2)
    #
    # print("rbf train: {}".format(y_rbf_score_train))
    # print("rbf test : {}".format(y_rbf_score_test ))
    # print("lin train: {}".format(y_lin_score_train))
    # print("lin test : {}".format(y_lin_score_test ))
    # print("poly train: {}".format(y_poly_score_train))
    # print("poly test : {}".format(y_poly_score_test ))
    # # coefficients = predictor.coef_
    # # pred_labels = isensee[300:]
    # for row in range(X.shape[0]):
    #     print('{}  {}  {}   {}'.format(y[row],y_rbf[row], y_lin[row], y_poly[row]))
    # visualize_regression_one_method(y[:25], [y_rbf[:25], y_poly[:25]], ['y_rbf', 'y_poly'])

        # print(X[row,:])
    # for i in range(outcome.shape[0]):
    #     print(regression_labels[i,:])
    #     print(outcome[i,:])
    # ptcp, outcm = zip(*sorted(zip(list(outcome[i, :]), participants)))
    # for i in range(len(ptcp)):
    #     print("{}: {}   {}".format(i+1, ptcp[-i-1], outcm[-i-1]))
if __name__ == '__main__':
    main()
