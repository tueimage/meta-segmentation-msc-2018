from meta_network import meta_regressor
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt

def main():
    tasks_list=  [ 'Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus','Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas','Task08_HepaticVessel','Task09_Spleen','Task10_Colon']
    participants = ['BCVuniandes', 'beomheep', 'CerebriuDIKU', 'EdwardMa12593', 'ildoo', 'iorism82', 'isarasua', 'Isensee', 'jiafucang', 'lesswire1', 'lupin', 'oldrich.kodym', 'ORippler', 'phil666', 'rzchen_xmu', 'ubilearn', 'whale', '17111010008', 'allan.kim01']
    meta_feature_names = ['nr of instances', 'nr of features', 'nr of target concept values', 'dataset dimensionality', 'mean correlation coefficient', 'mean skewness', 'mean kurtosis', 'nomalized class entropy', 'mean normalized feature entropy', 'mean mutual inf', 'max mutual inf', 'equivalent nr of features', 'noise signal ratio']

    meta_features = {}
    meta_labels   = {}
    n = 500
    nr_of_subsets = 50
    for task in tasks_list:
        m = meta_regressor(task)
        m.load_meta_labels()
        meta_labels[task] = m.meta_labels
        m.load_meta_features()
        meta_features[task] = m.meta_features
    ## visualise the meta_features and meta_labels
    # visualize_meta_features(meta_features, tasks_list, meta_feature_names)
    # visualize_meta_labels(meta_labels, tasks_list, participants)
    ## do the regression
    ## compose the regression features and labels
    regression_features = np.zeros((n, 13))
    regression_labels   = np.zeros((n, 19))
    for task in range(len(tasks_list)):
        for nr in range(nr_of_subsets):
            regression_features[task*nr_of_subsets+nr,:] = meta_features[tasks_list[task]][nr]
            regression_labels[task*nr_of_subsets+nr,:] = meta_labels[tasks_list[task]][nr]
    isensee = regression_labels[:,7]
    print(isensee)
    for i in range(regression_features.shape[1]):
        std = np.std(regression_features[:,i])
        if std == 0:
            std = 1
        regression_features[:,i] = (regression_features[:,i]-np.mean(regression_features[:,i]))/std

    pca = PCA(2)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(regression_features)
    plt.figure()
    plt.scatter(projected[:, 0], projected[:, 1],
            c=isensee, edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('spectral', 99))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.savefig("graphs/pca.png")
    # print(pca.explained_variance_ratio_)
if __name__ == '__main__':
    main()
