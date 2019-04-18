import os
import numpy as np
import random
from matplotlib import pyplot as plt
import keras.backend as K
import scipy.misc
import seaborn as sns
from itertools import cycle
from sklearn.manifold import TSNE
import csv
from scipy.stats import entropy as scipy_entropy
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
def visualize_meta_features(features, tasks_list, titles):
    mean_and_std = np.zeros((len(tasks_list), features[tasks_list[0]].shape[1], 2))
    for task in range(len(tasks_list)):
        organized_features = features[tasks_list[task]]
        for i in range(len(features[tasks_list[task]][0])):
            mean_and_std[task, i, 0] = np.mean(organized_features[:, i])
            mean_and_std[task, i, 1] = np.std(organized_features[:, i])
    for i in range(organized_features.shape[1]):
        sns.set_style('darkgrid')
        sns.set_palette('muted')
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(tasks_list)), mean_and_std[:,i,0], yerr=mean_and_std[:,i,1], align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Value [-]')
        ax.set_xlabel('Dataset')
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.set_xticks([x - 0.5 for x in np.arange(len(tasks_list))])
        ax.set_xticklabels(tasks_list, rotation = 45)
        ax.set_title(titles[i])
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig("graphs/meta_features/{}.png".format(titles[i]))
def visualize_meta_labels(labels, tasks_list, titles):
    mean_and_std = np.zeros((len(tasks_list), labels[tasks_list[0]].shape[1], 2))
    for task in range(len(tasks_list)):
        organized_labels = labels[tasks_list[task]]
        for i in range(len(labels[tasks_list[task]][0])):
            mean_and_std[task, i, 0] = np.mean(organized_labels[:, i])
            mean_and_std[task, i, 1] = np.std(organized_labels[:, i])
    for i in range(organized_labels.shape[1]):
        sns.set_style('darkgrid')
        sns.set_palette('muted')
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(tasks_list)), mean_and_std[:,i,0], yerr=mean_and_std[:,i,1], align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Dice coefficient')
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.set_xticks([x - 0.5 for x in np.arange(len(tasks_list))])
        ax.set_xticklabels(np.arange(len(tasks_list)), rotation = 45)
        ax.set_title(titles[i])
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig("graphs/meta_labels/{}.png".format(titles[i]))
def visualize_regression_result(result, labels, datasets, name, current_datasets, fold):
    markers = ['1','s','p','h','+','x','*','d','o','<']
    colors  = ['b','slategray','r','m','c','k','lime','indigo','gold','peru']
    plt.figure()
    plt.plot([0,1],[0,1])
    print("R_squared: {}".format(np.corrcoef(result, labels)[0,1]))
    count = 0
    legend_names = ['x=y']
    for dataset_id in range(len(datasets)):
        if dataset_id in current_datasets:
            print(dataset_id)

            plt.scatter(labels[count*100:count*100+100], result[count*100:count*100+100], c = colors[dataset_id], edgecolor='none', alpha=0.8)#, marker= markers[dataset_id])
            count+=1
            legend_names.append(datasets[dataset_id])
    # plt.scatter(labels, result, c= color_labels, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('spectral', 99))


    plt.legend(legend_names)
    plt.text(0.6, 0.5, 'R-squared = %0.2f' % np.corrcoef(result, labels)[0,1])

    # plt.legend(legend_names, loc='upper left')
    plt.xlabel('Ground truth label')
    plt.ylabel('Predicted label')
    plt.title("Ground truth vs prediction of meta label. {} set. fold {}".format(name, fold))
    # plt.colorbar()
    plt.savefig('graphs/correlationPlot/{}_{}.png'.format(name, fold))

def visualize_features_tSNE(features, datasets, name, current_datasets, fold):
    RS = 20150101
    markers = ['1','s','p','h','+','x','*','d','o','<']
    colors  = ['b','slategray','r','m','c','k','lime','indigo','gold','peru']
    plt.figure()
    count = 0
    legend_names = []
    tsne_variable = TSNE(random_state = RS).fit_transform(features)
    for dataset_id in range(len(datasets)):
        if dataset_id in current_datasets:

            plt.scatter(tsne_variable[count*100:count*100+100,0], tsne_variable[count*100:count*100+100,1], c = colors[dataset_id])
            plt.errorbar(np.mean(tsne_variable[count*100:count*100+100,0]), np.mean(tsne_variable[count*100:count*100+100,1]), np.std(tsne_variable[count*100:count*100+100,0]), np.std(tsne_variable[count*100:count*100+100,1]), c = colors[dataset_id] )
            count+=1
            legend_names.append(datasets[dataset_id])
    # plt.scatter(labels, result, c= color_labels, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('spectral', 99))


    plt.legend(legend_names)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('tSNE plot')
    plt.savefig('graphs/tSNE/tSNEplot_{}_{}.png'.format(name, fold))

def visualize_features_MDS(features, datasets, name, current_datasets, fold):
    markers = ['1','s','p','h','+','x','*','d','o','<']
    colors  = ['b','slategray','r','m','c','k','lime','indigo','gold','peru']
    mds = MDS(n_components=2)
    mds_variable = mds.fit_transform(features)
    plt.figure()
    count = 0
    legend_names = []
    for dataset_id in range(len(datasets)):
        if dataset_id in current_datasets:

            plt.scatter(mds_variable[count*100:count*100+100,0], mds_variable[count*100:count*100+100,1], c = colors[dataset_id])
            plt.errorbar(np.mean(mds_variable[count*100:count*100+100,0]), np.mean(mds_variable[count*100:count*100+100,1]), np.std(mds_variable[count*100:count*100+100,0]), np.std(mds_variable[count*100:count*100+100,1]), c = colors[dataset_id] )
            count+=1
            legend_names.append(datasets[dataset_id])
    # plt.scatter(labels, result, c= color_labels, edgecolor='none', alpha=0.8, cmap=plt.cm.get_cmap('spectral', 99))


    plt.legend(legend_names)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('MDS plot')
    plt.savefig('graphs/MDS/MDSplot_{}_{}.png'.format(name, fold))

def visualize_confusion_matrix(labels, datasets, name, fold):

    confusion_matrix = np.zeros((labels.shape[0], labels.shape[0]))
    triangle_coordinates = np.tril_indices(labels.shape[0])
    confusion_matrix[triangle_coordinates] = 1
    confusion_matrix = np.flipud(confusion_matrix)
    print(labels.shape)
    print(labels.shape)
    for x in range(labels.shape[0]):
        for y in range(labels.shape[0]):
            if confusion_matrix[y,labels.shape[0]-1-x] == 1:
                confusion_matrix[y,labels.shape[0]-1-x] = np.sqrt(distance.euclidean(labels[x,:], labels[y,:]))

    # row_sums = confusion_matrix.sum(axis=1)
    norm_confusion_matrix = (confusion_matrix/np.max(confusion_matrix)) #/ row_sums[:, np.newaxis]

    fig = plt.figure()
    img2 = plt.imshow(norm_confusion_matrix, interpolation='nearest',cmap = plt.cm.get_cmap('hot', 99),origin='lower')

    plt.colorbar(img2,cmap=plt.cm.get_cmap('hot', 99))


    plt.yticks(list(range(50,1050,100)), datasets, rotation=0)
    plt.title('Relative euclidian distance between meta {}'.format(name))
    datasets_reverse = datasets.copy()
    datasets_reverse.reverse()
    plt.xticks(list(range(50,1050,100)), datasets_reverse, rotation=20, ha = 'right')
    fig.savefig("graphs/gridMaps/gridMap_{}_{}.tif".format(name, fold))
def visualize_features_PCA(features, datasets, name, current_datasets, fold):
    markers = ['1','s','p','h','+','x','*','d','o','<']
    colors  = ['b','slategray','r','m','c','k','lime','indigo','gold','peru']

    pca = PCA(2)  # project from 64 to 2 dimensions
    pca_variable = pca.fit_transform(features)
    plt.figure()
    count = 0
    legend_names = []
    for dataset_id in range(len(datasets)):
        if dataset_id in current_datasets:

            plt.scatter(pca_variable[count*100:count*100+100,0], pca_variable[count*100:count*100+100,1], c = colors[dataset_id])
            plt.errorbar(np.mean(pca_variable[count*100:count*100+100,0]), np.mean(pca_variable[count*100:count*100+100,1]), np.std(pca_variable[count*100:count*100+100,0]), np.std(pca_variable[count*100:count*100+100,1]), c = colors[dataset_id] )
            count+=1
            legend_names.append(datasets[dataset_id])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('PCA plot')
    plt.savefig('graphs/PCA/PCAplot_{}_{}.png'.format(name, fold))
def pca_plot(features, labels, name = '', feature = '', cropped = False):
    pca = PCA(2)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(features)
    plt.figure()
    plt.scatter(projected[:, 0], projected[:, 1],
            c=labels, edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('spectral', 99))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title("PCA plot")
    plt.colorbar()
    if cropped:
        plt.savefig("graphs/pca_{}_without_{}.png".format(name, feature))
    else:
        plt.savefig("graphs/pca_{}.png".format(name))

def joined_tSNE_plot(features_list):
    RS = 20150101
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.figure(1)
    cycol = cycle('bgrcmk')
    for sample in range(features_list.shape[0]):
        feature = features_list[sample,:,:,:].reshape([2048,49])
        tsne_variable = TSNE(random_state = RS).fit_transform(feature)
        plt.scatter(tsne_variable[:,0], tsne_variable[:,1], c = next(cycol))
        plt.errorbar(np.mean(tsne_variable[:,0]), np.mean(tsne_variable[:,1]), np.std(tsne_variable[:,0]), np.std(tsne_variable[:,1]), c= next(cycol) )
        plt.savefig("testColon.png")
