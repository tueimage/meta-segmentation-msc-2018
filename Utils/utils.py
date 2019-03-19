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
def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))
def read_nifti(address):
    img = nib.load(address).get_data()
    return img
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def shannon_entropy(image, base=2):
    _, counts = np.unique(image, return_counts=True)
    return scipy_entropy(counts, base=base)

def mutual_information(hgram):
     """ Mutual information for joint histogram
     """
     # Convert bins counts to probability values
     pxy = hgram / float(np.sum(hgram))
     px = np.sum(pxy, axis=1) # marginal for x over y
     py = np.sum(pxy, axis=0) # marginal for y over x
     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
     # Now we can do the calculation using the pxy, px_py 2D arrays
     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
     return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def subset_index_to_address(subset_indices, train_addresses):
    addresses = []
    for i in subset_indices:
        addresses.append(train_addresses[i])
    return addresses
def makejpg(addresses, result_folder,task):
    count = 0
    if not os.path.isdir("/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}".format(result_folder,task)):
        os.mkdir("/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}".format(result_folder,task))
    if not os.path.isdir("/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}/images".format(result_folder,task)):
        os.mkdir("/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}/images".format(result_folder,task))
    if not os.path.isdir("/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}/labels".format(result_folder,task)):
        os.mkdir("/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}/labels".format(result_folder,task))
    for address in tqdm(addresses):
        img =  (nib.load(os.path.join('/home/tjvsonsbeek/decathlonData/'+task+'/imagesTr', address))).get_data()
        labels =  ((nib.load(os.path.join('/home/tjvsonsbeek/decathlonData/'+task+'/labelsTr', address))).get_data()>0).astype(np.uint8)
        if len(img.shape) == 4:
            channel = random.choice(range(img.shape[3]))
            img = img[:,:,:,channel]
        for z in range(img.shape[2]):
            if np.count_nonzero(labels[:,:,z])>90:
                scipy.misc.imsave('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}/images/{}.png'.format(result_folder,task,count),  cv2.resize(img[:,:,z],(224,224)))
                scipy.misc.imsave('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/{}_{}/labels/{}.png'.format(result_folder,task,count),  255*cv2.resize(labels[:,:,z],(224,224)))
                count+=1


def generator(addresses, minibatch_size, imageDimensions):
     f = 1
     # Create empty arrays to contain batch of features and labels#
     batch_features = np.zeros((minibatch_size, imageDimensions[0],imageDimensions[1],3))
     batch_labels = np.zeros((minibatch_size, imageDimensions[0],imageDimensions[1],1))
     while True:
        for i in range(minibatch_size):
            # choose random index in features
            index= random.choice(range(len(addresses)))
            im = cv2.imread(addresses[index])

            label_address = addresses[index].replace("images", "labels")

            label = cv2.imread(label_address, cv2.IMREAD_GRAYSCALE)
            label = [x / 255 for x in label]
            # print(im)
            # print(im.shape)
            batch_features[i,:,:,:] = im
            batch_labels[i,:,:,0] = label

        yield batch_features, batch_labels
def meta_generator_OM(addresses, labels, minibatch_size, imageDimensions):
     f = 1
     # Create empty arrays to contain batch of features and labels#
     batch_features = np.zeros((minibatch_size * 10, imageDimensions[0],imageDimensions[1],3))
     batch_labels = np.zeros((minibatch_size * 10, 3))
     while True:
        for batch in range(10):
            i = random.choice(range(len(addresses)))
            for j in range(minibatch_size):
                # choose random index in features

                im = cv2.imread(addresses[i][j])

                batch_features[batch*5+j,:,:,:] = im
                batch_labels[batch*5+j,:] = labels[i]


        yield batch_features, batch_labels
def meta_pred_generator_OM(addresses, minibatch_size, imageDimensions):
     f = 1
     # Create empty arrays to contain batch of features and labels#
     batch_features = np.zeros((minibatch_size * 10, imageDimensions[0],imageDimensions[1],3))
     while True:
        for batch in range(10):
            i = random.choice(range(len(addresses)))
            for j in range(minibatch_size):
                # choose random index in features

                im = cv2.imread(addresses[i][j])

                batch_features[batch*5+j,:,:,:] = im


        yield batch_features
def visualize_meta_features(features, tasks_list, titles):
    mean_and_std = np.zeros((len(tasks_list), len(features[tasks_list[0]][0]), 2))
    for task in range(len(tasks_list)):
        organized_features = np.zeros((len(features[tasks_list[task]]),len(features[tasks_list[task]][0])))
        for meta_feature in range(len(features[tasks_list[task]])):
            for i in range(len(features[tasks_list[task]][meta_feature])):
                organized_features[meta_feature, i] = features[tasks_list[task]][meta_feature][i]
        for i in range(len(features[tasks_list[task]][0])):
            mean_and_std[task, i, 0] = np.mean(organized_features[:, i])
            mean_and_std[task, i, 1] = np.std(organized_features[:, i])
    for i in range(len(features[tasks_list[task]][0])):
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
    mean_and_std = np.zeros((len(tasks_list), len(labels[tasks_list[0]][0]), 2))
    for task in range(len(tasks_list)):
        organized_labels = np.zeros((len(labels[tasks_list[task]]),len(labels[tasks_list[task]][0])))
        for meta_label in range(len(labels[tasks_list[task]])):
            for i in range(len(labels[tasks_list[task]][meta_label])):
                organized_labels[meta_label, i] = labels[tasks_list[task]][meta_label][i]
        for i in range(len(labels[tasks_list[task]][0])):
            mean_and_std[task, i, 0] = np.mean(organized_labels[:, i])
            mean_and_std[task, i, 1] = np.std(organized_labels[:, i])
    for i in range(len(labels[tasks_list[task]][0])):
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
def visualize_regression_one_method(gt, methods, methods_names):
    # RS = 20150101
    # sns.set_style('darkgrid')
    # sns.set_palette('muted')
    # sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    x = np.arange(len(gt))
    order = np.argsort(gt)
    # gt.sort()
    # plt.bar(x-0.3, gt, width = 0.5)
    plt.plot(x, gt)
    count = 0




    for method in methods:
        nMethod = np.array(method)[order]
        # plt.bar(x+0.2*count, nMethod, width = 0.2)
        plt.plot(x, method)
        count+=1
    legend_names = ['y'] + methods_names
    plt.legend(legend_names, loc='upper left')
    plt.savefig('graphs/lineNotOrdered.png')
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
    new_addresses = []
    indices = list(range(len(addresses)))
    random.shuffle(indices, random.random)
    indices = indices[:size]

    for index in indices:
        new_addresses.append(addresses[index])
    return new_addresses
def read_decathlon_data():


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



#prepare data
# taskslist= ['Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon']
# for task in taskslist:
#     print(task)
#     train_addresses = os.listdir('/home/tjvsonsbeek/decathlonData/'+task+'/imagesTr')
#     valid_addresses = train_addresses[:int(len(train_addresses)/4)]
#     train_addresses = train_addresses[int(len(train_addresses)/4):]
#     makejpg(train_addresses, 'train', task)
#     makejpg(valid_addresses, 'valid', task)

#load data addresses
