from MetaFeatureExtraction import MetaFeatureExtraction
from tqdm import tqdm
import numpy as np
import os
from keras.applications.vgg19 import VGG19
from keras.models import Model, model_from_json
from networks import EncoderDecoderNetwork
from keras.applications.mobilenet import relu6
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():
    tasks_list=['Task11_CHAOSLiver']
#

    dataset_xlsx = ['braintumor', 'heart', 'liver', 'hippocampus', 'prostate', 'lung', 'pancreas', 'hepaticvessel', 'spleen', 'colon']
    meta_feature_names = ['nr of instances', 'mean pixel value', 'std pixel value', 'coefficient of variation', 'mean correlation coefficient', 'mean skewness', 'mean kurtosis', 'nomalized class entropy', 'mean normalized feature entropy', 'mean mutual inf', 'max mutual inf', 'equivalent nr of features', 'noise signal ratio']
    participants = ['BCVuniandes', 'beomheep', 'CerebriuDIKU', 'EdwardMa12593', 'ildoo', 'iorism82', 'isarasua', 'Isensee', 'jiafucang', 'lesswire1', 'lupin', 'oldrich.kodym', 'ORippler', 'phil666', 'rzchen_xmu', 'ubilearn', 'whale', '17111010008', 'allan.kim01']
    nr_of_subsets = 1
    feature_extractors = ['STAT','VGG16','ResNet50','MobileNetV1']
    filters = {'VGG16': 512, 'MobileNetV1': 1024, 'ResNet50': 2048}
    subset_size_5 =  {'Task01_BrainTumour': 5,'Task02_Heart': 5,'Task03_Liver': 5,'Task04_Hippocampus': 5, 'Task05_Prostate': 5, 'Task06_Lung': 5, 'Task07_Pancreas': 5, 'Task08_HepaticVessel': 5, 'Task09_Spleen': 5, 'Task10_Colon': 5}
    subset_size_10 = {'Task01_BrainTumour': 10,'Task02_Heart': 6,'Task03_Liver': 10,'Task04_Hippocampus': 10, 'Task05_Prostate': 10, 'Task06_Lung': 10, 'Task07_Pancreas': 10, 'Task08_HepaticVessel': 10, 'Task09_Spleen': 10, 'Task10_Colon': 10}
    subset_size_20 = {'Task01_BrainTumour': 20,'Task02_Heart': 7,'Task03_Liver': 20,'Task04_Hippocampus': 20, 'Task05_Prostate': 11, 'Task06_Lung': 20, 'Task07_Pancreas': 20, 'Task08_HepaticVessel': 20, 'Task09_Spleen': 16, 'Task10_Colon': 20}

    dataset_size = {'Task01_BrainTumour': 266,'Task02_Heart': 10,'Task03_Liver': 60,'Task04_Hippocampus': 130, 'Task05_Prostate': 16, 'Task06_Lung': 32, 'Task07_Pancreas': 139, 'Task08_HepaticVessel': 139, 'Task09_Spleen': 20, 'Task10_Colon': 64}
    for task_id, task in enumerate(tasks_list):
        for fe in feature_extractors:
            meta_subset_size = 5#subset_size_5[task]
            #nr_of_subsets = dataset_size[task]
            labels = np.zeros((nr_of_subsets, meta_subset_size, len(participants)))
            if fe == 'STAT':
                features = np.zeros((nr_of_subsets,meta_subset_size, 33))
                model = None
                nr_of_filters = None
            else:
                nr_of_filters = filters[fe]
                features = np.zeros((nr_of_subsets,meta_subset_size, 7,7,nr_of_filters))
                model = EncoderDecoderNetwork(fe,task_id)
                model.task = task
                model.build_encoder()
                model.build_decoder()
                model.load_weights()
                model.update_encoder_weights()

            for subset in tqdm(range(nr_of_subsets)):
                m = MetaFeatureExtraction(task, meta_subset_size, fe, model, nr_of_filters)
                if fe != 'STAT': m.load_model(model.feature_extractor)
                m.gather_random_addresses()
                #m.gather_address(subset)
                # m.gather_list_meta_labels()
                m.gather_meta_features()
                if fe == 'STAT':
                    features[subset,:] = m.meta_features
                else:
                    features[subset,:,:,:,:] = m.meta_features
                # print(np.count_nonzero(m.meta_labels))
                # labels[subset,:,:]   = m.meta_labels

            # np.save('metadata/meta_regressor_labels_{}_{}.npy'.format(task, fe), labels)
            np.save('metadata/meta_regressor_features_{}_{}.npy'.format(task, fe), features)

if __name__ == '__main__':
    main()
