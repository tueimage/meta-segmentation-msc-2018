from meta_learner_utils import joint_normalize_metafeatures, feature_selection, dnn_model, reset_weights, feature_selection_test_set, task_specific_features, normalize_metafeatures, add_task_specific_metafeatures, preprocess_metalabels, preprocess_metafeatures, preprocess_metafeatures_test_set, regression_feature_selection, largest_indices
from visualization import visualize_features_tSNE, visualize_features_MDS, visualize_confusion_matrix, visualize_features_PCA
from medical_metafeatures.feature_extraction import MetaFeatureExtraction
from tqdm import tqdm
import os
import numpy as np
from sklearn.svm import SVR
import keras.backend as K
import matplotlib.pyplot as plt


def main():
    tasks_list=  ['Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon','Task11_CHAOSLiver', 'Task12_LITSLiver','Task13_ACDCHeart']
    feature_extractors = ['STAT','VGG16','ResNet50','MobileNetV1']
    regression_methods = ['SVR','DNN']
    best_frac = {'STAT': 0.37,'VGG16': 0.23,'ResNet50': 0.49,'MobileNetV1': 0.49}
    meta_subset_size = 20
    sample_size = 100
    visualization = False
    do_feature_selection = True
    test_results = np.zeros((len(feature_extractors), len(regression_methods), len(tasks_list[10:])))
# load the meta_features and load_meta_labels
    meta_features = {}
    meta_labels   = {}



    for fe_id, fe in enumerate(feature_extractors):
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

            regression_features_raw = add_task_specific_metafeatures(regression_features_raw, task_specific_features, tasks_list[:10], sample_size)
            regression_features_raw_test_set = add_task_specific_metafeatures(regression_features_raw_test_set, task_specific_features[10:], tasks_list[10:], 1)

        regression_features_raw, regression_features_raw_test_set =  joint_normalize_metafeatures(regression_features_raw, regression_features_raw_test_set)
        if visualization:
             visualize_confusion_matrix(regression_features_raw, dataset_xlsx, fe, legend = False)
            visualize_features_tSNE(np.concatenate([regression_features_raw, regression_features_raw_test_set],axis=0), dataset_xlsx, list(range(13)), fe+'testset',fe, legend = False)
            visualize_features_MDS(np.concatenate([regression_features_raw, regression_features_raw_test_set],axis=0), dataset_xlsx, list(range(13)), fe+'testset',fe, legend = False)
            visualize_features_PCA(np.concatenate([regression_features_raw, regression_features_raw_test_set],axis=0), dataset_xlsx, list(range(13)), fe+'testset',fe, legend = False)
        # regression_features_raw_test_set =  normalize_metafeatures(regression_features_raw_test_set)
        if do_feature_selection:
            # print('Feature selection')
            regression_features, features_to_keep = feature_selection(regression_features_raw, regression_labels, best_frac[fe_id])
            regression_features_test_set = feature_selection_test_set(regression_features_raw_test_set, features_to_keep)
        else:
            regression_features = regression_features_raw
            regression_features_test_set = regression_features_raw_test_set
            # print('Nofeature selection')
        
        for reg_id, regression_method in enumerate(regression_methods):
            for task_id, task in enumerate(tasks_list[10:]):
                if regression_method == 'DNN':
                    K.clear_session()
                    model = dnn_model(regression_features.shape[1])
                    np.random.seed(42)
                    np.random.shuffle(regression_features)
                    np.random.seed(42)
                    np.random.shuffle(regression_labels)
                    model = reset_weights(model)
                    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
                    model.fit(regression_features, regression_labels[:,7], epochs=10, verbose=0, shuffle=True)
                    test_result = model.predict(regression_features_test_set[task_id:(task_id+1),:])
                    print(task)
                    print(np.mean(test_result))
                    test_results[fe_id, reg_id, task_id] = np.mean(test_result)
                elif regression_method == 'SVR':
                    regression_model = SVR()
                    regression_model.fit(regression_features, regression_labels[:,7])
                    test_result = regression_model.predict(regression_features_test_set[task_id:(task_id+1),:])
                    print(task)
                    print(np.mean(test_result))
                    test_results[fe_id, reg_id, task_id] = np.mean(test_result)
                else:
                    print('Regression method not known')
                    
    np.save('output/test_results.npy', test_results)
            




if __name__ == '__main__':
    main()
