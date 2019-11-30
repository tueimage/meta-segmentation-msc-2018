import os
import numpy as np
import pickle
import traceback

class Data():
    def __init__(self, task):
        self.task = task
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.meta_train_data = []
        self.meta_val_data = []
        self.meta_test_data = []
    # def shuffle(self):
    #
    def subset_training_data(self):
        self.train_data = create_data_subsets(self.train_data, self.train_size)
    def subset_valid_data(self):
        self.val_data = create_data_subsets(self.val_data, self.val_size)
    def subset_test_data(self):
        self.test_data = create_data_subsets(self.test_data, self.test_size)
    def get_meta_subsets(self, nr_of_subsets, subset_size):
        self.meta_subsets = np.random.randint(0, len(self.train_data), size = (nr_of_subsets, subset_size))
    # def subset_meta_training_data(self, subset_indices):
    #     for i in subset_indices:
    #         self.meta_train_data.append(self.train_data[i])
    # def subset_meta_valid_data(self, subset_indices):
    #     for i in subset_indices:
    #         self.meta_val_data.append(self.val_data[i])
    # def subset_meta_test_data(self, subset_indices):
    #     for i in subset_indices:
    #         self.meta_test_data.append(self.test_data[i])

    def load_training_data(self):
        train_addresses = os.listdir('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/train_{}/images/'.format(self.task))
        for address in train_addresses:
            self.train_data.append('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/train_{}/images/{}'.format(self.task,address))
    def load_valid_data(self):
        valid_addresses = os.listdir('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/valid_{}/images/'.format(self.task))
        for address in valid_addresses:
            if address !='labels':
                self.val_data.append('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/valid_{}/images/{}'.format(self.task,address))
    def load_test_data(self):
        test_addresses = os.listdir('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/test_{}/images/'.format(self.task))
        for address in train_addresses:
            self.train_data.append('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/train_{}/images/{}'.format(self.task,address))


        self.meta_test_data = []
    # def shuffle(self):
    #
    def subset_training_data(self):
        self.train_data = create_data_subsets(self.train_data, self.train_size)
    def subset_valid_data(self):
        self.val_data = create_data_subsets(self.val_data, self.val_size)
    def subset_test_data(self):
        self.test_data = create_data_subsets(self.test_data, self.test_size)
    def get_meta_subsets(self, nr_of_subsets, subset_size):
        self.meta_subsets = np.random.randint(0, len(self.train_data), size = (nr_of_subsets, subset_size))
    # def subset_meta_training_data(self, subset_indices):
    #     for i in subset_indices:
    #         self.meta_train_data.append(self.train_data[i])
    # def subset_meta_valid_data(self, subset_indices):
    #     for i in subset_indices:
    #         self.meta_val_data.append(self.val_data[i])
    # def subset_meta_test_data(self, subset_indices):
    #     for i in subset_indices:
    #         self.meta_test_data.append(self.test_data[i])

    # def load_training_data(self):
    #     train_addresses = os.listdir('/home/tjvsonsbeek/decathlonData/{}/imagesTr'.format(self.task))
    #     for address in train_addresses:
    #         self.train_data.append('/home/tjvsonsbeek/decathlonData/{}/imagesTr/{}'.format(self.task, address))
    # def load_training_labels(self):
    #     for address in range(len(self.train_data)):
    #         nr = self.train_data[address[-10:-7]]
    #         if nr[0] == '_':
    #             nr = nr[1:]
    #         train_data = list(np.load('/home/tjvsonsbeek/featureExtractorUnet/decathlonMetaLabels/{}/{}.npy'.format(self.task, nr)))
    #         self.train_data[address] = [self.train_data[address], labels]
    # def load_valid_data(self):
    #     self.valid_data = self.train_data[np.ceil(len(self.train_data)*0.75):]
    #     self.train_data = self.train_data[:np.ceil(len(self.train_data)*0.75)]
    # def load_test_data(self):
    #     test_addresses = os.listdir('/home/tjvsonsbeek/decathlonData/{}/imagesTs'.format(self.task))
    #     for address in test_addresses:
    #         self.train_data.append('/home/tjvsonsbeek/decathlonData/{}/imagesTs/{}'.format(self.task,address))
['braintumour', 'heart', 'liver', 'hippocampus', 'prostate', 'lung', 'pancreas', 'spleen', 'colon']

['BCVuniandes', 'beomheep', 'CerebriuDIKU', 'EdwardMa12593', 'ildoo', 'iorism82', 'isarasua', 'Isensee', 'jiafucang', 'lesswire1', 'lupin', 'oldrich.kodym', 'ORippler', 'phil666', 'rzchen_xmu', 'ubilearn', 'whale', '17111010008', 'allan.kim01']
class MetaData():
    def __init__(self, task, fe):

        self.name = task
        self.feature_extractor = fe
        self.total_addresses = []
        self.total_results = []
        self.addresses = []
        self.results = []

    def save(self):
        """save class as self.name.txt"""
        file = open('meta_data/'+self.name+'_' + self.feature_extractor+'.txt','wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self):
        """try load self.name.txt"""
        file = open('meta_data/'+self.name+'_' + self.feature_extractor+'.txt','rb')
        dataPickle = file.read()
        file.close()

        self.__dict__ = pickle.loads(dataPickle)
