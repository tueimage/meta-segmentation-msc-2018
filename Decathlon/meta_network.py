from utils import meta_generator
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Flatten, Dense
import numpy as np
from scipy.stats import kurtosis, skew

class meta_learner():
    def __init__(self, fe):
        self.fe = fe
        self.minibatch_size= 5
        self.epochs = 10


    def build_meta_model(self):
        flat1  = Flatten()(self.feature_extractor.layers[-1].output)
        dense1 = Dense(1024, activation = 'relu')(flat1)
        dense2 = Dense(500, activation = 'relu')(dense1)
        dense3 = Dense(3, activation = 'softmax')(dense2)
        self.model =  Model(input = self.feature_extractor.layers[0].output, output = dense3)
        self.model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['accuracy'])

    def train(self, train_data, train_labels, imageDimensions, verbosity):
        self.history = self.model.fit_generator(meta_generator(train_data, train_labels, self.minibatch_size, imageDimensions), steps_per_epoch = 10, nb_epoch = self.epochs, verbose = verbosity)
    def save_model(self):
        print("---SAVING MODEL---")
        self.model.save_weights("models/meta.h5")
    def load_weights(self):
        self.weights_file = "/home/tjvsonsbeek/featureExtractorUnet/models/meta.h5"
        self.model.load_weights(self.weights_file)
    def build_feature_extractor(self):
        if self.fe == 'VGG16':
            from keras.applications.vgg16 import VGG16
            self.feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape = (224,224,3))
        elif self.fe == 'VGG19':
            from keras.applications.vgg19 import VGG19
            self.feature_extractor = VGG19(weights='imagenet', include_top=False)
        elif self.fe == 'ResNet50':
            from keras.applications.resnet50 import ResNet50
            self.feature_extractor = ResNet50(weights='imagenet', include_top=False)
        elif self.fe == 'MobileNetV1':
            from keras.applications.mobilenet import MobileNet
            self.feature_extractor = MobileNet(input_shape = (224,224,3), weights='imagenet', include_top=False)
            for layer in range(len(self.feature_extractor.layers)):
                print("{}_{}_{}".format(layer, self.feature_extractor.layers[layer], self.feature_extractor.layers[layer].output.shape))
        elif self.fe == 'MobileNetV2':
            from keras.applications.mobilenet_v2 import MobileNetV2
            self.feature_extractor = MobileNetV2(weights='imagenet', include_top=False)
        else:
            print("MASSIVE FAIL!! No Encoder loaded")
class meta_regressor():
    def __init__(self):
        self.placeholder = 0
    def gather_meta_features(dataset):
        for address in dataset:
            im = cv2.imread(address)
            mean = np.mean(im)
            std = np.std(im)
            skewness = skew(im)
            kurt = kurtosis(im)
