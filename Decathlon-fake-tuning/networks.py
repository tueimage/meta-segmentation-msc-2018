from keras.models import Model, load_model
from keras.layers import Input, Conv2D,MaxPooling2D, UpSampling2D, concatenate, Dropout,add, Dense
from keras.optimizers import Adam
from utils import fake_tune_generator, historyPlot, create_data_subsets, dice_coef_loss, auc, mean_iou, dice_coef
import os
from tqdm import tqdm
import numpy as np
import cv2
import random
import copy
from keras.preprocessing.image import ImageDataGenerator
class EncoderDecoderNetwork():
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.callbacks = []
    def load_weights(self):
        self.weights_file = "/home/tjvsonsbeek/featureExtractorUnet/models_fake_tune/final_model_{}_{}.h5".format(self.task, self.name)
        self.model.load_weights(self.weights_file)
    # def predict(self, model, data):
    #     result = model.predict_generator(generator(data, self.minibatch_size, self.imageDimensions), steps = 1)
    #     return result
    def update_encoder_weights(self):
        if self.name == 'VGG16':
            self.feature_extractor = Model(input = self.model.input, output = self.model.layers[18].output)
        elif self.name == 'ResNet50':
            self.feature_extractor = Model(input = self.model.input, output = self.model.layers[172].output)
        elif self.name == 'MobileNetV1':
            self.feature_extractor = Model(input = self.model.input, output = self.model.layers[81].output)


    def get_meta_data(self, addresses):
        # self.model_ft_meta = self.model

        self.model.fit_generator(generator(addresses, 1, self.imageDimensions), steps_per_epoch= self.minibatch_size, nb_epoch = 10, verbose = 0)
        subset_features = self.model.evaluate_generator(generator(addresses, 1, self.imageDimensions), steps= self.minibatch_size)
        return subset_features



    def add_callback(self, callback):
        self.callbacks.append(callback)
    def train(self, train_data, val_data, imageDimensions, verbosity):
        self.history = self.model.fit_generator(fake_tune_generator(train_data, self.minibatch_size, imageDimensions), steps_per_epoch = 200, nb_epoch = self.epochs, validation_data = fake_tune_generator(val_data, self.minibatch_size, imageDimensions), validation_steps = 50, verbose = verbosity)
    def save_model(self):
        print("---SAVING MODEL---")
        self.model.save_weights("models_fake_tune/final_model_{}_{}.h5".format(self.task, self.name))
    def build_encoder(self):
        if self.name == 'VGG16':
            from keras.applications.vgg16 import VGG16
            self.feature_extractor = VGG16(weights='imagenet', include_top=False)
        elif self.name == 'VGG19':
            from keras.applications.vgg19 import VGG19
            self.feature_extractor = VGG19(weights='imagenet', include_top=False)
        elif self.name == 'ResNet50':
            from keras.applications.resnet50 import ResNet50
            self.feature_extractor = ResNet50(input_shape = (224,224,3),weights='imagenet', include_top=False)
        elif self.name == 'MobileNetV1':
            from keras.applications.mobilenet import MobileNet
            self.feature_extractor = MobileNet(input_shape = (224,224,3), weights='imagenet', include_top=False)
            for layer in range(len(self.feature_extractor.layers)):
                print("{}_{}_{}".format(layer, self.feature_extractor.layers[layer], self.feature_extractor.layers[layer].output.shape))
        elif self.name == 'MobileNetV2':
            from keras.applications.mobilenet_v2 import MobileNetV2
            self.feature_extractor = MobileNetV2(weights='imagenet', include_top=False)
        else:
            print("MASSIVE FAIL!! No Encoder loaded")

    def build_classifier(self):
        if self.name == 'VGG16':
            from keras.applications.vgg16 import VGG16
            self.classifier = VGG16(weights='imagenet', include_top=False)
        elif self.name == 'VGG19':
            from keras.applications.vgg19 import VGG19
            self.classifier = VGG19(weights='imagenet', include_top=False)
        elif self.name == 'ResNet50':
            from keras.applications.resnet50 import ResNet50
            self.classifier = ResNet50(weights='imagenet', include_top=False)
        elif self.name == 'MobileNetV1':
            from keras.applications.mobilenet import MobileNet
            self.classifier = MobileNet(input_shape = (224,224,3), weights='imagenet', include_top=False)
        elif self.name == 'MobileNetV2':
            from keras.applications.mobilenet_v2 import MobileNetV2
            self.classifier = MobileNetV2(weights='imagenet', include_top=False)
        else:
            print("MASSIVE FAIL!! No Encoder loaded")
        # build top:
        dense1 = Dense(4096, activation = 'relu')(self.classifier.layers[-1].ouput)
        dense2 = Dense(1000, activation = 'relu')(dense1)
        dense3 = Dense(9, activation = 'softmax)(dense2')
        self.classifier = Model(input = self.classifier.layers[0].output, output = dense3)

    def build_decoder(self):
        if self.name == 'VGG16':
            self.build_decoder_VGG16()
        elif self.name == 'VGG19':
            self.build_decoder_VGG19()
        elif self.name == 'ResNet50':
            self.build_decoder_RESNET50()
        elif self.name == 'MobileNetV1':
            self.build_decoder_MobileNetV1()
        elif self.name == 'MobileNetV2':
            self.build_decoder_MobileNetV1()
        else:
            print("MASSIVE FAIL!! No Decoder loaded")
    def build_decoder_VGG16(self):
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(self.feature_extractor.layers[18].output)
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        merge6 = concatenate([self.feature_extractor.layers[17].output,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([self.feature_extractor.layers[13].output,up7], axis = 3)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([self.feature_extractor.layers[9].output,up8], axis = 3)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([self.feature_extractor.layers[5].output,up9], axis = 3)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
        merge10 = concatenate([self.feature_extractor.layers[2].output,up10], axis = 3)
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
        self.model =  Model(input = self.feature_extractor.layers[0].output, output = conv10)
    def build_decoder_VGG19(self):
        conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(self.feature_extractor.layers[21].output)
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        merge6 = concatenate([self.feature_extractor.layers[20].output,up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([self.feature_extractor.layers[15].output,up7], axis = 3)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([self.feature_extractor.layers[10].output,up8], axis = 3)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([self.feature_extractor.layers[5].output,up9], axis = 3)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
        merge10 = concatenate([self.feature_extractor.layers[2].output,up10], axis = 3)
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
        self.model =  Model(input = self.feature_extractor.layers[0].output, output = conv10)
    def build_decoder_RESNET50(self):
        print(self.feature_extractor.summary())
        conv5 = Conv2D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(self.feature_extractor.layers[172].output)
        up6 = Conv2D(2048, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
        merge6 = concatenate([self.feature_extractor.layers[140].output,up6], axis = 3)
        conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([self.feature_extractor.layers[78].output,up7], axis = 3)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([up8,up8], axis = 3)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([self.feature_extractor.layers[3].output,up9], axis = 3)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
        merge10 = concatenate([self.feature_extractor.layers[0].output,up10], axis = 3)
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
        self.model = Model(input = self.feature_extractor.layers[0].output, output = conv10)

    def build_decoder_MobileNetV1(self):
        up6 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(self.feature_extractor.layers[81].output))
        merge6 = concatenate([self.feature_extractor.layers[69].output,up6], axis = 3)
        conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(1024, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([self.feature_extractor.layers[33].output,up7], axis = 3)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([self.feature_extractor.layers[21].output,up8], axis = 3)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([self.feature_extractor.layers[9].output,up9], axis = 3)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

        up10 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
        merge10 = concatenate([self.feature_extractor.layers[0].output,up10], axis = 3)
        conv10 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
        self.model = Model(input = self.feature_extractor.layers[0].output, output = conv10)
# def create_train_generator(self):
#     image_datagen = ImageDataGenerator()
#     mask_datagen = ImageDataGenerator()
#     seed = random.seed()
#
#     image_generator = mask_datagen.flow_from_directory('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/train_{}/images/'.format(self.task), target_size = (224,224), batch_size = self.minibatch_size, seed=seed)
#
#     mask_generator = mask_datagen.flow_from_directory('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/train_{}/labels/'.format(self.task), target_size = (224,224), batch_size = self.minibatch_size, seed=seed)
#     # combine generators into one which yields image and masks
#     self.train_generator = zip(image_generator, mask_generator)
# def create_val_generator(self):
#     image_datagen = ImageDataGenerator()
#     mask_datagen = ImageDataGenerator()
#     seed = random.seed()
#     print('/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/valid_{}/images'.format(self.task))
#     image_generator = mask_datagen.flow_from_directory('decathlonDataProcessed/valid_{}/images'.format(self.task), target_size = (224,224), class_mode=None, batch_size = self.minibatch_size, seed=seed)
#
#     mask_generator = mask_datagen.flow_from_directory('decathlonDataProcessed/valid_{}/labels'.format(self.task), target_size = (224,224), class_mode=None, batch_size = self.minibatch_size, seed=seed)
#     # combine generators into one which yields image and masks
#     self.valid_generator = zip(image_generator, mask_generator)
