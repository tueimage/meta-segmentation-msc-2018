import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, TimeDistributed, ConvLSTM2D
from keras.models import Model
from keras.optimizers import Adam, SGD
import numpy as np
import cv2
import keras.backend as K
import random
from keras.models import Model, load_model

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    loss = 0
    loss -= dice_coef(y_true, y_pred)

    return loss

def unet(input_size = (64, 128, 128,1)):
    base = 16
    inputs = Input(input_size)
    conv1 = TimeDistributed(Conv2D(base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(inputs)
    conv1 = TimeDistributed(Conv2D(base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = TimeDistributed(Conv2D(2*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool1)
    conv2 = TimeDistributed(Conv2D(2*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    conv3 = TimeDistributed(Conv2D(4*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool2)
    conv3 = TimeDistributed(Conv2D(4*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    conv4 = TimeDistributed(Conv2D(8*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool3)
    conv4 = TimeDistributed(Conv2D(8*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)

    conv5 = TimeDistributed(Conv2D(16*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(pool4)
    conv5 = TimeDistributed(Conv2D(16*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv5)
    # drop5 = Dropout(0.3)(conv5)
    # print(drop5.shape)
    lstm5 = ConvLSTM2D(16*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format = 'channels_last', return_sequences = True)(conv5)

    up6 = TimeDistributed(Conv2D(16*base, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(TimeDistributed(UpSampling2D(size = (2,2)))(lstm5))
    merge6 = concatenate([conv4, up6], axis = 4)
    conv6 = TimeDistributed(Conv2D(8*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge6)
    conv6 = TimeDistributed(Conv2D(8*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv6)

    up7 = TimeDistributed(Conv2D(4*base, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(TimeDistributed(UpSampling2D(size = (2,2)))(conv6))
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7 = TimeDistributed(Conv2D(4*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge7)
    conv7 = TimeDistributed(Conv2D(4*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv7)

    up8 = TimeDistributed(Conv2D(2*base, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(TimeDistributed(UpSampling2D(size = (2,2)))(conv7))
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8 = TimeDistributed(Conv2D(2*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge8)
    conv8 = TimeDistributed(Conv2D(2*base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv8)

    up9 = TimeDistributed(Conv2D(base, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(TimeDistributed(UpSampling2D(size = (2,2)))(conv8))
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9 = TimeDistributed(Conv2D(base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(merge9)
    conv9 = TimeDistributed(Conv2D(base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv9)
    conv9 = TimeDistributed(Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))(conv9)
    conv10 = TimeDistributed(Conv2D(1, 1, activation = 'sigmoid'))(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = SGD(lr = 1e-4), loss = dice_coef_loss, metrics = ['accuracy'])
    model.summary()
    return model
def save_model(network):
    print("---SAVING NETWORK WEIGHTS---")
    network.save_weights("model1.h5")
