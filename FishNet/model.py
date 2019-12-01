import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, TimeDistributed, ConvLSTM2D
from keras.models import Model
from keras.models import Model, load_model
from keras.utils import plot_model
import numpy as np
base = 4
def conv_block(input_layer, base):
    inter_layer = Conv2D(base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    output_layer = Conv2D(base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inter_layer)
    return output_layer

def conv_maxpool_block(input_layer, base):
    inter_layer = Conv2D(base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_layer)
    inter_layer = Conv2D(base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inter_layer)
    output_layer = MaxPooling2D(pool_size=(2, 2))(inter_layer)
    return output_layer
def FishNet():
    inputs = Input((128,128,3))
    conv1 = Conv2D(base, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    
    pool2 = MaxPooling2D(pool_size=(16,16))(conv1)
    up2a = UpSampling2D(size = (2,2))(pool2)
    up2b = UpSampling2D(size = (2,2))(up2a)

    conv3a  = conv_maxpool_block(conv1, base)
    conv3b  = conv_maxpool_block(conv3a, base)
    
    merge4 = concatenate([conv3b,up2b], axis = 3)
    conv4 = conv_block(merge4, base)
    
    conv5a  = conv_maxpool_block(conv4, base)
    conv5b  = conv_maxpool_block(conv5a, base)
    conv5c  = conv_maxpool_block(conv5b, base)
    
    up5a = UpSampling2D(size = (2,2))(conv5c)
    conv5d = conv_block(up5a, base)
    up5b = UpSampling2D(size = (2,2))(conv5d)
    conv5e = conv_block(up5b, base)
    up5c = UpSampling2D(size = (2,2))(conv5e)
    conv5f = conv_block(up5c, base)
    up5d = UpSampling2D(size = (2,2))(conv5f)
    conv5g = conv_block(up5d, base)
    up5e = UpSampling2D(size = (2,2))(conv5g)
    conv5h = conv_block(up5e, base)
    
    
    up6a = UpSampling2D(size = (2,2))(conv4)
    conv6a = conv_block(up6a, base)
    up6b = UpSampling2D(size = (2,2))(conv6a)
    conv6b = conv_block(up6b, base)
    
    merge7 = concatenate([conv5h,conv6b], axis = 3)
    model = Model(input = inputs, output = merge7)

    model.summary()
    return model