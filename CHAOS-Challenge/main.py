import model
from utils import train_generator
import os
from keras.callbacks import ModelCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
def main():
    m = model.unet()
    callbacks = [ModelCheckpoint('best_model.h5', monitor='loss', period = 20)]
    history = m.fit_generator(train_generator(5), steps_per_epoch = 30, epochs = 2000, callbacks = callbacks, verbose = 1)
    model.save_model(m)
    print(history)



if __name__ == '__main__':
    main()
