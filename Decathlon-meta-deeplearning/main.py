
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from utils import historyPlot,dice_coef_loss, makejpg, auc, mean_iou, dice_coef
from keras.callbacks import ModelCheckpoint, EarlyStopping
from networks import EncoderDecoderNetwork
from data import Data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():

    tasks_list= ['Task09_Spleen', 'Task10_Colon', 'Task02_Heart','Task03_Liver','Task05_Prostate', 'Task06_Lung']
    feature_extractors = ['VGG16', 'ResNet50', 'MobileNetV1']
    counter = 1
    for task in tasks_list:
        print("TASK: {}".format(task))
        data = Data(task)

        data.train_size = 1000
        data.val_size = 100
        data.imageDimensions = (224, 224)
        data.load_training_data()
        data.load_valid_data()
        # data.get_meta_subsets(100, 10)
        for fe in feature_extractors:
            print("FEATURE EXTRACTOR: {}".format(fe))
            struct = EncoderDecoderNetwork(fe, counter)
            struct.task = task
            struct.minibatch_size = 5
            struct.epochs = 10
            # struct.create_train_generator()
            # struct.create_val_generator()
            # struct.add_callback(ModelCheckpoint(filepath='best_model_{}_{}.h5'.format(struct.name, struct.id), verbose=2, save_best_only=True, save_weights_only = True, period = 1))
            # struct.add_callback(EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0))
            struct.build_encoder()
            print(len(struct.feature_extractor.layers))
            struct.build_decoder()
            struct.model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics = ['accuracy', auc, mean_iou])
            struct.train(data.train_data, data.val_data, data.imageDimensions, verbosity = 1)
            struct.save_model()
            historyPlot(struct.history, "plots/accloss_{}_{}".format(struct.task, struct.name))
            counter+=1

if __name__ == '__main__':
    main()
