import networks
from utils import joined_tSNE_plot
def main():
    tasks_list= [ 'Task09_Spleen', 'Task10_Colon']
# 'Task02_Heart','Task03_Liver','Task04_Hippocampus','Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel',
    feature_extractors = ['VGG16', 'ResNet50', 'MobileNetV1']
    fe = 'ResNet50'
    task = 'Task09_Spleen'
    print("FEATURE EXTRACTOR: {}".format(fe))
    struct = networks.encoderDecoderNetwork(fe, 2)
    struct.task = task
    struct.imageDimensions = (224,224)
    struct.minibatch_size = 5
    struct.epochs = 10
    struct.train_size = 1000
    struct.val_size = 100

    struct.load_training_data()
    struct.subset_training_data()
    struct.load_valid_data()
    struct.subset_valid_data()
    # struct.create_train_generator()
    # struct.create_val_generator()
    # struct.add_callback(ModelCheckpoint(filepath='best_model_{}_{}.h5'.format(struct.name, struct.id), verbose=2, save_best_only=True, save_weights_only = True, period = 1))
    # struct.add_callback(EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0))
    struct.build_encoder()

    struct.build_decoder()
    struct.load_weights()
    struct.update_encoder_weights()
    result = struct.predict(struct.feature_extractor, struct.val_data)
    joined_tSNE_plot(result)
if __name__ == '__main__':
    main()
