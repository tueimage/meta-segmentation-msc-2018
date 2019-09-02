# from meta_network import meta_learner
from data import Data, MetaData
from utils import subset_index_to_address
from utils import meta_pred_generator, historyPlot, create_data_subsets, dice_coef_loss, auc, mean_iou, dice_coef
from keras.optimizers import Adam
from meta_network import meta_learner
from networks import EncoderDecoderNetwork
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():
    tasks_list=  ['Task10_Colon','Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus','Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel']
# ,'Task09_Spleen',
    feature_extractors = ['VGG16', 'ResNet50', 'MobileNetV1']
    meta_data = MetaData('t', 's')
    for task in tasks_list:
        # try:
            # data = Data(task)
            #
            # data.train_size = 1000
            # data.val_size = 100
            # data.imageDimensions = (224, 224)
            # meta_subset_size = 5
            # nr_of_meta_subsets = 5
            #
            # data.load_training_data()
            # data.load_valid_data()
            # data.get_meta_subsets(nr_of_meta_subsets, meta_subset_size)

        for fe in feature_extractors:
            meta_inter = MetaData(task, fe)
            try:
                meta_inter.load()
                for x in range(5):

                    meta_data.total_addresses.append(meta_inter.addresses[x])
                    meta_data.total_results.append(meta_inter.results[x])
            except:
                print("oei")

            # meta_data = MetaData(task, fe)
            # struct = EncoderDecoderNetwork(fe, 2)
            # struct.task = task
            # struct.minibatch_size = 5
            # struct.epochs = 10
            # struct.imageDimensions = (224, 224)
            # struct.build_encoder()
            # struct.build_decoder()
            # struct.load_weights()
            # struct.model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics = ['accuracy', auc, mean_iou, dice_coef])
            # for subset in range(data.meta_subsets.shape[0]):
            #     addresses = subset_index_to_address(data.meta_subsets[subset, :], data.train_data)
            #     meta_data.addresses.append(addresses)
            #     result = struct.get_meta_data(addresses)[2:]
            #     print(result)
            #     meta_data.results.append(result)
            #     meta_data.save()
    print(meta_data.total_addresses)
    print(meta_data.total_results)
    print(len(meta_data.total_results))
        # except:
        #     print("MASSIVE FAIL")
    meta_model = meta_learner('VGG16')
    meta_model.build_feature_extractor()
    meta_model.build_meta_model()
    meta_model.train(meta_data.total_addresses, meta_data.total_results, (224, 224), 1)
    # historyPlot(meta_model.history, "testmeta.png")
    # meta_model.save_model()
    # meta_model.load_weights()
    # for x in range(30):
    #     pred = meta_model.model.predict_generator(meta_pred_generator(meta_data.total_addresses[x], meta_model.minibatch_size, (224,224)), steps = 1)
    #     print("pred: {}".format(pred))
    #     print("result: {}".format(meta_data.total_results[x]))


if __name__ == '__main__':
    main()
