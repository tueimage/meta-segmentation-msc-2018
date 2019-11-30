import os
import cv2
import numpy as np
from tqdm import tqdm
tasks_list=  ['Task01_BrainTumour','Task02_Heart','Task03_Liver','Task04_Hippocampus', 'Task05_Prostate', 'Task06_Lung', 'Task07_Pancreas', 'Task08_HepaticVessel', 'Task09_Spleen', 'Task10_Colon']

def main():
    for task in tasks_list:
        counts = []
        for address in tqdm(os.listdir(r'/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/train_{}/labels'.format(task))):
            im = cv2.imread(r'/home/tjvsonsbeek/featureExtractorUnet/decathlonDataProcessed/train_{}/labels/'.format(task)+address,0)
            count = np.count_nonzero(im)
            counts.append(count)
        final_count = np.mean(counts)
        std_count = np.std(counts)
        coef_var = std_count/final_count
        print('{}:      {}      {}      {}'.format(task, final_count, std_count, coef_var))

if __name__ == '__main__':
    main()
