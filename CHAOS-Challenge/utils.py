import os
import pydicom
import numpy as np
import cv2
import random
from scipy.ndimage import zoom

def train_generator(batch_size):
    """
    read batch size number of images
    """
    while True:
        images = np.zeros((batch_size, 64, 128, 128,1))
        labels = np.zeros((batch_size, 64, 128, 128,1))
        for batch in range(batch_size):
            dataset = random.choice(['CT_data_batch1', 'CT_data_batch2'])
            nr = random.choice(os.listdir('/home/tjvsonsbeek/chaosData/{}'.format(dataset)))
            nr_of_slices = len(os.listdir('/home/tjvsonsbeek/chaosData/{}/{}/DICOM_anon'.format(dataset, nr)))
            addresses_im = os.listdir('/home/tjvsonsbeek/chaosData/{}/{}/DICOM_anon'.format(dataset, nr))
            addresses_la = os.listdir('/home/tjvsonsbeek/chaosData/{}/{}/Ground'.format(dataset, nr))
            dcm_base = '/home/tjvsonsbeek/chaosData/{}/{}/DICOM_anon/{}'.format(dataset, nr, addresses_im[0])
            lab_base = '/home/tjvsonsbeek/chaosData/{}/{}/Ground/{}'.format(dataset, nr, addresses_la[0])
            total_im = np.zeros((512,512,len(range(int(nr_of_slices*0.30), int(nr_of_slices*0.70)))))
            total_la = np.zeros((512,512,len(range(int(nr_of_slices*0.30), int(nr_of_slices*0.70)))))
            for slice in range(int(nr_of_slices*0.30), int(nr_of_slices*0.70)):
                # print(dcm_base[-18:-15])
                if dcm_base[-18:-15] == 'IMG':
                    dcm_address = dcm_base[:-9]+str(slice + 1).zfill(5)+'.dcm'
                    lab_address = lab_base[:-7]+str(slice + 1).zfill(3)+'.png'
                else:
                    dcm_address = dcm_base[:-14]+ str(slice).zfill(4)+ ',0000b.dcm'
                    lab_address = lab_base[:-7]+str(slice).zfill(3)+'.png'


                # print(dcm_address)
                # print(lab_address)
                dcim = pydicom.dcmread(dcm_address)
                im = dcim.pixel_array*dcim.RescaleSlope + dcim.RescaleIntercept
                im[im<0]=0

                la = cv2.imread(lab_address, cv2.IMREAD_GRAYSCALE)
                # cv2.imwrite('test{}.png'.format(slice), im)
                # cv2.imwrite('lab{}.png'.format(slice), la)
                total_im[:,:,slice-int(nr_of_slices*0.30)] = im
                total_la[:,:,slice-int(nr_of_slices*0.30)] = la
            # print("SFSG")

            #rescale_im = cv2.resize(total_im, (128,128,64))
            # rescale_la = cv2.resize(total_la, (128,128,64))

            rescale_im = zoom(total_im, (0.25, 0.25, float(64/len(range(int(nr_of_slices*0.30), int(nr_of_slices*0.70))))))
            rescale_la = zoom(total_la, (0.25, 0.25, float(64/len(range(int(nr_of_slices*0.30), int(nr_of_slices*0.70))))))
            # print("SFSTG")
            rescale_im -= np.min(rescale_im)
            rescale_la -= np.min(rescale_la)

            rescale_im = np.swapaxes(rescale_im, 0, 2)*(255/(np.max(rescale_im)))
            rescale_la = (np.swapaxes(rescale_la, 0 ,2)*(1/(np.max(rescale_la)))>0.5).astype(np.uint8)

            images[batch, :,:,:,0] = rescale_im
            labels[batch, :,:,:,0] = rescale_la
            # print("ok")
        yield images, labels
