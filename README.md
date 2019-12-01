# meta-segmentation-msc-2018

MSc project Tom van Sonsbeek
# Contents
* Code to perform meta-learning on metafeautures. metafeaures can be extracted using the medical metafeatures ython package. Installation instructionscan be found in its repository: https://github.com/tjvsonsbeek/medical-mfe 
* Code of some of the (sub)projects 

# Project abstract
In medical imaging deep learning has led to state-of-the-art results for many tasks, such as segmentation of different anatomical structures in medical scans. With the increased numbers of deep learning publications and openly available code on platforms such as GitHub, the approach to choosing a model for a new task gets more complicated. Model choices increase, while time and (computational) resources are still limited. Faster model selection could enable a more efficient way of doing research. 

A possible solution to choosing a model is meta-learning, a learning method in which prior performance of a model is used to predict the performance for new tasks. Until now, meta-learning has been primarily used for standard machine learning datasets, and is relatively unknown in medical imaging. We investigate meta-learning for segmentation across ten datasets of different organs and modalities. We propose four ways to summarize each dataset into metafeatures; compressed representations of dataset used for learning the relationship between dataset and model performance. One of these metafeatures is based on statistical features of the images and three based on deep learning features. With support vector regression and a deep neural network the relationship between the metafeatures and prior model performance is learnt. On test datasets these methods yielded results between 0.10 from the true Dice score of model. These results show that meta-learning can predict performance of medical imaging methods and indicates that meta-learning can be of value in medical imaging. 


