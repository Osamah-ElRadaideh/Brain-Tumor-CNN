# Brain-Tumor-CNN
A CNN for classifying brain tumors, with evaluation script and a script for evaluating user images

to train or evaluate, download the dataset from https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset and place it in D:/Datasets folder, or change the location of training_test.json and testing_set.json
the first 350 examples of the testing set has been used as a validation set, while the evaluation runs on the remaining examples
Training for 250 epochs results in a test accuracy of 95.73%


requirements: 
einops==0.6.1
lazy_dataset==0.0.14
numpy==1.23.5
opencv_python==4.7.0.72
torch==1.13.1+cu117
tqdm==4.64.1
