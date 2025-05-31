Note in 2025: 
I have been attempting to improve on this experiment to see if the lack of improvement in classification performance was a bug in the code or if it was the FMRI data
and have upgraded to a more readable tf.keras implementation of the network. From what I've found so far is that the CORNETz implementation (basically alexnet) is 
correct, and the pairwise representation similarity auxillary loss function is also correct and if I use a pretrained CNN as the teacher for the similarity targets and 
pretrain a student network with only the representational similarity loss it subsequently learns the classification task faster than the teacher network. This strongly 
suggests that something about the FMRI reprepresentational similarities must be re-evaluated, or maybe a different FMRI dataset should be used...
Currently I don't have free GPU access and Tensorflow-metal has a bug at the moment that makes it unusable with my M4 mac mini so I will have to update with later when I find affordable compute
to train a signiicant number of models. 

Also this repo was based on a repo from Cassidy Pirlot who was extending the project with CCA, and prior Sahir was working on the project extended by adding PCA on the neural recordings. The original paper this work extends is Feder et. al 2019. The work I did ontop of this was update to tf2.0, added remote monitoring of training, preprocessed FMRI data from a human dataset, and replicated the prior experiements with this human dataset instead of using neural recordings from the visual cortex anesthetized monkeys. 

# FMRI_Similarity_CNN
 For my first psychology undergraduate research project. The corresponding paper is "independent_study.pdf"
 Using either representational similarity analysis or canonical correlation analysis with fmri data to try improve object recognition in a CNN

## Installing
- pip install tensorflow 2, scipy, pandas, numpy and optionally comet_ml
- there may be other dependencies needed, until a requirements.txt is added manual search is needed

## Data
### CIFAR100 Images
- CIFAR100 image data is used for classification and is stored separately from this repo split into training and testing directories
- train_directory.tgz and test_directory.tgz needs to be copied from the google drive into the experiment folder (CORNetZ_{cca/rsa}) and exctracted

### Stimulus Images
 These images were shown to the subjects while the brain data was being recorded. There are two compressed directories, stimuli.tgz, and V1.tgz. 

#### Stimuli
 These images are used as stimuli during the human fmri recordings, so they are relevant for experiments using the fmri data.

#### V1
 This directory contains a directory containing "natural images" which are images shown to monkey subjects while direct neuron recordings are taken from neurons in V1 of the occipital cortex.

stimuli.tgz and v1.tgz need to be copied from the google drive and decompressed outside of the experiment directory in it's parent directory. 

## Experiment Folders
 These include CORNetZ_cca and CORNetZ_rsa, the former uses canonical correlation analysis (CCA) in the composite cost function and the latter uses representational similarity matricies that depend on cosine distance as the similarity metric. 

### CORNetZ_cca 
 #### Working Scripts:
 - cca_cornetZ_ratio.py
    - working script to run experiment
 - CORNetZ_V.py
    - defines the CNN model
 - custom_cca_2.py
    - does cca for cca_cornetZ_ratio.py
 - datagenerator.py
    - used to feed batches of CIFAR100 images to CNN
 - datagenerator_v_pca.py
    - used to feed batches of neural data from a numpy file and it's associated stimuli images to the CNN
 - label_maps.py 
    - used to calculate coarse accuracy(accuracy within superclass) given the fine accuracy(actual classification performance)
 #### Scripts in progress
 - cca_cornetZ_ratio_cdp.py
    - uses a different cca based method for improving cnn performance
 - CCA_cdp.py
    - performs cca for the above script

 #### Data
 - ROI_fmri_pca16.npy
    - the ROI are ventral temporal areas involved in object classification
    - preprocessed fmri data in a 6x84x16 shaped array
    - this is arranged by 6 subjects, 84 stimuli images, 16 principle components of fmri data
 - ROI_fmri.npy
    - the same neural data as above except pca isn't applied, so it is shaped 6x84x2294 where 2294 are the number of voxels in the ROI
 - neural_data_pca.npy
    - monkey v1 data (preprocessing not included) for cca method
    - shaped 10x956x16, which is 10 subjects, 956 images, and 16 principle components
 - ROI_train_cca.txt
    - contain's the paths to stimuli images in the stimuli directory which is in the parent directory of the experiment folder
    - beside each path there is also an integer representing the index of the image axis (second axis) of the ROI_*.npy file which contains neural data associated with viewing this stimulus
 - train.txt
    - contains paths to CIFAR100 images in the train_directory (which needs to be added from the google drive)
 - val.txt
   - contains paths to CIFAR100 images in the test_directory (which needs to be added from the google drive)

 #### Running CCA Experiment
 to run an experiment simply run the following with the desired arguments as follows:
 ```
 python cca_cornetZ_ratio.py --{insert arg}
 ```
 The nessesary and optional arguments and their uses can be examined in the python file 

### CORNetZ_rsa
 #### Working Scripts
 - run_cornetZ_ratio.py
    - used to run an experiment using the rsa method
 - CORNetZ_V.py
    - defines the CNN model
 - datagenerator_v.py
    - feeds batches of cosine distances and their associated pairs of stimulus images to the CNN
 - datagenerator.py
    - feeds batches of CIFAR100 images to the CNN
 - label_maps.py
    - used to calculate coarse accuracy(accuracy within superclass) given the fine accuracy(actual classification performance)
 #### Data
 - ROI_train_distance.txt
    - each line contains two stimulus image paths and the cosine distance between the fmri data recorded while viewing the images which is averaged across subjects
    - the image paths point to the stimuli directory in the parent directory
 - V1_train.txt
    - similar to the above file except produced from v1 neural recordings from monkey subjects
    - the images are located in the V1 directory in the parent directory
 - train.txt
    - contains paths to CIFAR100 images in the train_directory (which needs to be added from the google drive)
 - val.txt
   - contains paths to CIFAR100 images in the test_directory (which needs to be added from the google drive)

 #### Running RSA experiments
  to run an experiment simply run the following with the desired arguments as follows:
 ```
 python run_cornetZ_ratio.py --{insert arg}
 ```
 The nessesary and optional arguments and their uses can be examined in the python file 
