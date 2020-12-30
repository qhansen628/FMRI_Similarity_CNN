# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

# from tensorflow.contrib.data import Dataset
Dataset = tf.compat.v1.data.Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import os


# IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
tf.compat.v1.disable_eager_execution()

class ImageDataGeneratorV2(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, npy_file, mode, batch_size, img_size=227,
                 buffer_size=1000):

        self.npy_file = npy_file
        self.txt_file = txt_file

        # retrieve data from npy & images
        self._read_fmri_npy_file()

        self.img_size = img_size

        self._extend_data()

        # number of samples in the dataset
        self.data_size = len(self.img_paths)
        self.batch_size = batch_size

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        # print(self.neurons.shape)
        print(np.array(self.neurons).shape)
        self.neurons = convert_to_tensor(np.array(self.neurons))

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.neurons))

        data = data.map(self._parse_function_train)

        data = data.batch(batch_size,drop_remainder=True)

        self.data = data

    def _read_npy_file(self):
        """Read the content of the npy file and store it into lists."""
        self.img_paths = []
        self.neurons = []

        data = np.load(self.npy_file)

        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            i, j = 0, 0
            for line in lines:
                if i == 956:
                    i = 0
                    j = j + 1
                line = line.rstrip('\n')
                self.img_paths.append(line)
                self.neurons.append(data[j][i])
                i = i + 1
    
    def _read_fmri_npy_file(self):
        """Read the content of the npy file and store it into lists."""

        data = np.load(self.npy_file)

        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            self.img_paths = [line.strip() for line in lines]
        self.neurons = np.vstack(data)
        assert len(self.img_paths) == self.neurons.shape[0]

    def _extend_data(self):
        times = 3
        img_paths = self.img_paths.copy()
        neurons = self.neurons.copy()

        for i in range(times):
            self.img_paths.extend(img_paths)
            self.neurons = np.append(self.neurons,neurons,0)

    def _parse_function_train(self, filename1, data):
        """Input parser for samples of the training set."""
        # load and preprocess the image
        img1_string = tf.io.read_file(filename1)
        img1_decoded = tf.image.decode_png(img1_string, channels=3)
        img1_resized = tf.image.resize(img1_decoded, [int(self.img_size), int(self.img_size)])

        # check on below:
        """
        Dataaugmentation comes here.
        """
        # RGB -> BGR
        img_bgr1 = img1_resized[:, :, ::-1]

        return img_bgr1, data

    def _file_to_img(self, filename):
        # load and preprocess the image
        img_string = tf.io.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize(img_decoded, [32, 32])
        # img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_resized[:, :, ::-1]

        return img_bgr
