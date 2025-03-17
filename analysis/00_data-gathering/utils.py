# -*- coding: utf-8 -*-

"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
# import matplotlib.pyplot as plt
import os, gzip

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import tensorflow.contrib.slim as slim
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler


num_cells_train = 6605 # cell이 gex
noise_input_size = 100


def load_epidermal(dataset_name, gex_size):
    # dataset_name = dataset_name

    data_dir = os.path.join("../Training_Dataprep/Results/training_data", dataset_name)
    print(data_dir)
    set_name = "%s_aug_train.csv" % dataset_name
    path = os.path.join(data_dir, set_name)
    input_ltpm_matrix = genfromtxt(path, delimiter=',', skip_header=1)
    scaler = MinMaxScaler()
    input_ltpm_matrix = np.transpose(input_ltpm_matrix)
    scaler.fit(input_ltpm_matrix)

    input_ltpm_matrix = scaler.transform(input_ltpm_matrix)
    input_ltpm_matrix = np.transpose(input_ltpm_matrix)
    input_ltpm_matrix = np.transpose(input_ltpm_matrix)

    print(np.ptp(input_ltpm_matrix, axis=1))
    print(np.ptp(input_ltpm_matrix, axis=1).shape[0])

    input_ltpm_matrix_unwitheld = input_ltpm_matrix
    input_ltpm_matrix = input_ltpm_matrix[:,0:gex_size]
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(input_ltpm_matrix)
    return input_ltpm_matrix, input_ltpm_matrix_unwitheld
    # return input_ltpm_matrix_unwitheld, input_ltpm_matrix

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# def show_all_variables():
#    model_vars = tf.trainable_variables()
#    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
