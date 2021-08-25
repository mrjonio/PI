# title                 :data_loader.py
# description           :It manages the tomographic datasets that will feed the Deep Learning models
# author                :Dr. Luis Filipe Alves Pereira (luis.filipe@ufrpe.br or luisfilipeap@gmail.com)
# date                  :2019-08-05
# version               :1.0
# notes                 :Please let me know if you find any problem in this code
# python_version        :3.6
# numpy_version         :1.16.3
# scipy_version         :1.2.1
# pandas_version        :0.24.2
# pytorch_version       :1.1.0
# ==============================================================================


import pandas as pd
import numpy as np
import scipy.misc
import random
import os
import skimage.io as io

from data_utils import data_mean_value
import torch
from torch.utils.data import Dataset


"""
Parameters to manage the tomographic datasets 

original_src:   directory of the high quality images. It will be passed directly to the batch for future comparisons with the proposed solution.  
train_dim:      dimensions (h,w) of the cropped files used in the training stage. h and w must be smaller than the size of the original images.

note: please see the comment above the attribute self.data when the phase is 'train' in the constructor of Tomographic_Dataset
"""

original_src = 'C:\\LInux\\DATASET-256-CORONAL\\'
train_dim   = (128,128)


class Tomographic_Dataset(Dataset):

    #directory of training files is passed to obtain the mean value of the images in the trained set which is not trained in the CNN
    def __init__(self, csv_file, phase, input_dir, target_dir, crop=False, flip_rate=0., train_csv = ""):
        self.data      = pd.read_csv(csv_file)
        self.means     = data_mean_value(train_csv, input_dir) / 255.
        self.input_dir = input_dir
        self.target_dir= target_dir
        self.n_class   = 2

        self.flip_rate = flip_rate
        self.crop      = crop
        if phase == 'train':
            #self.crop = True
            self.flip_rate = 0.25
            self.new_h = train_dim[0]
            self.new_w = train_dim[1]
            #The training set is replicated 10 times when the dataset used has only 909 images. Since each training sample is randomly cropped in windows h/2 and w/2,
            #it should be no repetition in the set of cropped samples usaed for training
            #self.data = pd.concat([ self.data], ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label_name = self.data.iloc[idx, 1]

        if(random.randrange(10) > 4):
            img        = io.imread(self.input_dir+img_name, pilmode='F')
            label      = 0
        else:
            img = np.load(self.target_dir+label_name)
            label      = 1


        (_, file) = os.path.split(label_name)
        original   = original_src+file[0:len(file)-3]+'png'

        # reduce mean
        img = img / 255. - self.means



        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        #target = torch.IntTensor(label)
        #label = img-label

        # create one-hot encoding
        h, w = img.size()
        input_tensor = torch.zeros(1, h, w)
        target = np.zeros(1)
        for c in range(self.n_class):
            target[0] =  label

        target = torch.from_numpy(target.copy()).long()



        for H in range(h):
            for W in range(w):
                input_tensor[0,H,W] = img[H,W]


        sample = {'X': input_tensor, 'Y': target, 'l': label, 'o': original, 'file': img_name}
        return sample




