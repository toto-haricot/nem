import cv2
import time
import torch
import scipy.stats

import numpy as np
import pandas as pd

from utils import *
from utils_brisque import *
from torch.utils.data import Dataset, DataLoader

class DatasetNoise(Dataset):

    def __init__(self, data:list, type_='train'):
        
        self.image_paths_and_exposure = data
        self.type_ = type_
        
        self.n_images = len(self.image_paths_and_exposure)

    def __getitem__(self, index):

        # t1 = time.time()

        image_path = self.image_paths_and_exposure[index][0]
        image_exposure = self.image_paths_and_exposure[index][1]
        image_exposure = np.array(image_exposure).reshape(1,1)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # t2 = time.time()
        # time_reading = round(t2-t1, 4)

        image = central_crop(image, 900, 600)

        # t3 = time.time()
        # time_cropping = round(t3-t2, 4)

        image_noisy, noise_type, noise_value = create_noise(image)
        noise_score = create_noise_score_sigmoid(noise_type, noise_value)

        # t4 = time.time()
        # time_noise_creation = round(t4-t3, 4)

        # brsq_ftrs = brisque(image_noisy)
        kurt_ftrs = compute_kurtosis(discrete_cosine_transform(image_noisy, block_size=8))
        features = np.concatenate((image_exposure, kurt_ftrs), axis=1)
        features = torch.tensor(features)

        # t5 = time.time()
        # time_computing_features = round(t5-t4, 4)

        if self.type_  == 'train' : 
            return features, noise_type, noise_score#, time_reading, time_cropping, time_noise_creation, time_computing_features

        elif self.type_  == 'test' : 
            return features, image_path, noise_type, noise_value, noise_score


    def __len__(self):

        return self.n_images