#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module to load the data set """

__author__ = "Mauricio Vanzulli"
__email__ = "mvanzulli@fing.edu.uy"
__status__ = "Development"
__date__ = "12/22"


#Import third-party libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

import pandas as pd

import yaml

import numpy as np

from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2


# read config file
with open('config_surrogate.yaml') as file:
    config = yaml.safe_load(file)


class CSVDatset(Dataset):
    """CSV dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file.
        """
        super(CSVDatset, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.root = csv_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index,1:]
        return sample

    def rand_split(self, split_ratio = config['data']['split_ratio'], seed_id =config['data']['manual_seed'] ):
        """
        Args:
            split_ratio (string): Path to the csv file.
            seed_id (num): Seed number.
        """

        train_set_size = int(len(self) * (1-split_ratio))
        valid_set_size = len(self) - train_set_size

        train_set, test_set = random_split(
            self,
            [train_set_size, valid_set_size],
            generator=torch.Generator().manual_seed(seed_id),
        )

        return train_set, test_set


def show_samples(data_loader, num_samples = 10):

    samples_ploted = []

    while len(samples_ploted) <= num_samples:
        sample_picked = np.random.randint(0, len(data_loader))

        sample_values = data_loader.dataset[sample_picked].to_numpy()
        G_sample = sample_values[0]
        T_sample = sample_values[1:]
        label =  'G = %.2f' % G_sample
        if sample_picked not in samples_ploted:
            plt.plot(T_sample,label=label)
            samples_ploted.append(sample_picked)

    plt.xlabel('# $X*$')
    plt.ylabel('$T(K)$')
    plt.legend(loc='upper left', prop={'size': 6})
    plt.show()


dataset = CSVDatset(config['data']['csv_path'])
train_set,test_set = dataset.rand_split()
data_loader = DataLoader(train_set, batch_size=config['data']['batch_size'], num_workers=4)
show_samples(data_loader)
print(len(train_set))

print(len(test_set))
