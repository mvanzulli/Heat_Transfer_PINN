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
from pandas.core.series import Series
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

    def __init__(self, csv_file, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file.
            transform (optional func): Optional transformation to perform on a sample.
        """
        super(CSVDatset, self).__init__()
        self.data = pd.read_csv(csv_file)
        self.root = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index,1:]

        if self.transform:
            return self.transform(self.data, sample)

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

def get_labels(config):
    LABELS_T = config['data']['feature_labels']
    LABEL_G = config['data']['target_label']
    T_B = config['data']['T_B']
    T_INF = config['data']['T_INF']

    return LABEL_G, LABELS_T, T_B, T_INF

def theta_transform(data, sample):
    """
    Computes theta for a data frame sample with a given data frame.
    Args:
        data (DataFrame): raw data frame.
        sample (Sample): data frame sample.
    """

    LABEL_G, LABELS_T, T_B, T_INF = get_labels(config)

    T_max = data.max()
    T_max = np.array([T_max[T_label] for T_label in LABELS_T])
    T_min = data.min()
    T_min = np.array([T_min[T_label] for T_label in LABELS_T])
    T_sample = np.array([sample[T_label] for T_label in LABELS_T])

    Theta_sample = np.multiply(np.add(T_sample, [-T_INF]), [1/(T_B - T_INF)])

    sample = np.concatenate((sample[LABEL_G],Theta_sample), axis=0)

    return Series(sample)

def normtheta_transform(data, sample):

    """
    Computes theta for a data frame sample with a given data frame.
    Args:
        data (DataFrame): raw data frame.
        sample (Sample): data frame sample.
    """

    LABEL_G, LABELS_T, T_B, T_INF = get_labels(config)

    sample_id = sample.name
    G_sample = sample[LABEL_G]

    theta_df = (data - T_INF)/(T_B - T_INF)
    theta_sample = theta_df.iloc[sample_id]
    theta_sample = np.array([theta_sample[T_label] for T_label in LABELS_T])

    theta_max = theta_df.max()
    theta_max = np.array([theta_max[T_label] for T_label in LABELS_T])
    theta_min = theta_df.min()
    theta_min = np.array([theta_min[T_label] for T_label in LABELS_T])

    theta_sample_norm = 2 * np.divide((theta_sample[1:] - theta_min[1:]) , (theta_max[1:] - theta_min[1:])) - 1

    sample = np.concatenate((G_sample,theta_sample_norm), axis=0)

    return Series(sample)


def plot_t_vs_G(data_loader, name_fig, num_samples = 10):
    """
    Plots T(x) for different G samples.

    Args:
        data_loader (DataLoader): data loader to plot.
        name_fig(String)        : figure name.
        num_samples (Int)       : number of different samples to be shown.
    """
    samples_plotted = []

    plt.cla()

    for i in range (num_samples):

        sample_picked = np.random.randint(0, len(data_loader))
        # sample_picked = i
        # print(sample_picked)
        sample_values = data_loader.dataset[sample_picked].to_numpy()
        # [G, T_1 ..., T__9]
        G_sample = sample_values[0]
        T_sample = sample_values[1:]
        # auxiliary array of ones to plot G
        aux_ones = np.ones((T_sample.shape[0],))
        label =  'G = %.2f' % G_sample

        if sample_picked not in samples_plotted:
            Y_plot = G_sample * aux_ones
            plt.scatter(
                T_sample, Y_plot,
                label=label, marker=".", s=40,
            )
            samples_plotted.append(sample_picked)

    plt.ylabel('$G$')
    if T_sample.mean() <=10:
        plt.xlabel(r'$(\theta_1,...,\theta_9)$')
    else:
        plt.xlabel(r'$(T_1,...,T_9)$')

    plt.legend(loc='best', prop={'size': 6})
    plt.savefig(config['data']['fig_path'] + name_fig)


    return None

def plot_t_vs_x(data_loader, name_fig, num_samples = 10):
    """
    Plots T(x) for different G samples.

    Args:
        data_loader (DataLoader): data loader to plot.
        name_fig(String)        : figure name.
        num_samples (Int)       : number of different samples to be shown.
    """
    samples_plotted = []

    plt.cla()
    while len(samples_plotted) < num_samples:

        sample_picked = np.random.randint(0, len(data_loader))
        sample_values = data_loader.dataset[sample_picked].to_numpy()
        G_sample = sample_values[0]
        T_sample = sample_values[1:]
        label =  'G = %.2f' % G_sample

        if sample_picked not in samples_plotted:
            plt.plot(T_sample,label=label, marker = ".")
            samples_plotted.append(sample_picked)

    plt.xlabel('# $X*$')
    if T_sample.mean() <=10:
        plt.ylabel(r'$\theta$')
    else:
        plt.ylabel('$T(K)$')


    plt.legend(loc='best', prop={'size': 6})
    plt.savefig(config['data']['fig_path'] + name_fig)


def plot_temperature(namefig, transform, num_samples = 10):

    # load data set with theta_transform
    dataset = CSVDatset(config['data']['csv_path'], transform=transform)
    # split them into train and test
    train_set, test_set = dataset.rand_split()

    data_loader = DataLoader(
        train_set, batch_size=config['data']['batch_size'],
        num_workers=4,
    )

    plot_t_vs_x(data_loader, namefig + 'vs_x_' + str(num_samples) + '_sample', num_samples)
    plot_t_vs_G(data_loader, namefig + 'vs_G_' + str(num_samples) + '_sample', num_samples)




if __name__ == "__main__":

    # plot_temperature('theta', transform = theta_transform, num_samples = 100)
    # plot_temperature('T', transform = None, num_samples = 100)
    # plot_temperature('theta_norm', transform = normtheta_transform, num_samples = 5)

    transform = None

    # load data set with theta_transform
    dataset = CSVDatset(config['data']['csv_path'], transform=transform)
    # split them into train and test
    train_set, test_set = dataset.rand_split()

    data_loader = DataLoader(
        train_set, batch_size=config['data']['batch_size'],
        num_workers=4,
    )
