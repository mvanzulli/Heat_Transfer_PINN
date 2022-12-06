
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Module to load the data set """

__author__ = "Mauricio Vanzulli"
__email__ = "mvanzulli@fing.edu.uy"
__status__ = "Development"
__date__ = "12/22"

#Import built-in libraries

try:
    from load_data import CSVDatset, theta_transform, normtheta_transform
except:
    from src.data.load_data import CSVDatset, theta_transform, normtheta_transform

#Import third-party libraries

import numpy as np

import yaml

from torch.utils.data import  DataLoader

from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt

# read config file
with open('config_surrogate.yaml') as file:
    config = yaml.safe_load(file)

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
        label =  'G = %.4f' % G_sample

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

    plt.legend(loc='upper left', prop={'size': 6})
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
        label =  'G = %.4f' % G_sample

        if sample_picked not in samples_plotted:
            plt.plot(T_sample,label=label, marker = ".")
            samples_plotted.append(sample_picked)

    plt.xlabel('# $X*$')
    if T_sample.mean() <=10:
        plt.ylabel(r'$\theta$')
    else:
        plt.ylabel('$T(K)$')


    plt.legend(loc='upper left', prop={'size': 6})
    plt.savefig(config['data']['fig_path'] + name_fig)

def plot_G_vs_tx(data_loader, name_fig, num_samples = 10):
    """
    Plot G vs [T_1, ...] and [x_1].

    Args:
        data_loader (DataLoader): data loader to plot.
        name_fig(String)        : figure name.
        num_samples (Int)       : number of different samples to be shown.
    """
    samples_plotted = []

    ax = plt.figure().add_subplot(projection='3d')
    while len(samples_plotted) < num_samples:

        sample_picked = np.random.randint(0, len(data_loader))
        sample_values = data_loader.dataset[sample_picked].to_numpy()
        G_sample = sample_values[0]
        T_sample = sample_values[1:]
        X_sample = np.linspace(0,1,T_sample.shape[0])
        label =  'G = %.4f' % G_sample

        if sample_picked not in samples_plotted:
            ax.plot(X_sample, T_sample, G_sample, label=label, marker = ".")
            ax.plot(X_sample, T_sample, np.multiply(0,G_sample), color = 'grey',  marker = ".")
            ax.plot(np.multiply(0,X_sample), T_sample, G_sample, color = 'grey',  marker = ".")
            samples_plotted.append(sample_picked)

    ax.set_xlabel('$X*$')
    ax.set_zlabel('$G$')
    if T_sample.mean() <=10:
        ax.set_ylabel(r'$\theta$')
    else:
        ax.set_ylabel('$T(K)$')

    ax.set_xlim(X_sample[0], X_sample[-1])
    ax.set_zlim(0, 1)
    # ax.legend(loc='upper left', prop={'size': 6})
    plt.savefig(config['data']['fig_path'] + name_fig)
    plt.show()


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
    plot_G_vs_tx(data_loader, namefig + '_x_vs_G' + str(num_samples) + '_sample', num_samples )

if __name__ ==  "__main__":

    ns = 30
    plot_temperature('theta', transform = theta_transform, num_samples = ns)
    plot_temperature('T', transform = None, num_samples = ns)
    plot_temperature('theta_norm', transform = normtheta_transform, num_samples = ns)
