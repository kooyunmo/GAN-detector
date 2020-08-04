import os
import time
from collections import OrderedDict
import matplotlib as mp
import matplotlib.pyplot as plt


def visualize(model_name, train_loss, valid_loss, train_acc, valid_acc, epochs):
    font_size = 10
    mp.rcParams['font.family'] = 'serif'
    mp.rcParams['font.size'] = font_size-1
    mp.rcParams['xtick.labelsize'] = font_size+1
    mp.rcParams['ytick.labelsize'] = font_size+1
    mp.rcParams['axes.labelsize'] = font_size+5
    mp.rcParams['legend.fontsize'] = font_size
    mp.rcParams['legend.frameon'] = False

    plt.title(model_name + ': training log', fontsize=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss or Acc (%)', fontsize=12)
    plt.grid()

    plt.plot(epochs, train_loss)
    plt.scatter(epochs, train_loss, c='blue', label='train loss')
    plt.plot(epochs, valid_loss)
    plt.scatter(epochs, valid_loss, c='cornflowerblue', label='valid loss')
    plt.plot(epochs, train_acc)
    plt.scatter(epochs, train_acc, c='red', label='train acc')
    plt.plot(epochs, valid_acc)
    plt.scatter(epochs, valid_acc, c='coral', label='valid acc')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    lgnd = plt.legend(by_label.values(), by_label.keys(), loc='upper left')
    for handle in lgnd.legendHandles:
        handle.set_sizes([50])
    
    plt.savefig(model_name + '_training_log.png')
