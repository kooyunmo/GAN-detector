import os
import time
import matplotlib.pyplot as plt


def visualize(model_name, train_loss, valid_loss, train_acc, valid_acc, epochs):
    plt.plot(epochs, train_loss)
    plt.scatter(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.scatter(epochs, valid_loss)
    plt.plot(epochs, train_acc)
    plt.scatter(epochs, train_acc)
    plt.plot(epochs, valid_acc)
    plt.scatter(epochs, valid_acc)
    plt.savefig(model_name + '_training_log.png')
