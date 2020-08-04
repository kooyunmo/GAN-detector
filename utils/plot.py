import os
import time
import matplotlib.pyplot as plt


def visualize(train_vis, valid_vis, epoch_vis, save_dir):
    plt.plot(epoch_vis, train_vis)
    plt.scatter(epoch_vis, train_vis)
    plt.plot(epoch_vis, valid_vis)
    plt.scatter(epoch_vis, valid_vis)
    plt.savefig(os.path.join(save_dir, 'xception_loss.png'))
