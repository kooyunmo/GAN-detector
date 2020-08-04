import time
import matplotlib.pyplot as plt


def visualize(train_vis, valid_vis, epoch, save_dir):
    plt.plot(range(epoch), train_vis)
    plt.scatter(range(epoch), train_vis)
    plt.plot(range(epoch), valid_vis)
    plt.scatter(range(epoch), valid_vis)
    plt.savefig(save_dir + '/xception_loss.png')
