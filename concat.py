import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


losses = []
val_loss = []
log_dir    = os.path.join('E:\code\classification-pytorch\classification-pytorch\logs', "concat" )




def loss_plot():
    iters = range(len(losses))

    plt.figure()
    plt.plot(iters, losses, 'red', linewidth=2, label='train loss')
    plt.plot(iters, val_loss, 'coral', linewidth=2, label='val loss')
    try:
        if len(losses) < 25:
            num = 5
        else:
            num = 15

        plt.plot(iters, scipy.signal.savgol_filter(losses, num, 3), 'green', linestyle='--', linewidth=2,
                 label='smooth train loss')
        plt.plot(iters, scipy.signal.savgol_filter(val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                 label='smooth val loss')
    except:
        pass

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(log_dir, "epoch_loss.png"))

    plt.cla()
    plt.close("all")

if __name__ == '__main__':
    with open(os.path.join(log_dir, "epoch_loss.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            loss = float(line.strip())
            losses.append(loss)
            # print(line.strip())
    losses.pop()
    print(losses)
    with open(os.path.join(log_dir, "epoch_val_loss.txt"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            val = float(line.strip())
            val_loss.append(val)
            # print(line.strip())
    print(val_loss)
    loss_plot()