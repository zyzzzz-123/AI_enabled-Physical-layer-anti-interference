import csv
from _csv import reader
from math import sqrt
import random

import numpy
import pandas as pd
import numpy as np
import torch

# Energy Constraints
from numpy import array


def energy_constraint(x, args):
    # Energy Constraint
    n = (x.norm(dim=-1)[:, None].view(-1, 1).expand_as(x))
    x = sqrt(args.n_channel) * (x / n)
    return x


# def awgn(x, args, device):
#     SNR = 10 ** (args.EbN0_dB_train / 10)
#     R = args.n_source / args.n_channel
#     noise = torch.randn(x.size(), device=device) / ((2 * R * SNR) ** 0.5)
#     noise.to(device)
#     x += noise
#     return x

def awgn(x, args, device):
    SNR = 10 ** (args.EbN0_dB_train / 10)
    # R = args.n_source / args.n_channel
    input = x.detach()
    x_power = torch.sum(input ** 2) / input.numel()
    noise_power = x_power / SNR
    noise = torch.randn(x.size(), device=device) * (noise_power ** 0.5)
    noise.to(device)
    x += noise
    return x


def wlan(x, args, device):
    file = pd.read_csv("/home/zyz/python_project/project_14_autoencoder/WLAN.csv")
    with open("/home/zyz/python_project/project_14_autoencoder/WLAN.csv",
              'r') as csv_file:
        csv_reader = reader(csv_file)
        wifi = list(csv_reader)
    input = x.detach()
    [m, l] = input.shape
    num = int(m * l * 0.5)  # the number of complex needed
    length = len(wifi)
    total_wlan = []
    for i in range(num):
        s = random.randint(0,3599)
        b = wifi[s]
        n = b[0]
        a = n.replace("i", "j")
        n = complex(a)
        total_wlan.append(n.real * 100)
        total_wlan.append(n.imag * 100)
    # st = random.randint(0, length - num - 1)  # select a beginning point randomly
    # # st = 0
    # random_wifi = wifi[st: st + num]
    # total_wlan = []
    # for i in random_wifi:
    #     n = i[0]
    #     a = n.replace("i", "j")
    #     n = complex(a)
    #     total_wlan.append(n.real*100)
    #     total_wlan.append(n.imag*100)
    tensor_wlan = torch.from_numpy(numpy.array(total_wlan).reshape(m, l)).to(device)
    temp = x + tensor_wlan.float()
    return temp
