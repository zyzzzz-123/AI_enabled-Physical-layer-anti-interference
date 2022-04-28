import csv
from _csv import reader
from math import sqrt
import random

import numpy
import pandas as pd
import numpy as np
import torch
from scipy.io import loadmat
# Energy Constraints
from numpy import array
from numpy.matlib import repmat

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

def channel_load(args, train_flag):
    """
    :param args:
    :param mode:
    :param train_flag:  True means Train; False means Test
    :return:
    """
    address_500 = "/home/zyz/python_project/TA_project/interfsdatas/mixinterf_500_t9.mat"
    address_64 = "/home/zyz/python_project/TA_project/interfsdatas/mixinterf_64_t9.mat"
    interf_500 = loadmat(address_500)
    interf_500 = interf_500["orinterf"]
    interf_64 = loadmat(address_64)
    interf_64 = interf_64["orinterf"]

    ### this part is copied from TA's program, seeming to change relative amplitude of interference
    obj1 = np.array(range(900, 1000, 1))
    for i in range(2, 26, 1):
        allobj = np.hstack((obj1, range((i - 1) * 1000 + 900, i * 1000)))
        obj1 = allobj
    if (train_flag):
        interf_train_real = interf_500[:, :, 0][:25000] * args.AMP / 9
        interf_train_real = np.delete(interf_train_real, allobj, axis=0)
        interf_train_real = torch.from_numpy(interf_train_real)
        interf_train_imag = interf_500[:, :, 1][:25000] * args.AMP / 9
        interf_train_imag = np.delete(interf_train_imag, allobj, axis=0)
        interf_train_imag = torch.from_numpy(interf_train_imag)
        interf_500_ret = torch.complex(interf_train_real,interf_train_imag)

        interf_train_real = interf_64[:, :, 0][:25000] * args.AMP / 9
        interf_train_real = np.delete(interf_train_real, allobj, axis=0)
        interf_train_real = torch.from_numpy(interf_train_real)
        interf_train_imag = interf_64[:, :, 1][:25000] * args.AMP / 9
        interf_train_imag = np.delete(interf_train_imag, allobj, axis=0)
        interf_train_imag = torch.from_numpy(interf_train_imag)
        interf_64_ret = torch.complex(interf_train_real, interf_train_imag)
    else:
        interf_test_real = interf_500[:, :, 0][-3100:-100] * args.AMP / 9
        interf_test_real = torch.from_numpy(interf_test_real)
        interf_test_imag = interf_500[:, :, 1][-3100:-100] * args.AMP / 9
        interf_test_imag = torch.from_numpy(interf_test_imag)
        interf_500_ret = torch.complex(interf_test_real, interf_test_imag)

        interf_test_real = interf_64[:, :, 0][-3100:-100] * args.AMP / 9
        interf_test_real = torch.from_numpy(interf_test_real)
        interf_test_imag = interf_64[:, :, 1][-3100:-100] * args.AMP / 9
        interf_test_imag = torch.from_numpy(interf_test_imag)
        interf_64_ret = torch.complex(interf_test_real, interf_test_imag)

    return interf_500_ret, interf_64_ret


def channel_init(interf, args, mode):
    """
    return channel 500 // 64 interference, adding AWGN / SFO / CFO
    :param args:
    :param mode: 500/64
    :return: interf_real , interf_imag
    """
    interf_real = torch.real(interf)
    interf_imag = torch.real(interf)
    if ( args.AWGN ):
        interf_real = AWGN_mod(interf_real, args, mode)
        interf_imag = AWGN_mod(interf_imag, args, mode)

    if (args.SFO):
        interf_real, interf_imag = SFO_mod(interf_real, interf_imag, args, mode)

    if (args.CFO):
        interf_real, interf_imag = CFO_mod(interf_real, interf_imag, args, mode)

    return interf_real , interf_imag


### invoked by channel_init
def AWGN_mod(interf_, args, mode):
    if (mode == "500"):
        length = 500
    elif (mode == "64"):
        length = 64
    m, _ = interf_.shape
    a = np.random.normal(size=(m, length)) * args.AMP_n
    print(interf_.shape)
    print(a.shape)
    # interf_ += np.random.normal(size=(args.N_OFDM_SYMS, length)) * args.AMP_n
    interf_ += np.random.normal(size=(m, length)) * args.AMP_n
    return interf_


### invoked by channel_init
def SFO_mod(interf_real, interf_imag, args, mode):
    ## transform the input to complex format and use fft.
    interf_complex = torch.complex(interf_real, interf_imag)  # Tensor(500,500)  # 用这种语法构造一个复数
    interf_complex = torch.fft.fft(interf_complex)                               # fft
    interf_f_real =torch.real(interf_complex)
    interf_f_imag = torch.imag(interf_complex)

    # generate SFO coefficient
    SFO_cos_500, SFO_sin_500 = GenerateSFO(args, mode)

    # 这里相当于手动做了个复数乘法
    interf_SFO_real = interf_f_real * SFO_cos_500 - interf_f_imag * SFO_sin_500
    interf_SFO_imag = interf_f_imag * SFO_cos_500 + interf_f_real * SFO_sin_500
    interf_SFO = torch.complex(interf_SFO_real, interf_SFO_imag)

    interf_SFO = torch.fft.ifft(interf_SFO)                         # ifft
    interf_SFO_real = torch.real(interf_SFO)
    interf_SFO_imag = torch.imag(interf_SFO)

    return interf_SFO_real, interf_SFO_imag

def CFO_mod(interf_real, interf_imag, args, mode):
    CFO_cos_500, CFO_sin_500 = GenerateCFO(args, mode)

    input_CFO_real = interf_real * CFO_cos_500 - interf_imag * CFO_sin_500
    input_CFO_imag = interf_imag * CFO_cos_500 + interf_real * CFO_sin_500
    # input_CFO = torch.complex(input_CFO_real, input_CFO_imag)

    return input_CFO_real, input_CFO_imag

### this part is copied from TA's programme, to generate SFO coefficient
### invoked by SFO_mod
def GenerateSFO(args, mode):
    if (mode == "500"):
        N_ORDERS = 500
    elif (mode == "64"):
        N_ORDERS = 64
    SFO_delta = (np.random.rand(args.N_OFDM_SYMS, 1)) * args.SFO_delta_max
    SFO_delta = repmat(SFO_delta, 1, N_ORDERS)

    SFO_exp = np.zeros((args.N_OFDM_SYMS, N_ORDERS))
    for i in range(int(np.round(N_ORDERS / 2))):
        SFO_exp[:, i] = i
    for i in range(int(np.round(N_ORDERS / 2)), N_ORDERS):
        SFO_exp[:, i] = N_ORDERS - i

    SFO_exp = SFO_exp * SFO_delta * 2 * np.pi / N_ORDERS
    SFO_cos = np.cos(SFO_exp)
    SFO_sin = np.sin(SFO_exp)
    return SFO_cos, SFO_sin

### this part is copied from TA's programme, to generate CFO coefficient
### invoked by CFO_mod
def GenerateCFO(args, mode):
    if (mode == "500"):
        N_ORDERS = 500
    elif (mode == "64"):
        N_ORDERS = 64
    CFO_delta = (np.random.rand(args.N_OFDM_SYMS, N_ORDERS)) * args.CFO_delta_max
    n_arrays = np.arange(0, N_ORDERS, 1)
    CFO_narrays = repmat(n_arrays, args.N_OFDM_SYMS, 1)
    CFO_exp = CFO_delta * 2 * np.pi * CFO_narrays / N_ORDERS
    CFO_cos = np.cos(CFO_exp)
    CFO_sin = np.sin(CFO_exp)
    return CFO_cos, CFO_sin




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
