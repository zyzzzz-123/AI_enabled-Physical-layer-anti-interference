import numpy as np
import torch
from scipy.io import savemat
import torch.utils.data as Data
from channels import channel_load

def prepare_data(args, mode):
    """
    to generate data and dataset dataloader to be transmittered.
    :param args: args
    :param mode: "train" or "val"
    :return:
    """
    if (mode == "train"):
        set_size = args.train_set_size
        interf_500, interf_64 = channel_load(args, train_flag=True)
    elif (mode == "val"):
        set_size = args.val_set_size
        interf_500, interf_64 = channel_load(args, train_flag=False)
    else:
        raise ("data mode error!")
    k = args.n_source
    batch_size = args.batch_size
    input_samples = []
    for row in range(0, set_size):
        data = np.random.binomial(n=1, p=0.5, size=(k,))
        input_samples.append(data)
    data = np.asarray(input_samples)
    dataset = Data.TensorDataset(torch.from_numpy(data.astype('float32')), torch.from_numpy(data.astype('float32')), interf_500, interf_64)
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # labels = (torch.rand(set_size) * class_num).long()
    # data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=labels)
    # dataset = Data.TensorDataset(data, labels)
    # loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataset, loader, data


def d2b(d, n):
    d = np.array(d)
    d = np.reshape(d, (1, -1))
    power = np.flipud(2 ** np.arange(n))
    g = np.zeros((np.shape(d)[1], n))
    for i, num in enumerate(d[0]):
        g[i] = num * np.ones((1, n))
    b = np.floor((g % (2 * power)) / power)
    return np.fliplr(b)


def generate_encoded_sym_dict(n_channel, k, net, device):
    # Exporting Dictionaries
    bit_dict = d2b(torch.arange(2 ** k), k)
    input_dict = torch.eye(2 ** k).to(device)
    enc_output = net.transmitter(input_dict)
    S_encoded_syms = (enc_output.cpu()).detach().numpy()

    dict1 = {'S_encoded_syms': S_encoded_syms, 'bit_dict': bit_dict.astype(np.int8)}
    savemat('mfbanks/ae_mfbank_AWGN_BPSK_' + str(n_channel) + str(k) + '.mat', dict1)
    print('Generated dictionaries and encoded symbols')

# def get_plots():
#     # Plot 1 -
#     plt.plot(train_acc_store,'r-o')
#     plt.plot(test_acc_store,'b-o')
#     plt.xlabel('number of epochs')
#     plt.ylabel('accuracy')
#     plt.ylim(0.85,1)
#     plt.legend(('training','validation'),loc='upper left')
#     plt.title('train and test accuracy w.r.t epochs')
#     plt.show()
#
#     # Plot 2 -
#     plt.plot(train_loss_store,'r-o')
#     plt.plot(test_loss_store,'b-o')
#     plt.xlabel('number of epochs')
#     plt.ylabel('loss')
#     plt.legend(('training','validation'),loc='upper right')
#     plt.title('train and test loss w.r.t epochs')
#     plt.show()

def calculateSNR(signal, interference):
    signal = signal.detach().cpu().numpy()
    interference = interference.detach().cpu().numpy()
    signal_p = np.sum(np.abs(signal) ** 2) / 64
    interference_p = np.sum(np.abs(interference) ** 2) / 64
    SNR = 10 * np.log10(signal_p / interference_p)
    print("SNR:{:.2f}".format(SNR))
    return SNR