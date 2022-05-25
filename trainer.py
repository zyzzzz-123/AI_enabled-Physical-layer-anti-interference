import torch
import channels
import numpy as np
from channels import channel_init
import matplotlib.pyplot as plt
from channels import  plot_fft_one_channel_2
from channels import channel_choose, encoded_normalize
from utils import calculateSNR
import torchviz
import hiddenlayer as h
from torchviz import make_dot
from tensorboardX import SummaryWriter


def train(trainloader,Ob_Autoencoder, optimizer, criterion,global_step, device, args):
    step_total =0
    running_loss = 0.0
    running_corrects = 0
    EbN0_dB_train = 0.0
    Ob_Autoencoder.train()
    acc = 0
    writer = SummaryWriter(logdir = "log")
    for step, (x, y,interf_500, interf_64) in enumerate(trainloader):  # gives batch data
        step_total +=1
        global_step +=1
        # Move batches to GPU
        x = x.to(device)
        y = y.to(device)
        ## add interference to interf_500 / 64 ##
        interf_500_1 = channel_init(interf_500,args,"500").to(torch.float).detach().to(device)
        interf_500_2 = channel_init(interf_500,args,"500").to(torch.float).detach().to(device)
        interf_64 = channel_init(interf_64, args, "64").to(torch.float).detach().to(device)

        optimizer.zero_grad()  # clear gradients for this training step

        output = Ob_Autoencoder(x, interf_500_1 , interf_500_2, interf_64, global_step, device=device)

        # if (step_total == 1):
        #     with SummaryWriter('net') as w:
        #         w.add_graph(Ob_Autoencoder, input_to_model=(x, interf_64))
        loss = criterion(output, y)  # Apply cross entropy loss
        writer.add_scalar('loss', step_total,loss)
        # Backward and optimize
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients
        optimizer.zero_grad()

        # exp_lr_scheduler.step()
        acc += np.sum(((output.detach() > 0.5) == y.bool()).cpu().numpy())

        # statistics
        running_loss += loss.item()
        # running_corrects += accuracy
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model

    train_epoch_loss = running_loss / step_total
    train_epoch_acc = acc / (args.train_set_size *args.n_source)
    return train_epoch_loss, train_epoch_acc, global_step


def validate(Ob_Autoencoder, valloader, criterion,  device, args):
    step_total =0
    EbN0_dB_train = 0.0
    Ob_Autoencoder.eval()
    acc = 0
    with torch.no_grad():
        for step, (val_data, val_labels, interf_500, interf_64) in enumerate(valloader):
            step_total += 1
            val_data = val_data.to(device)
            val_labels = val_labels.to(device)

            ## add interference to interf_500 / 64 ##
            interf_500_1 = channel_init(interf_500, args, "500").to(torch.float).detach().to(device)
            interf_500_2 = channel_init(interf_500, args, "500").to(torch.float).detach().to(device)
            interf_64 = channel_init(interf_64, args, "64").to(torch.float).detach().to(device)

            output = Ob_Autoencoder(val_data, interf_500_1, interf_500_2,interf_64, step_total,device)

            val_loss = criterion(output, val_labels)  # Apply cross entropy loss
            acc += ((output.detach() > 0.5) == val_labels.bool()).cpu().numpy().sum()
            # val_pred_labels = torch.max(val_decoded_signal, 1)[1].data.squeeze()
            # val_accuracy = sum(val_pred_labels == val_labels) / float(batch_size)
        val_loss = val_loss / step_total
        val_accuracy = acc / (args.val_set_size* args.n_source)
    return val_loss, val_accuracy


def test(net, args, testloader, device, EbN0_test):
    net.eval()
    with torch.no_grad():
        for test_data, test_labels in testloader:
            test_data = test_data.to(device)
            test_labels = test_labels.to(device)

            encoded_signal = net.transmitter(test_data)
            constrained_encoded_signal = channels.energy_constraint(encoded_signal, args)
            noisy_signal = channels.wlan(constrained_encoded_signal, args, EbN0_test, device)
            decoded_signal = net.receiver(noisy_signal)

            pred_labels = torch.max(decoded_signal, 1)[1].data.squeeze()
            test_BLER = sum(pred_labels != test_labels) / float(test_labels.size(0))
    return test_BLER
