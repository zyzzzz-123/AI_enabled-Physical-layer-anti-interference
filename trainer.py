import torch

import channels
import numpy as np
from channels import channel_init

def train(trainloader, net, ob_net,ob_net2, encoder, decoder, optimizer, criterion, device, loss_vec,acc_vec, args):
    step_total =0
    running_loss = 0.0
    running_corrects = 0
    EbN0_dB_train = 0.0
    net.train()
    ob_net.train()
    ob_net2.train()
    encoder.train()
    decoder.train()
    acc = 0
    for step, (x, y,interf_500, interf_64) in enumerate(trainloader):  # gives batch data
        step_total +=1
        # Move batches to GPU
        x = x.to(device)
        y = y.to(device)
        ## add interference to interf_500 / 64 ##
        interf_500 = channel_init(interf_500,args,"500").to(torch.float).detach().to(device)
        interf_64 = channel_init(interf_64, args, "64").to(torch.float).detach().to(device)

        optimizer.zero_grad()  # clear gradients for this training step
        ob_result = ob_net(interf_500)
        ob_result2  = ob_net2(interf_500)
        encode_input = torch.cat([ob_result,x],dim=1)
        encoded = encoder(encode_input)
        ## need normalization for encoded here ##

        ##

        trasmitted = encoded + interf_64   # add interference to signal
        decoder_input =  torch.cat([trasmitted[:,:,0],trasmitted[:,:,1],ob_result2],dim=1)          # decoder
        output = decoder(decoder_input)

        # this part is last stage work
        # train_encoded_signal = net.transmitter(x)
        # # val_constrained_encoded_signal = channels.energy_constraint(val_encoded_signal, args)
        # train_noisy_signal = channels.wlan(train_encoded_signal, args, device)
        # output = net.receiver(train_noisy_signal)

        loss = criterion(output, y)  # Apply cross entropy loss

        # Backward and optimize
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients
        optimizer.zero_grad()

        loss_vec.append(loss.item())  # Append to loss_vec
        # exp_lr_scheduler.step()
        acc += np.sum(((output.detach() > 0.5) == y.bool()).cpu().numpy())
        # acc_vec.append(acc)
        # pred_labels = torch.max(output, 1)[1].data.squeeze()
        # accuracy = sum(pred_labels == y) / float(batch_size)

        # statistics
        running_loss += loss.item()
        # running_corrects += accuracy
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model

    train_epoch_loss = running_loss / step_total
    # train_epoch_acc = running_corrects / step
    train_epoch_acc = acc / (args.train_set_size *args.n_source)
    return train_epoch_loss, train_epoch_acc


def validate(net, valloader, criterion,  device, args):
    step_total =0
    EbN0_dB_train = 0.0
    net.eval()
    acc = 0
    with torch.no_grad():
        for step, (val_data, val_labels,interf_500, interf_64) in enumerate(valloader):
            step_total += 1
            val_data = val_data.to(device)
            val_labels = val_labels.to(device)

            val_encoded_signal = net.transmitter(val_data)
            # val_constrained_encoded_signal = channels.energy_constraint(val_encoded_signal, args)
            # val_noisy_signal = channels.awgn(val_constrained_encoded_signal, args, EbN0_dB_train, device)
            val_noisy_signal = channels.wlan(val_encoded_signal, args, device)
            val_decoded_signal = net.receiver(val_noisy_signal)

            val_loss = criterion(val_decoded_signal, val_labels)  # Apply cross entropy loss
            acc += ((val_decoded_signal.detach() > 0.5) == val_labels.bool()).cpu().numpy().sum()
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
