import torch.nn as nn
import channels

class FC_Autoencoder(nn.Module):
    def __init__(self, k, n_channel):
        self.k = k
        self.n_channel = n_channel

        super(FC_Autoencoder, self).__init__()
        self.transmitter = nn.Sequential(
            nn.Linear(in_features=self.k, out_features=self.n_channel, bias=True),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Linear(in_features=self.n_channel, out_features=self.n_channel, bias=True),
            # nn.ReLU(inplace=True),
            # nn.PReLU(),
            # nn.Linear(in_features=self.n_channel, out_features=self.n_channel, bias=True))
            )
        self.receiver = nn.Sequential(
            nn.Linear(in_features=self.n_channel, out_features= self.n_channel , bias=True),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Linear(in_features=self.n_channel, out_features=self.k, bias=True),
            # nn.ReLU(inplace=True),
            # nn.PReLU(),
            # nn.Linear(in_features=self.k, out_features=self.k, bias=True),
        )

    def forward(self, x):
        # x_transmitted = self.transmitter(x)
        # # x_normalized = self.energy_normalize(x_transmitted)
        # x_noisy = channels.awgn(x_transmitted,args)  # Gaussian Noise
        # x = self.receiver(x_noisy)
        # x = x.to(device)
        return x

class cnn_Autoencoder(nn.Module):
    def __init__(self, k, n_channel):
        self.k = k
        self.n_channel = n_channel

        super(cnn_Autoencoder, self).__init__()
        layers1 = []
        num_of_layers = 3
        for _ in range(num_of_layers):
            layers1.append(nn.Conv1d(in_channels=self.k, out_channels=self.k, kernel_size=3, padding=1, bias=False))
            layers1.append(nn.BatchNorm2d(self.k))
            layers1.append(nn.ReLU(inplace=True))
        layers1.append(nn.Conv2d(in_channels=self.k, out_channels=self.n_channel, kernel_size=3, padding=1, bias=False))
        self.transmitter =nn.Sequential(*layers1)

        layers2 = []
        layers2.append(nn.Conv2d(in_channels=self.n_channel, out_channels=self.k, kernel_size=3, padding=1, bias=False))
        for _ in range(num_of_layers):
            layers2.append(nn.Conv2d(in_channels=self.k, out_channels=self.k, kernel_size=3, padding=1, bias=False))
            layers2.append(nn.BatchNorm2d(self.k))
            layers2.append(nn.ReLU(inplace=True))
        self.receiver = nn.Sequential(*layers2)

    def forward(self, x):
        # x_transmitted = self.transmitter(x)
        # # x_normalized = self.energy_normalize(x_transmitted)
        # x_noisy = channels.awgn(x_transmitted,args)  # Gaussian Noise
        # x = self.receiver(x_noisy)
        # x = x.to(device)
        return x
