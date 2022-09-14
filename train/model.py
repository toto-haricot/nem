import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):

    def __init__(self, layers_structure:list):

        super(NeuralNetwork, self).__init__()

        self.net = nn.Sequential()

        for i, u in enumerate(layers_structure[:-1]):

            self.net.add_module(f"dense_{i}", nn.Linear(u, layers_structure[i+1]))
            self.net.add_module(f"activation_{i}", nn.ReLU())

        # self.layer_01 = nn.Linear(input_size, 128)
        # self.activation_01 = nn.ReLU()
        # self.layer_02 = nn.Linear(128, 128)
        # self.activation_02 = nn.ReLU()
        # self.layer_03 = nn.Linear(128, 128)
        # self.activation_03 = nn.ReLU()
        # self.layer_04 = nn.Linear(128, 64)
        # self.activation_04 = nn.ReLU()
        # self.layer_05 = nn.Linear(64, 32)
        # self.activation_05 = nn.ReLU()
        # self.layer_06 = nn.Linear(32, 8)
        # self.activation_06 = nn.ReLU()
        # self.layer_07 = nn.Linear(8, 1)
        # self.activation_07 = nn.ReLU()


    def forward(self, x):

        out = self.net(x)

        return out

        # out = self.layer_01(x)
        # out = self.activation_01(out)
        # out = self.layer_02(out)
        # out = self.activation_02(out)
        # out = self.layer_03(out)
        # out = self.activation_03(out)
        # out = self.layer_04(out)
        # out = self.activation_04(out)
        # out = self.layer_05(out)
        # out = self.activation_05(out)
        # out = self.layer_06(out)
        # out = self.activation_06(out)
        # out = self.layer_07(out)
        # out = self.activation_07(out)

        # return out


