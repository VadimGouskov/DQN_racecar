import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from racecar.utils import showImage, saveImage, showTensor, saveTensor

import numpy as np

class Qnetwork(nn.Module):
    def __init__(self, actionSpaceSize , alpha):
        super(Qnetwork, self).__init__()
        self.alpha = alpha
        self.conv1 = nn.Conv2d(1, 16, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32,4, stride=2)
        # TODO flattening hier ? 
        self.fc1 = nn.Linear(32*10*10, 256)
        self.fc2 = nn.Linear(256, actionSpaceSize)

        self.optimizer = optimizer.RMSprop(self.parameters(), lr=self.alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        # showImage(input)
        self.printSize(input[:], "before list")
        input = list(input) # convert to list to accomidate pyTorch's format
        self.printSize(input[:], "after list")
        # print(input[:])
        prop = torch.Tensor(input).to(self.device)
        self.printSize(prop, "prop")
        # saveTensor(prop)

        prop = prop.view(-1, 1, 96, 96)
        print(len(prop[0][0]))

        # CONVOLUTION LAYERS

        prop = F.relu(self.conv1(prop))
        print(len(prop[0][0]))

        prop = F.relu(self.conv2(prop))
        print(len(prop[0][0]))

        print("output length of conv2: \n" + str(len(prop[0])) + "x" + str(len(prop[0][0])) + "x" + str(len(prop[0][0][0])) )

        # FLATTENING
        flat = prop.view(-1, 32 * 10 * 10)

        # FULLY CONNECTED
        prop = F.relu(self.fc1(flat))
        out = self.fc2(prop)
        # print(out)

        return out


    def printSize(self, observation, message = "<message>"):
        print(message + " - input size of is: " + str(len(observation)) + " x " + str(len(observation[0])))


