import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5) # (1 * 5 * 5) * 6 = 150
        self.s2 = torch.nn.MaxPool2d(kernel_size=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5) # (6 * 5 * 5) * 16 = 2,400
        self.s4 = torch.nn.MaxPool2d(kernel_size=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5) # (16 * 5 * 5) * 120 = 48,000
        self.fc6 = nn.Linear(120, 84) # 120 * 84 = 10,080
        self.output = nn.Linear(84, 10) # 84 * 10 = 840
        # total parameter = 61,470
        
    def forward(self, img):
        img = torch.tanh(self.c1(img))
        img = self.s2(img)
        img = torch.tanh(self.c3(img))
        img = self.s4(img)
        img = torch.tanh(self.c5(img))
        img = img.reshape(img.shape[0], 120)
        img = torch.tanh(self.fc6(img))
        output = self.output(img)
        return output

class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=1024, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=32)
        self.fc6 = nn.Linear(in_features=32, out_features=32)
        self.fc7 = nn.Linear(in_features=32, out_features=32)
        self.fc8 = nn.Linear(in_features=32, out_features=32)
        self.fc9 = nn.Linear(in_features=32, out_features=10)
        self.relu = nn.ReLU()
        # total parameter = 61,888
        
    def forward(self, img):
        img = img.reshape(img.shape[0], -1)
        img = self.relu(self.fc1(img))
        img = self.relu(self.fc2(img))
        img = self.relu(self.fc3(img))
        img = self.relu(self.fc4(img))
        img = self.relu(self.fc5(img))
        img = self.relu(self.fc6(img))
        img = self.relu(self.fc7(img))
        img = self.relu(self.fc8(img))
        output = self.fc9(img)

        return output