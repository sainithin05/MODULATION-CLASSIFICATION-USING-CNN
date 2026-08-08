import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(2, 128, 16)
        self.conv1_bn = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 64, 18)
        self.conv2_bn = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(6144, 256)
        self.fc1_bn = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 11)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))

        x = x.flatten(1)

        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)

        return x
