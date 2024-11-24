import torch.nn as nn
import torch.nn.functional as F

from .operation import ModelOper


class MLP(nn.Module, ModelOper):
    def __init__(self, input_size, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
