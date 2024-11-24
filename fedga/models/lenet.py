import torch.nn as nn
import torch.nn.functional as F

from .operation import ModelOper


class LeNet(nn.Module, ModelOper):
    def __init__(self, channel, dim1, dim2, num_classes=10):
        super(LeNet, self).__init__()
        self._n_features = (dim1-12) * (dim2-12)
        self.conv1 = nn.Conv2d(channel, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(self._n_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self._n_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    lenet_model = LeNet(3, 32, 32, 10)
    print(lenet_model)
    import torch
    x = torch.randn(2, 3, 32, 32)
    y = lenet_model(x)
    print("y", y.shape)
