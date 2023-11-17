import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNCifar10(nn.Module):
    def __init__(self, args):
        super(CNNCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x






# class CNN_Cifar10(nn.Module):
#     def __init__(self):
#         super(CNN_Cifar10, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.fc1 = nn.Sequential(
#             nn.Linear(8 * 8 * 32, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#         )
#         self.fc2 = nn.Linear(64, 10)

#         # Use Kaiming initialization for layers with ReLU activation
#         @torch.no_grad()
#         def init_weights(m):
#             if type(m) == nn.Linear or type(m) == nn.Conv2d:
#                 torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
#                 torch.nn.init.zeros_(m.bias)

#         self.conv.apply(init_weights)
#         self.fc1.apply(init_weights)

#     def forward(self, x):
#         conv_ = self.conv(x)
#         fc_ = conv_.view(-1, 8 * 8 * 32)
#         fc1_ = self.fc1(fc_)
#         output = self.fc2(fc1_)
#         return output
