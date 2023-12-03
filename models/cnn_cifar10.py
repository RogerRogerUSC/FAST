import torch
import torch.nn as nn
import torch.nn.functional as F


# class CNNCifar10(nn.Module):
#     def __init__(self):
#         super(CNNCifar10, self).__init__()
#         self.conv_layer = nn.Sequential(
#             # Conv Layer block 1
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # Conv Layer block 2
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout2d(p=0.05),
#             # Conv Layer block 3
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.fc_layer = nn.Sequential(
#             nn.Dropout(p=0.1),
#             nn.Linear(4096, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1),
#             nn.Linear(512, 10),
#         )
#         self.dropout = nn.Dropout(0.25)

#     def forward(self, x):
#         """Perform forward."""
#         # conv layers
#         x = self.conv_layer(x)
#         # flatten
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         # fc layer
#         x = self.fc_layer(x)

#         return x
    

class CNNCifar10(nn.Module):
    def __init__(self):
        super(CNNCifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class CNNCifar10_test(nn.Module):
    def __init__(self):
        super(CNNCifar10_test, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class CNNCifar10(nn.Module):
#     def __init__(self):
#         super(CNNCifar10, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.fc1 = nn.Sequential(
#             nn.Linear(8 * 8 * 32, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 64),
#             nn.LeakyReLU(),
#         )
#         self.fc2 = nn.Linear(64, 10)
#         self.dropout = nn.Dropout(0.4)

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
#         d = self.dropout(fc1_)
#         output = self.fc2(d)
#         return output

# class CNNCifar10_test(nn.Module):
#     def __init__(self):
#         super(CNNCifar10_test, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.fc1 = nn.Sequential(
#             nn.Linear(8 * 8 * 32, 256),
#             nn.LeakyReLU(),
#             nn.Linear(256, 64),
#             nn.LeakyReLU(),
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
