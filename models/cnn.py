import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFashionMNIST(nn.Module):
    def __init__(self):
        super(CNNFashionMNIST, self).__init__()

        # Layer 1: Conv2d (input: 28x28x1 -> output: 28x28x3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(3)
        
        # Layer 2: Conv2d (input: 28x28x3 -> output: 28x28x5)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(5)

        # MaxPool (input: 28x28x5 -> output: 14x14x5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Conv2d (input: 14x14x5 -> output: 14x14x5)
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(5)
        
        # Layer 4: Conv2d (input: 14x14x5 -> output: 14x14x3)
        self.conv4 = nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(3)
        
        # MaxPool (input: 14x14x3 -> output: 7x7x3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 3, 147)    # Flattened input: 7*7*3 = 147
        self.fc2 = nn.Linear(147, 10)           # Output: 10 classes

        # Xavier initialization
        self._initialize_weights()

    def forward(self, x):
        # Convolutional layers with ReLU and BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Flatten the feature map
        x = x.view(x.size(0), -1)

        # Fully connected layers with Sigmoid activation
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _initialize_weights(self):
        # Xavier initialization for all layers
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)