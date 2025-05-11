import torch
import torch.nn as nn

class FamilyCnn(nn.Module):
    def __init__(self, num_classes):
        super(FamilyCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.dropout1 = nn.Dropout(0.25) 
        self.dropout2 = nn.Dropout(0.5)   

        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)  # Apply dropout after conv blocks

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)  # Apply dropout before final layer

        return self.fc2(x)