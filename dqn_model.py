
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        #self.fc1 = nn.Linear(512, 1)
        self.fc1 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x

class DuelingDQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DuelingDQN, self).__init__()
        # Shared layers
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(32 * 6 * 6, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(32 * 6 * 6, num_actions)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean()
