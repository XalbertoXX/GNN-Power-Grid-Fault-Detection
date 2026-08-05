import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class BaselineCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]
        return self.fc(x)

class BaselineLSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, num_classes):
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class ST_GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4):
        super(ST_GAT, self).__init__()
        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=True)
        self.gat = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        lstm_out, _ = self.lstm(x)
        temporal_embed = lstm_out[:, -1, :]
        spatial_embed = F.elu(self.gat(temporal_embed, edge_index))
        global_embed = torch.mean(spatial_embed, dim=0, keepdim=True)
        return self.fc(global_embed)