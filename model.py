import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm # noqa


class ChebyNet(torch.nn.Module):
    def __init__(self, in_channels=1024, z_dim=32):
        super(ChebyNet, self).__init__()

        self.GConv1 = GCNConv(in_channels,2 * z_dim)

        self.GConv2 = GCNConv(2 * z_dim, z_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.GConv1(x, edge_index)
        x = F.relu(x)

        x = self.GConv2(x, edge_index)

        return x



class autoencoder(nn.Module):
    def __init__(self,in_channels=1024, z_dim=32):
        super(autoencoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * z_dim)
        self.conv2 = GCNConv(2 * z_dim, z_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, 2 * z_dim),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(2 * z_dim, in_channels),
            nn.ReLU(inplace=True))
        
    def encode(self, x, edge_index):

        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
   
    def decode(self, x):
        x = self.fc1(x)
        return self.fc2(x)

