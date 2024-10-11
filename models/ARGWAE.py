import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.autoencoder import ARGVA
from torch_geometric.nn import GCNConv,global_mean_pool,GATConv,BatchNorm

EPS = 1e-15
MAX_LOGSTD = 10

class SGWConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, bias=True):
        super(SGWConv, self).__init__()
        self.Lev = Lev
        self.shrinkage = shrinkage
        self.threshold = threshold

        self.thr = nn.Parameter(torch.zeros(in_features) + threshold)
        self.crop_len = (Lev - 1) * num_nodes
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1).cuda())
        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        x = torch.matmul(x, self.weight)
        x = torch.sparse.mm(torch.cat(d_list, dim=0), x)
        if self.shrinkage is not None:
            if self.shrinkage == 'soft':
                x = torch.mul(torch.sign(x), (torch.abs(x) - self.thr))

            elif self.shrinkage == 'hard':
                x = torch.mul(x, (torch.abs(x) > self.threshold))
            else:
                raise Exception('Shrinkage type is invalid')

        # Hadamard product in spectral domain
        x = self.filter * x

        x = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:], dim=1), x[self.crop_len:, :])
        if self.bias is not None:
            x += self.bias

        return x


class ARGWAE(torch.nn.Module):
    """Adverasially 。
    """

    def __init__(self, feature, nhid, out_channel, r, Lev, num_nodes, shrinkage=None, threshold=1e-4,
                 dropout_prob=0.5):
        super().__init__()

        self.conv1 = SGWConv(feature, nhid, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        
        self.conv2 = SGWConv(nhid, out_channel, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)


        self.fc1 = nn.Sequential(
            nn.Linear(out_channel, 2 * out_channel),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(2 * out_channel, feature),
            nn.ReLU(inplace=True))
            # nn.Sigmoid())

        self.discriminator = Discriminator(in_channels=out_channel,
                                           hidden_channels=256, out_channels=1)

    def reset_parameters(self):
        super(self).reset_parameters()
        reset(self.discriminator)

    def reg_loss(self, z):
        r"""Computes the regularization loss of the encoder.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(z))
        # print('real',real.shape)
        real_loss = -torch.log(real + EPS).mean()

        return real_loss

    def discriminator_loss(self, z):
        r"""Computes the loss of the discriminator.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss

    def encode(self, x, d_list):
        """编码功能"""
        x = self.conv1(x, d_list)
        x = F.relu(x)
        x = self.conv2(x, d_list)
        # x = torch.sigmoid(x)
        x = F.relu(x)
        return x

    def decode(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, data, d_list):
        x = data.x

        z = self.encode(x, d_list)
        self.latent = z
        x_hat = self.decode(z)

        return x_hat, z

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

