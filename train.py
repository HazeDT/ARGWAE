import torch
from torch import optim
import logging
import numpy as np
import statsmodels.api as sm
import models
from scipy import sparse
from scipy.sparse.linalg import lobpcg
from torch_geometric.utils import get_laplacian
from test import show_pre_recall_f1
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


def learning_threshold(train_recon_error, alpha=0.1):
    dens = sm.nonparametric.KDEUnivariate(train_recon_error.astype(np.float64))
    bound = np.linspace(min(train_recon_error),max(train_recon_error),1024)
    dens.fit(bw='silverman',gridsize=2024)
    threshold=bound[min(np.where(dens.cdf>alpha)[0])]
    return threshold

@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)
    return torch.sparse_coo_tensor(index, value, A.shape)


# function for pre-processing
def ChebyshevApprox_KF(f, n,scale,lambda_max):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(scale * lambda_max * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)
    return c

def ChebyshevApprox_SF(f, n,lambda_max):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)

    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(lambda_max * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)
    return c



# function for pre-processing
def get_operator(L, n, s, scale_j, J, Q, lambda_max):
    r = s+1
    KF = lambda x: (x*lambda_max) * np.exp(-x*lambda_max)
    SF = lambda x: np.exp(-1) * np.exp(-np.power((x*Q) / (0.6 * lambda_max), 4)) # define the scale function


    c = [None] * r
    c[0] = ChebyshevApprox_SF(SF, n,lambda_max)
    for j in range(1,r):
        scale = scale_j(J[j-1])
        c[j] = ChebyshevApprox_KF(KF, n, scale, lambda_max)

    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    Lev = 1

    # for l in range(1, Lev + 1):

    for l in range(1, Lev + 1):

        for j in range(r):
            if j == 0:
                scale = 0
            else:
                scale = scale_j(J[j-1])

            T0F = FD1
            T1F = ((2 * (2*scale/lambda_max - 1)) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                # TkF = (2 * ((2 * (2*scale/lambda_max - 1)) * L)) @ T1F - 2 * T1F - T0F
                TkF = ((2 * (2 * scale / lambda_max - 1)) * L) @ T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d

def gen_filter_coe(dataset, num_nodes, s, Lev, n, r, device):
    '''
    :param dataset: pygeometric dataset
    :return: filter coefficent
    '''
    Q = 2
    L = get_laplacian(dataset.edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init)
    lambda_max = lambda_max[0]
    lambda_min = lambda_max/2


    scale_J = 2/lambda_min
    scale_1 = 1/lambda_max

    scale_j = lambda x:  scale_1 * np.power(scale_J/scale_1,(x-1)/(s-1))

    J = [i+1 for i in range(s)]  # scale list

    # get matrix operators
    d = get_operator(L, n, s, scale_j, J, Q, lambda_max)

    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0
    # store the matrix operators (torch sparse format) into a list: row-by-row
    d_list = list()

    for i in range(r):
        for l in range(Lev):
            d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))

    return d_list

class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.tet_loader = data['train'], data['val']
        self.device = device
        self.model_name = args.model_name
        self.Lev = args.Lev  # level of transform
        self.s = args.s  # dilation scale
        self.n = args.n  # n - 1 = Degree of Chebyshev Polynomial Approximation
        self.num_nodes = args.per_node * args.batch_size
        self.r = self.s + 1

        self.InputType = args.Input_type
        if self.InputType == "TD":
            feature = args.sample_length
        elif self.InputType == "FD":
            feature = int(args.sample_length/2)
        elif self.InputType == "other":
            feature = 1
        else:
            print("The InputType is wrong!!")


        self.model = getattr(models, args.model_name)(feature=feature, nhid=args.nhid, out_channel=self.args.latent_dim,
                                                      r=self.r, Lev=self.Lev, num_nodes=self.num_nodes, shrinkage=None,
                                                      threshold=1e-3, dropout_prob=args.dropout)

        self.criterion = nn.MSELoss()



    def train(self):
        """Training the Deep SVDD model"""
        self.model = self.model.to(self.device)

        if self.args.pretrain==True:
            state_dict = torch.load('./AEWeights/pretrained_parameters.pth')
            self.model.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            # net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(), lr=0.01)
        discriminator_scheduler = optim.lr_scheduler.MultiStepLR(discriminator_optimizer,
                                                   milestones=[5], gamma=0.1)
        self.model.train()
        # print(self.model)
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            train_recon_error = []
            for data in self.train_loader:

                d_list = gen_filter_coe(dataset=data, num_nodes=self.num_nodes, s=self.s,
                                        Lev=self.Lev, n=self.n, r=self.r, device=self.device)
                inputs = data.to(self.device)
                optimizer.zero_grad()
                x, z = self.model(inputs, d_list)
                num_nodes = data.num_nodes

                for i in range(10):
                    idx = range(num_nodes)
                    self.model.discriminator.train()

                    discriminator_optimizer.zero_grad()
                    discriminator_loss = self.model.discriminator_loss(z[idx])  # Comment
                    discriminator_loss.backward(retain_graph=True)
                    discriminator_optimizer.step()
                    # discriminator_scheduler.step()
                #-----------------------------------------------------------------------
                score = (torch.square(inputs.x - x)).mean(axis=1) #+ self.model.reg_loss(z)

                loss = self.model.reg_loss(z) + self.criterion(inputs.x, x)

                score = global_mean_pool(score, inputs.batch)
                loss.backward()
                optimizer.step()
                train_recon_error = np.hstack((train_recon_error, score.cpu().detach().numpy()))
                total_loss += loss.item()

            scheduler.step()

            print('Training GWAE... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))
        self.c = c

        self.threshold = learning_threshold(train_recon_error)
        # self.threshold = np.mean(train_recon_error)

        logging.info("threshold {:.4f}".format(self.threshold))

    def Mtest(self):
        scores = []
        labels = []
        self.model.eval()
        print('Testing...')
        start = True
        with torch.no_grad():
            for data in self.tet_loader:
                d_list = gen_filter_coe(dataset=data, num_nodes=self.num_nodes, s=self.s,
                                        Lev=self.Lev, n=self.n, r=self.r, device=self.device)
                inputs = data.to(self.device)
                y = inputs.y
                x, z = self.model(inputs, d_list)

                #-----------------------------------------------------------------------
                score = (torch.square(inputs.x - x)).mean(axis=1)

                score = global_mean_pool(score, inputs.batch)

                scores.append(score.detach().cpu())
                labels.append(y.cpu())

        labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()

        Accuracy, AUC, precision, recall, f1score, FDR, y_pred = show_pre_recall_f1(scores, labels, self.threshold)

        logging.info('Acc: {:.2f}, AUC: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1score: {:.2f}, FDR: {:.2f}'.format(
            Accuracy, AUC, precision, recall, f1score, FDR))