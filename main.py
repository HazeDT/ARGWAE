import argparse
import torch
import datasets
from train import TrainerARGWAE
from torch_geometric.data import DataLoader
import os
from utils.logger import setlogger
from datetime import datetime
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ARGWAE', help='the name of the model')
    parser.add_argument('--sample_length', type=int, default=1024,
                        help='batchsize of the training process')
    parser.add_argument('--data_name', type=str, default='RYsystemMultiRadius_AE',
                        help='the name of the data')
    parser.add_argument('--Input_type', choices=['TD', 'FD','other'],type=str, default='TD',
                        help='the input type decides the length of input')
    parser.add_argument('--data_dir', type=str, default= "E:\Data\RY_1500\RYsystemMultiRadius_AE.pkl",
                        help='the directory of the data')
    parser.add_argument('--task', choices=['Node', 'Graph'], type=str,default='Graph',
                        help='Node classification or Graph classification')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='the directory to save the model')
    #Model Hyperparameter--------------------------------------------------------
    parser.add_argument('--per_node', type=int, default=5,
                        help='the number of nodes of each subgraph')
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size")
    parser.add_argument('--nhid', type=int, default=512,
                        help='number of hidden units (default: 16)')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Dimension of the latent variable z')
    parser.add_argument('--Lev', type=int, default=1,
                        help='level of transform (default: 1)')
    parser.add_argument('--s', type=int, default=2,
                        help='dilation scale > 1 (default: 2)')
    parser.add_argument('--n', type=int, default=2,
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    #Other Hyperparameter--------------------------------------------------------
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--num_epochs_ae", type=int, default=100,
                        help="number of epochs for the pretraining")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.5e-3,
                        help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--weight_decay_ae', type=float, default=0.5e-3,
                        help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--lr_ae', type=float, default=1e-3,
                        help='learning rate for autoencoder')
    parser.add_argument('--lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument('--pretrain', type=bool, default=False,
                        help='Pretrain the network using an autoencoder')
    parser.add_argument('--normal_class', type=int, default=0,
                        help='Class to be treated as normal. The rest will be considered as anomalous.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='the number of training process')
    #parsing arguments.
    args = parser.parse_args() 

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the datasets
    Dataset = getattr(datasets, args.data_name)
    dataset = {}

    dataset['train'], dataset['val'] = Dataset(args.sample_length, args.data_dir, args.Input_type,
                                                           args.task).data_preprare()

    dataloaders = {x: DataLoader(dataset[x], batch_size=args.batch_size,
                                      shuffle=(True if x == 'train' else False),
                                      num_workers=args.num_workers,
                                      pin_memory=(True if device == 'cuda' else False))
                        for x in ['train', 'val']}

    Trainer = TrainerARGWAE(args, dataloaders, device)

#-----------------------------------------------------------------
    sub_dir = args.model_name+'_'+args.data_name + '_' + args.Input_type +'_'+datetime.strftime(datetime.now(), '%m%d-%H%M%S')

    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))
    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))


    if args.pretrain:
        Trainer.pretrain()
    Trainer.train()
    Trainer.Mtest()