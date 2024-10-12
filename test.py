import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (accuracy_score, f1_score)


def show_pre_recall_f1(test_recon_error, test_label, threshold):
    y_pred = test_recon_error.copy()
    larger_idx = np.where(y_pred >= threshold)
    lower_idx = np.where(y_pred < threshold)
    y_pred[lower_idx[0]] = 0
    y_pred[larger_idx[0]] = 1
    Accuracy = accuracy_score(test_label, y_pred)*100
    AUC= roc_auc_score(test_label, test_recon_error) * 100
    f1score = f1_score(test_label, y_pred)*100

    return Accuracy, AUC, f1score

def eval(net, c, threshold, dataloader, device):
    """Testing the Deep SVDD model"""

    scores = []
    labels = []
    net.eval()
    print('Testing...')

    with torch.no_grad():
        for data in dataloader:
            inputs = data.to(device)
            y = inputs.y
            z = net(inputs)
            score = (torch.square(inputs.x - z)).mean(axis=1)
            scores.append(score.detach().cpu())
            labels.append(y.cpu())
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()

    Accuracy, AUC, f1score = show_pre_recall_f1(scores, labels, threshold)

    print('Acc: {:.2f}, AUC: {:.2f}, f1score: {:.2f}'.format(
        Accuracy,AUC, f1score))

    return Accuracy, AUC, f1score