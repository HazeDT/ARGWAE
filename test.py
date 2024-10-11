import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (accuracy_score,precision_recall_curve, auc,average_precision_score,
                             roc_curve,precision_score, recall_score, f1_score)
from scipy import interpolate


def show_pre_recall_f1(test_recon_error, test_label, threshold):
    y_pred = test_recon_error.copy()
    larger_idx = np.where(y_pred >= threshold)
    lower_idx = np.where(y_pred < threshold)
    y_pred[lower_idx[0]] = 0
    y_pred[larger_idx[0]] = 1

    fpr, tpr, thresholds = roc_curve(test_label, test_recon_error)
    fpr95 = float(interpolate.interp1d(fpr, tpr)(0.95))

    Accuracy = accuracy_score(test_label, y_pred)*100

    precision = precision_score(test_label, y_pred)*100
    recall = recall_score(test_label, y_pred)*100
    AUC= roc_auc_score(test_label, test_recon_error) * 100
    f1score = f1_score(test_label, y_pred)*100
    CMatrix = confusion_matrix(test_label, y_pred)
    # FAR = CMatrix[1][0]/(CMatrix[1][0]+CMatrix[0][0])*100
    FDR = CMatrix[1][1]/(CMatrix[0][1]+CMatrix[1][1])*100
    print(CMatrix)
    print('prcesion', CMatrix[1][1]/(CMatrix[1][0]+CMatrix[1][1])*100)
    return Accuracy, AUC, precision, recall, f1score, FDR, y_pred

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

    Accuracy, AUC, f1score, FAR, FDR = show_pre_recall_f1(scores, labels, threshold)

    print('Acc: {:.2f}, AUC: {:.2f}, f1score: {:.2f}, FAR: {:.2f}, FDR: {:.2f}'.format(
        Accuracy,AUC, f1score, FAR, FDR))

    return Accuracy,AUC, f1score, FAR, FDR