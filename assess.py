# Evaluation metrics
import numpy as np

# Calculate confusion matrix
def _fast_hist(label_true, label_pred, n_class):

    mask = (label_true >= 0) & (label_true < n_class)

    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def hist_sum(label_trues, label_preds, n_class):

    hist = np.zeros((n_class, n_class))

    for lbt, lbp in zip(label_trues, label_preds):

        hist += _fast_hist(lbt.flatten(), lbp.flatten(), n_class)

    return hist


def compute_metrics(hist):
    oa = np.diag(hist).sum() / hist.sum()

    expected_accuracy = (hist.sum(axis=1) * hist.sum(axis=0)).sum() / (hist.sum() ** 2)
    kappa = (oa - expected_accuracy) / (1 - expected_accuracy)

    precision = np.diag(hist) / (hist.sum(axis=0) + np.finfo(np.float32).eps)

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + np.finfo(np.float32).eps)
    miou = np.nanmean(iou)

    recall = np.diag(hist) / (hist.sum(axis=1) + np.finfo(np.float32).eps)

    F1 = 2 * precision * recall / (precision + recall + np.finfo(np.float32).eps)

    return miou, oa, kappa, precision[1], recall[1], iou[1], F1[1]