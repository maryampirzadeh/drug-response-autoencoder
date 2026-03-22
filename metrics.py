import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch

# to convert torch tensor or any other array to 1d array
def to_numpy_1d(arr):
    #if it's a torch tensor
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy()
    #if it's list, tupple, etc.
    else:
        arr = np.asarray(arr)
    #convert to 1d
    return arr.reshape(-1)

# ============================================================
# calculating all metrics
# real_labels are arrays of true labels (0 or 1)
# classifier_output are arrays of predicted probabilities
# ============================================================

def all_metrics(real_labels, classifier_output):
    #convert inputs
    real_labels = to_numpy_1d(real_labels)
    classifier_output = to_numpy_1d(classifier_output)
    # turn probabilities to 0 and 1
    predictions = (classifier_output > 0.5).astype(int)
    # calculating auc
    fp, tp, _ = metrics.roc_curve(real_labels, classifier_output)
    auc = metrics.auc(fp, tp)
    # AUPRC
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(real_labels, classifier_output)
    auprc = metrics.auc(recall_curve, precision_curve)

    # confusion matrix(its going to be 2x2 cause we only have 0 and 1) + accuracy
    accuracy = accuracy_score(real_labels, predictions)
    matrix_ = confusion_matrix(real_labels, predictions)
    #precision, recall, f1-score (if devided by zero = 0)
    precision = precision_score(real_labels, predictions, zero_division=0)
    recall = recall_score(real_labels, predictions, zero_division=0)
    f1 = f1_score(real_labels, predictions, zero_division=0)
    #to print confusion matrix and all results
    print(f"\nConfusion Matrix:")
    print(f"                Predicted 0   Predicted 1")
    print(f"Actual 0 (TN): {matrix_[0, 0]:10}   {matrix_[0, 1]:10}")
    print(f"Actual 1 (TP): {matrix_[1, 0]:10}   {matrix_[1, 1]:10}")
    print("="*60)
    print(
        f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\nF1 score: {f1:.4f}\n"
        f"AUC: {auc:.4f}\nAUPRC: {auprc:.4f}\n"
    )
    # to return all results in a dict
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1,
        "AUC": auc,
        "AUPRC": auprc,
    }


import matplotlib.pyplot as plt

# ============================================================
#visualize auc and auprc
# ============================================================

def plot_auc_auprc(real_labels, classifier_output):
    # all operations are the same as the function above
    real_labels = to_numpy_1d(real_labels)
    classifier_output = to_numpy_1d(classifier_output)
    # auc
    fp, tp, _ = metrics.roc_curve(real_labels, classifier_output)
    print(_)
    auc_score = metrics.auc(fp, tp)
    # auprc
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(
        real_labels, classifier_output
    )
    print(_)
    auprc_score = metrics.auc(recall_curve, precision_curve)

    #auc
    #empty canvas
    plt.figure()
    #fp is x, tp is y
    plt.plot(fp, tp, label=f"ROC AUC = {auc_score:.4f}")
    # y=x 
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.title("ROC Curve")
    #to show the label
    plt.legend()
    plt.grid(True)
    plt.show()

    #auprc
    #all operations are the same as auc
    plt.figure()
    plt.plot(recall_curve, precision_curve, label=f"AUPRC = {auprc_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
