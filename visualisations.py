import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_regression_scatter(y_true_regression, y_pred_regression):
    fig = plt.figure(figsize=(8, 5))
    sns.regplot(x=y_true_regression, y=y_pred_regression, label='Régression linéaire', color='y', marker='o', 
                scatter_kws={"color": "darkblue", "alpha": 0.3, "s": 200})
    plt.title('Check for Linearity:\n Actual Vs Predicted value (Régression Linéaire)')
    plt.legend()
    plt.show()
    return fig


def plot_regression_histogram(y_true, y_pred):
    fig = plt.figure(figsize=(8, 5))
    residuals = [yt - yp for yt, yp in zip(y_true, y_pred)]
    # residuals = y_true - y_pred
    plt.hist(residuals, bins=20, color="gold", ec="darkblue")
    plt.xlabel("Résidus")
    plt.ylabel("Fréquence")
    plt.title("Histogramme des Résidus (Régression Linéaire)")
    return fig


def plot_logistic_confusion_matrix(y_true, y_pred):
    fig = plt.figure(figsize=(8, 5))
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies étiquettes")
    plt.title("Matrice de Confusion (Régression Logistique)")
    return fig


def plot_logistic_roc_curve(y_true, y_pred):
    # Binarize the output
    y_test_bin_logis = label_binarize(y_true, classes=np.unique(y_true))
    y_pred_bin_logis = label_binarize(y_pred, classes=np.unique(y_pred))
    Classes = ['0', '1', '2']

    fig = plt.figure(figsize=(8, 5))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_class = 3
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin_logis[:, i], y_pred_bin_logis[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], linestyle='--', label='%s vs Rest (AUC=%0.2f)' % (Classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC curve Régression Logistique')
    plt.legend(loc='lower right')
    return fig


def plot_rf_confusion_matrix(y_true, y_pred):
    fig = plt.figure(figsize=(8, 5))
    conf_matrix_rf = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies étiquettes")
    plt.title("Matrice de Confusion (Random Forest)")
    return fig


def plot_rf_roc_curve(y_true, y_pred):
    # Binarize the output
    y_test_bin = label_binarize(y_true, classes=np.unique(y_true))
    y_pred_bin = label_binarize(y_pred, classes=np.unique(y_pred))
    Classes = ['0', '1', '2']

    fig = plt.figure(figsize=(8, 5))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_class = 3
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], linestyle='--', label='%s vs Rest (AUC=%0.2f)' % (Classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC curve')
    plt.legend(loc='lower right')
    return fig


def plot_knn_confusion_matrix(y_true, y_pred):
    conf_matrix_knn = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix_knn, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies étiquettes")
    plt.title("Matrice de Confusion (K-Nearest Neighbors)")
    return fig


def visualize_selected_model(selected_algo, y_true_regression, y_pred_regression):
    regression = ["Regression_Lineaire", "Régression_Ridge"]
    classification = ["Random_Forest", "KNN_Classification", "Regression_Logistique"]
    if selected_algo in regression:
        fig_1 = plot_regression_scatter(y_true_regression, y_pred_regression)
        fig_2 = plot_regression_histogram(y_true_regression, y_pred_regression)
        return fig_1, fig_2
    elif selected_algo in classification:
        fig_3 = plot_rf_confusion_matrix(y_true_regression, y_pred_regression)
        fig_4 = plot_rf_roc_curve(y_true_regression, y_pred_regression)
        return fig_3, fig_4
