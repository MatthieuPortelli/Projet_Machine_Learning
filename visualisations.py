import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_regression_scatter(selected_model, y_true_regression, y_pred_regression):
    fig = plt.figure(figsize=(8, 5))
    sns.regplot(x=y_true_regression, y=y_pred_regression, label='Régression linéaire', color='y', marker='o', 
                scatter_kws={"color": "darkblue", "alpha": 0.3, "s": 200})
    plt.title('Check for Linearity:\n Actual Vs Predicted value ({})'.format(selected_model))
    plt.legend()
    return fig


def plot_regression_histogram(selected_model, y_true, y_pred):
    fig = plt.figure(figsize=(8, 5))
    residuals = [yt - yp for yt, yp in zip(y_true, y_pred)]
    # residuals = y_true - y_pred
    plt.hist(residuals, bins=20, color="gold", ec="darkblue")
    plt.xlabel("Résidus")
    plt.ylabel("Fréquence")
    plt.title("Histogramme des Résidus ({})".format(selected_model))
    return fig


def plot_confusion_matrix(selected_model, y_true, y_pred):
    fig = plt.figure(figsize=(8, 5))
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies étiquettes")
    plt.title("Matrice de Confusion ({})".format(selected_model))
    return fig


def plot_roc_curve(selected_model, y_true, y_pred):
    # Calculer les taux de faux positifs (FPR) et les taux de vrais positifs (TPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # Calcul de l'aire sous la courbe ROC (AUC)
    roc_auc = auc(fpr, tpr)
    # Tracer la courbe ROC
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title('Courbe ROC ({})'.format(selected_model))
    plt.legend(loc='lower right')
    return fig


def visualize_selected_model(selected_algo, y_true_regression, y_pred_regression):
    regression = ["Regression_Lineaire", "Regression_Ridge"]
    classification = ["Random_Forest", "KNN_Classification", "Regression_Logistique"]
    if selected_algo in regression:
        fig_1 = plot_regression_scatter(selected_algo, y_true_regression, y_pred_regression)
        fig_2 = plot_regression_histogram(selected_algo, y_true_regression, y_pred_regression)
        return fig_1, fig_2
    elif selected_algo in classification:
        fig_3 = plot_confusion_matrix(selected_algo, y_true_regression, y_pred_regression)
        fig_4 = plot_roc_curve(selected_algo, y_true_regression, y_pred_regression)
        return fig_3, fig_4
