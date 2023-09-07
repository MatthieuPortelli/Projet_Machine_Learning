import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve


def plot_regression_scatter(selected_model, y_true_regression, y_pred_regression):
    fig = plt.figure(figsize=(8, 5))
    sns.regplot(x=y_true_regression, y=y_pred_regression, label='Régression linéaire', color='#FFB923', marker='o',
                scatter_kws={"color": "#213C55", "alpha": 0.3, "s": 200})
    plt.title('Vérification de la Linéarité :\n Valeur Réelle Vs Valeur Prédite ({})'.format(selected_model))
    # Texte explicatif
    help_text = "\n \n \n Evaluation de la relation entre les variables en examinant la tendance des points \n " \
                "en fonction de la dispersion autour de la ligne ou de la tendance. \n " \
                "Les variations sont un changement proportionnel.".format(
        selected_model)
    # plt.text(0.5, -0.15, help_text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=10, color='#34516F')
    plt.legend()
    return fig, help_text


def plot_regression_histogram(selected_model, y_true, y_pred):
    fig = plt.figure(figsize=(8, 5))
    residuals = [yt - yp for yt, yp in zip(y_true, y_pred)]
    # residuals = y_true - y_pred
    plt.hist(residuals, bins=20, color="#FFB923", ec="#34516F")
    plt.xlabel("Résidus")
    plt.ylabel("Fréquence")
    plt.title("Histogramme des Résidus ({})".format(selected_model))
    # Texte explicatif
    help_text = "\n \n Représentation de la distribution des erreurs de prédiction du modèle. \n " \
                "Une distribution autour de zéro et symétrique indique généralement des prédictions précises."\
        .format(selected_model)
    # plt.text(0.5, -0.15, help_text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=10, color='#34516F')
    return fig, help_text


def plot_confusion_matrix(selected_model, y_true, y_pred):
    fig = plt.figure(figsize=(8, 5))
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Valeurs Prédites")
    plt.ylabel("Valeurs Réelles")
    plt.title("Matrice de Confusion ({})".format(selected_model))
    # Texte explicatif
    help_text = "\n \n Distribution des erreurs de prédiction du modèle. Centrée autour de zéro et symétrique, " \
                "le modèle fait des prédictions précises, \n " \
                "en identifiant correctement les vrais positifs et négatifs ainsi que les faux positifs et négatifs."\
        .format(selected_model)
    # plt.text(0.5, -0.15, help_text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=10, color='#34516F')
    return fig, help_text


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
    plt.title('Courbe ROC {}'.format(selected_model))
    # Texte explicatif
    help_text = "\n \n Ratio entre les faux positifs et les vrais positifs. \n " \
                "Une courbe ROC idéale se rapproche du coin supérieur gauche, elle distingue efficacement les classes,\n " \
                "une courbe se rapprochant de la ligne diagonale aléatoire suggère une performance médiocre." \
        .format(selected_model)
    # plt.text(0.5, -0.15, help_text, ha='center', va='center', transform=plt.gca().transAxes, fontsize=10,
    #          color='#34516F')
    plt.legend(loc='lower right')
    return fig, help_text


def visualize_selected_model(selected_algo, y_true_regression, y_pred_regression):
    regression = ["Regression_Lineaire", "Regression_Ridge"]
    classification = ["Random_Forest", "KNN_Classification", "Regression_Logistique"]
    if selected_algo in regression:
        fig_1, help_text_1 = plot_regression_scatter(selected_algo, y_true_regression, y_pred_regression)
        fig_2, help_text_2 = plot_regression_histogram(selected_algo, y_true_regression, y_pred_regression)
        return fig_1, help_text_1, fig_2, help_text_2
    elif selected_algo in classification:
        fig_3, help_text_1 = plot_confusion_matrix(selected_algo, y_true_regression, y_pred_regression)
        fig_4, help_text_2 = plot_roc_curve(selected_algo, y_true_regression, y_pred_regression)
        return fig_3, help_text_1, fig_4, help_text_2


def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label="Précision d'entraînement")
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='#FFB923', linestyle='--', marker='s', markersize=5, label="Précision de validation")
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='#FFB923')
    plt.xlabel("Nombre d'échantillons d'entraînement")
    plt.ylabel('Précision')
    plt.legend(loc='lower right')
    plt.title("Courbe d'apprentissage")
    return plt
