from constantes import STRUCTURE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score


def train_model(model, X_train, y_train):
    if model in STRUCTURE['regression']:
        model_info = STRUCTURE['regression'][model]
    elif model in STRUCTURE['classification']:
        model_info = STRUCTURE['classification'][model]
    else:
        raise ValueError("Modèle non trouvé dans la structure.")
    model = model_info['model']
    # Récupération des hyperparameters
    hyperparameters = {param: param_info['values'] for param, param_info in model_info['hyperparameters'].items()}
    # Instance GridSearchCV pour effectuer une recherche sur la grille d'hyperparamètres avec validation croisée
    # Possibilité d'ajuster le nombre de plis de validation croisée (cv) selon vos besoins
    grid_search = GridSearchCV(model, hyperparameters, cv=5)
    # Entraîner le modèle avec les meilleurs hyperparamètres
    grid_search.fit(X_train, y_train)
    # Récupérez le meilleur modèle, les meilleurs paramètres et le meilleur score
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    return best_model, best_params, best_score


def evaluate_model(model, X_test, y_test, model_family):
    # Faites des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    if model_family == 'regression':
        # Métriques de régression (MSE, R²)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {'MSE': mse, 'R²': r2}
    elif model_family == 'classification':
        # Métriques de classification (précision, rappel, F1-score, etc.)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    else:
        raise ValueError("Type de modèle non valide.")
    # Retournez les métriques
    return metrics
