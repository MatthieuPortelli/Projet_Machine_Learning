from constantes import STRUCTURE
from sklearn.model_selection import GridSearchCV


def train_model(selected_model, X_train, y_train):
    if selected_model in STRUCTURE['regression']:
        model_info = STRUCTURE['regression'][selected_model]
    elif selected_model in STRUCTURE['classification']:
        model_info = STRUCTURE['classification'][selected_model]
    else:
        raise ValueError("Modèle non trouvé dans la structure.")
    model = model_info['model']
    hyperparameters = model_info['hyperparameters']
    # Instance GridSearchCV pour effectuer une recherche sur la grille d'hyperparamètres avec validation croisée
    # Possibilité d'ajuster le nombre de plis de validation croisée (cv) selon vos besoins
    grid_search = GridSearchCV(model, hyperparameters, cv=5)
    # Entraîner le modèle avec les meilleurs hyperparamètres
    grid_search.fit(X_train, y_train)
    # Récupérez le meilleur modèle et les meilleurs paramètres
    best_model = grid_search.best_estimator_
    print('models.py / best_model: ', best_model)
    best_params = grid_search.best_params_
    print('models.py / best_params: ', best_params)
    best_score = grid_search.best_score_
    print('models.py / best_score: ', best_score)
    # Retournez à la fois le meilleur modèle et les meilleurs paramètres
    return best_model, best_params, best_score
