from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier

STRUCTURE = {
    'regression': {
        'Regression_Lineaire': {
            'model': LinearRegression(),
            'hyperparameters': {
                'fit_intercept': {
                    'description': "Calculer l'ordonnée à l'origine.",
                    'values': [True, False],
                },
                'copy_X': {
                    'description': "Copier les données d'entraînement avant l'ajustement du modèle.",
                    'values': [True, False],
                },
            }
        },
        'Regression_Ridge': {
            'model': Ridge(),
            'hyperparameters': {
                'alpha': {
                    'description': 'Paramètre de régularisation L2.',
                    'values': [0.1, 1.0, 10.0],
                },
                'fit_intercept': {
                    'description': "Calculer l'ordonnée à l'origine.",
                    'values': [True, False],
                },
                'copy_X': {
                    'description': "Copier les données d'entraînement avant l'ajustement du modèle.",
                    'values': [True, False],
                },
                'solver': {
                    'description': "Algorithme pour résoudre le problème d'optimisation.",
                    'values': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                },
            }
        },
    },
    'classification': {
        'KNN_Classification': {
            'model': KNeighborsClassifier(),
            'hyperparameters': {
                'n_neighbors': {
                    'description': 'Nombre de voisins à considérer lors de la prédiction.',
                    'values': [3, 5, 10],
                },
                'weights': {
                    'description': 'Pondération des voisins lors de la prédiction.',
                    'values': ['uniform', 'distance'],
                },
            }
        },
        'Regression_Logistique': {
            'model': LogisticRegression(),
            'hyperparameters': {
                'penalty': {
                    'description': 'Type de pénalité pour la régularisation.',
                    'values': ['l1'],
                },
                'solver': {
                    'description': "Algorithme pour résoudre le problème d'optimisation.",
                    'values': ['liblinear'],
                },
                'C': {
                    'description': "Paramètre d'inversion de la force de régularisation.",
                    'values': [0.1, 1.0, 10.0],
                },
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'hyperparameters': {
                'n_estimators': {
                    'description': "Nombre d'arbres.",
                    'values': [10, 100, 1000],
                },
                'criterion': {
                    'description': "Fonction de critère pour la division des noeuds de l'arbre.",
                    'values': ['gini', 'entropy', 'log_loss'],
                },
                'max_depth': {
                    'description': 'Profondeur maximale de chaque arbre.',
                    'values': [100, 200, 300],
                },
                'min_samples_split': {
                    'description': "Nombre minimum d'échantillons requis pour diviser un noeud interne.",
                    'values': [2, 5, 10],
                },
            }
        },
    }
}


def get_algo_reg():
    list_algo_regression = []
    for model_name, model_info in STRUCTURE['regression'].items():
        list_algo_regression.append(model_name)
        print("Voici la liste reg", list_algo_regression)
    return list_algo_regression


def get_algo_class():
    list_algo_classification = []
    for model_name, model_info in STRUCTURE['classification'].items():
        list_algo_classification.append(model_name)
        print("Voici la liste class", list_algo_classification)
    return list_algo_classification


def get_hyperparameters(model_name):
    if model_name in STRUCTURE['regression']:
        return STRUCTURE['regression'][model_name]['hyperparameters']
    elif model_name in STRUCTURE['classification']:
        return STRUCTURE['classification'][model_name]['hyperparameters']
    else:
        return []
