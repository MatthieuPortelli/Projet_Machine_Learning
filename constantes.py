from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier

STRUCTURE = {
    'regression': {
        'Regression_Lineaire': {
            'model': LinearRegression(),
            'hyperparameters': {
                'fit_intercept': [True, False],
                'copy_X': [True, False],
            }
        },
        'Régression_Ridge': {
            'model': Ridge(),
            'hyperparameters': {
                'alpha': [0.1, 1.0, 10.0],
                'fit_intercept': [True, False],
                'normalize': [True, False],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            }
        },
    },
    'classification': {
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'hyperparameters': {
                'n_estimators': [100, 200, 500],
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_depth': [100, 200, 300],
                'min_samples_split': [2, 5, 10]
            }
        },
        'KNN_Classification': {
            'model': KNeighborsClassifier(),
            'hyperparameters': {
                'n_neighbors': [3, 5, 10],
                'weights': ['uniform', 'distance'],
            }
        },
        'Regression_Logistique': {
            'model': LogisticRegression(),
            'hyperparameters': {
                'penalty': ['l1'],  # Utilisez 'l1' comme pénalité
                'solver': ['liblinear'],  # Spécifiez le solver 'liblinear'
                'C': [0.1, 1.0, 10.0],
            }
        },
    }
}


def get_algo_reg():
    list_algo_regression = []
    for model_name, model_info in STRUCTURE['regression'].items():
        list_algo_regression.append(model_name)
        print("voici la liste reg", list_algo_regression)
    return list_algo_regression


def get_algo_class():
    list_algo_classification = []
    for model_name, model_info in STRUCTURE['classification'].items():
        list_algo_classification.append(model_name)
        print("voici la liste class", list_algo_classification)
    return list_algo_classification


def get_hyperparameters(model_name):
    if model_name in STRUCTURE['regression']:
        return STRUCTURE['regression'][model_name]['hyperparameters']
    elif model_name in STRUCTURE['classification']:
        return STRUCTURE['classification'][model_name]['hyperparameters']
    else:
        return {}
