from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

STRUCTURE = {
    'regression': {
        'Regression_Lineaire': {
            'model': LinearRegression(),
            'hyperparameters': {
                'fit_intercept': [True, False],
                'copy_X': [True, False],
            }
        },
        'Regression_Test': {
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
        'Classification_Test': {
        }
    }
}
