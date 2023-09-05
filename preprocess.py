from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, dataframe, target_column, test_size=0.2):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.data = dataframe.copy()
        self.handle_missing_values()
        self.remove_rows_with_outliers_iqr()
        self.encode_categorical_features()
        self.standardize_data()
        self.split_data(target_column, test_size)
        self.processed_data = self.data.copy()

    def handle_missing_values(self, strategy='median'):
        # Sélectionnez uniquement les colonnes numériques
        numeric_columns = self.data.select_dtypes(include=['number']).columns
        if strategy == 'median':
            # Utilisez la médiane pour imputer les valeurs manquantes uniquement dans les colonnes numériques
            self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].median())
        elif strategy == 'mean':
            # Utilisez la moyenne pour imputer les valeurs manquantes uniquement dans les colonnes numériques
            self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].mean())
        else:
            raise ValueError("Stratégie d'imputation non valide.")

    def remove_rows_with_outliers_iqr(self, threshold=1.5):
        # Sélectionnez uniquement les colonnes numériques
        numeric_columns = self.data.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            if self.data[column].dtype != 'object':
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]

    def encode_categorical_features(self):
        label_encoder = LabelEncoder()
        for column in self.data.columns:
            self.data[column] = label_encoder.fit_transform(self.data[column])

    def standardize_data(self):
        scaler = StandardScaler()
        self.data[self.data.columns[:-1]] = scaler.fit_transform(self.data[self.data.columns[:-1]])

    def split_data(self, target_column, test_size=0.2):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
