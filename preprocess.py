from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, dataframe, target_column, test_size=0.2):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.data = dataframe.copy()
        self.encode_categorical_features()
        self.handle_missing_values()
        self.remove_rows_with_outliers_iqr()
        self.standardize_data()
        self.split_data(target_column, test_size)

    def encode_categorical_features(self):
        label_encoder = LabelEncoder()
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                self.data[column] = label_encoder.fit_transform(self.data[column])

    def handle_missing_values(self, strategy='median'):
        if strategy == 'median':
            self.data.fillna(self.data.median(), inplace=True)
        elif strategy == 'mean':
            self.data.fillna(self.data.mean(), inplace=True)
        else:
            raise ValueError("Stratégie d'imputation non valide.")

    def remove_rows_with_outliers_iqr(self, threshold=1.5):
        for column in self.data.columns:
            if self.data[column].dtype != 'object':
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]

    def standardize_data(self):
        scaler = StandardScaler()
        self.data[self.data.columns] = scaler.fit_transform(self.data[self.data.columns])

    def split_data(self, target_column, test_size=0.2):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
