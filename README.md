# Utilisation class DataPreprocessor
*Déclaration des données*  
data = pd.read_csv("diabete.csv")  

*Instance de la classe DataPreprocessor*  
preprocessor = DataPreprocessor(data, target_column='target')  

*Données nettoyées et divisées*  
X_train, X_test, y_train, y_test = preprocessor.X_train, preprocessor.X_test, preprocessor.y_train, preprocessor.y_test  