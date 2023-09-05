import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import db
import constantes

from preprocess import DataPreprocessor
from models import train_model


def intro():
    return st.set_page_config(
        page_title="ML Playground",
        layout="centered",
        initial_sidebar_state="auto"
    )


def header():
    st.title('Bienvenue')


def sidebar():
    # Obtenir la liste des tables depuis la fonction db_connection() de db.py
    table_list = db.db_connection()
    # Créer une liste déroulante pour sélectionner la table
    selected_table = st.sidebar.selectbox("Sélectionnez un dataset", table_list)
    # Bouton pour créer le DataFrame
    if selected_table:
        # Appeler la fonction pour créer un DataFrame en fournissant le nom de la table
        df, target_type = db.create_dataframe(selected_table)
        # Afficher le DataFrame
        with st.expander("DataFrame"):
            st.write(df)
        with st.expander("Description du DataFrame"):
            description = df.describe()
            st.write(description)
        selected_model = get_algo(target_type)
        with st.expander("GridSearchCV"):
            # Nettoyez les données en utilisant DataPreprocessor
            data_preprocessor = DataPreprocessor(df, target_column='target')
            X_train = data_preprocessor.X_train
            y_train = data_preprocessor.y_train
            # Récupération des résultats de GridSearchCV
            best_model, best_params, best_score = train_model(selected_model, X_train, y_train)
            st.write(best_model)
            st.write(best_params)
            st.write(best_score)
        # with st.expander("Matrice de Corrélation"):
        #     correlation_matrix = df.corr()
        #     plt.figure(figsize=(10, 8))
        #     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        #     # Ne pas afficher le message warning
        #     st.set_option('deprecation.showPyplotGlobalUse', False)
        #     st.pyplot()


def get_algo(target_type):
    # Proposition des algorithmes selon le type de dataset
    if target_type == "int64":
        print("C'est une régression")
        list_algo_regression = constantes.get_algo_reg()
        selected_algo = st.sidebar.selectbox("Sélectionnez un algorithme", list_algo_regression)
        if selected_algo:
            print(f"Vous avez choisi {selected_algo}")
            get_params(selected_algo)
            return selected_algo
    elif target_type == "object":
        print("C'est une classification")
        list_algo_classification = constantes.get_algo_class()
        selected_algo = st.sidebar.selectbox("Sélectionnez un algorithme", list_algo_classification)
        if selected_algo:
            print(f"Vous avez choisi {selected_algo}")
            get_params(selected_algo)
            return selected_algo


def get_params(selected_algo):
    on = st.sidebar.toggle("Modifier la taille de l'entraînement")
    if on:
        print("Ok on modifie")
    st.sidebar.markdown('---')
    on = st.sidebar.toggle("Modifier les hyperparamètres")
    if on:
        # Afficher les hyperparamètres selon le modèle choisi
        hyperparameters = constantes.get_hyperparameters(selected_algo)
        selected_params = {}
        for parametre, values in hyperparameters.items():
            print(parametre, values)
            if all(isinstance(value, bool) for value in values):
                print("isinstance: bool")
                selected_value = st.sidebar.selectbox(parametre, options=values, key=parametre)
            elif all(isinstance(value, (int, float)) for value in values):
                print("isinstance: int, float")
                selected_value = st.sidebar.select_slider(parametre, options=values, key=parametre)
            else:
                print("isinstance: else (str)")
                selected_value = st.sidebar.selectbox(parametre, options=values, key=parametre)
            # Ajouter le paramètre au dictionnaire
            selected_params[parametre] = selected_value
            print("Paramètres sélectionnés :", selected_params[parametre])
    st.sidebar.markdown('---')


def main():
    # uploaded_file = st.sidebar.file_uploader('Chargez votre fichier CSV ici')
    # if uploaded_file:
    #     df = pd.read_csv(uploaded_file)
    #     st.write(df)
    pass


if __name__ == '__main__':
    print('------')
    intro()
    header()
    sidebar()
    main()
