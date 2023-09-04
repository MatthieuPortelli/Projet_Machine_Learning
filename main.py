import streamlit as st
import db
import constantes


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
        st.write("DataFrame :")
        st.write(df)
        get_algo(target_type)


def get_algo(target_type):
    # Proposition des algorithmes selon le type de dataset
    if target_type == "int64":
        print("C'est une régression")
        list_algo_regression = constantes.get_algo_reg()
        selected_algo = st.sidebar.selectbox("Sélectionnez un algorithme", list_algo_regression)
        if selected_algo:
            print(f"Vous avez choisi {selected_algo}")
            get_params(selected_algo)

    elif target_type == "object":
        print("C'est une classification")
        list_algo_classification = constantes.get_algo_class()
        selected_algo = st.sidebar.selectbox("Sélectionnez un algorithme", list_algo_classification)
        if selected_algo:
            print(f"Vous avez choisi {selected_algo}")
            get_params(selected_algo)


def get_params(selected_algo):
    on = st.sidebar.toggle("Modifier les hyperparamètres")
    if on:
        # Afficher les hyperparamètres selon le modèle choisi
        hyperparameters = constantes.get_hyperparameters(selected_algo)
        selected_params = {}
        for parametre, values in hyperparameters.items():
            print(parametre, values)
            if all(isinstance(value, bool) for value in values):
                print("rentre dans bool")
                selected_value = st.sidebar.radio(parametre, options=values, key=parametre)
            elif all(isinstance(value, (int, float)) for value in values):
                print("rentre dans int float")
                selected_value = st.sidebar.select_slider(parametre, options=values, key=parametre)
            else:
                print("rentre dans else (str)")
                selected_value = st.sidebar.radio(parametre, options=values, key=parametre)

            # Ajouter le paramètre au dictionnaire
            selected_params[parametre] = selected_value
            print("paramètres sélectionnés :", selected_params[parametre])


if __name__ == '__main__':
    intro()
    header()
    sidebar()
