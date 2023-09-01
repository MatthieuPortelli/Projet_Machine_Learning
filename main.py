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
    # st.sidebar.caption('Choisir le dataset')
    # Obtenir la liste des tables depuis la fonction db_connection() de db.py
    table_list = db.db_connection()

    if isinstance(table_list, list):
        # Créer une liste déroulante pour sélectionner la table
        selected_table = st.sidebar.selectbox("Sélectionnez un dataset", table_list)
        # Bouton pour créer le DataFrame
        if st.sidebar.button("Importer les données"):
            if selected_table:
                # Appeler la fonction pour créer un DataFrame
                df, target_type = db.create_dataframe(selected_table)
                print(df)
                print(target_type)
                # Afficher le DataFrame
                st.write("DataFrame:")
                st.write(df)
    elif isinstance(table_list, str):
        st.sidebar.write(table_list)


if __name__ == '__main__':
    intro()
    header()
    sidebar()
