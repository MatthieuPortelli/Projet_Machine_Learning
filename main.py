import streamlit as st
import db


def intro():
    return st.set_page_config(
        page_title="ML Playground",
        layout="centered",
        initial_sidebar_state="auto"
    )


def header():
    st.title('Bienvenue')


def sidebar():
    st.sidebar.caption('Choisir le dataset')
    # Obtenir la liste des tables depuis la fonction db_connection() de db.py
    table_list = db.db_connection()
    # Créer une liste déroulante pour sélectionner la table
    selected_table = st.sidebar.selectbox("Sélectionnez une table", table_list)
    print(selected_table)


if __name__ == '__main__':
    intro()
    header()
    sidebar()
