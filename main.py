import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import db
import constantes
from report import generate_pdf_report
from visualisations import visualize_selected_model
from preprocess import DataPreprocessor
from models import train_model, evaluate_model
import base64


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
        expander = st.sidebar.expander("Sélectionnez des colonnes spécifiques")
        # columns = []  # Initialisation à une liste vide
        try:
            columns = expander.multiselect(
                "Choissisez les colonnes sur lesquelles appliquer le modèle, n'oubliez pas la colonne target", df.columns)
            if columns:
                df = df[columns]
            with st.expander("**DataFrame**"):
                st.write(df)
            with st.expander("**Description du DataFrame**"):
                description = df.describe()
                st.write(description)
            selected_model, model_family, selected_params, test_size = get_algo(target_type)
            if test_size is None:
                data_preprocessor = DataPreprocessor(df, target_column='target')
            else:
                data_preprocessor = DataPreprocessor(df, target_column='target', test_size=test_size)
            processed_dataframe = data_preprocessor.processed_data
            X_train = data_preprocessor.X_train
            X_test = data_preprocessor.X_test
            y_train = data_preprocessor.y_train
            y_test = data_preprocessor.y_test
            with st.expander("**Matrice de Corrélation**"):
                correlation_matrix = processed_dataframe.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
                # Ne pas afficher le message warning
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            with st.expander("**GridSearchCV**"):
                # Récupération des résultats de GridSearchCV
                best_model, best_params, best_score = train_model(selected_model, X_train, y_train, selected_params)
                st.write("**Les hyperparamètres optimaux sont :**")
                for param_name, param_value in best_params.items():
                    st.markdown(f"{param_name} : <span style='color: #FFB923'>{param_value}</span>",
                                unsafe_allow_html=True)
                st.write("**Le score obtenu est :**")
                st.write(f'<span style="color: #FFB923;"> {best_score}</span>', unsafe_allow_html=True)
            with st.expander("**Métriques**"):
                metrics, y_pred = evaluate_model(best_model, X_test, y_test, model_family)
                for metric_name, metric_value in metrics.items():
                    st.markdown(f"<span style='color: #FFB923'>{metric_name}</span> : {metric_value}",
                                unsafe_allow_html=True)
                    if metric_name == "Accuracy":
                        st.markdown("_Proportion de prédictions correctes parmi toutes les prédictions effectuées._")
                    elif metric_name == "Precision":
                        st.markdown("_Proportion de vrais positifs parmi toutes les prédictions positives._")
                    elif metric_name == "Recall":
                        st.markdown("_Proportion de vrais positifs parmi toutes les instances réelles positives._")
                    elif metric_name == "F1-Score":
                        st.markdown(
                            "_Moyenne harmonique de la précision et du rappel, une mesure globale de la performance._")
                    elif metric_name == "MSE":
                        st.markdown(
                            "_Mean Squared Error (Erreur quadratique moyenne), une mesure de la précision des prédictions._")
                    elif metric_name == "R²":
                        st.markdown(
                            "_Coefficient de détermination R², une mesure de l'ajustement du modèle aux données._")
                    else:
                        st.markdown(f"{metric_name} : {metric_value}")
            with st.expander("**Visualisations**"):
                fig_1, fig_2 = visualize_selected_model(selected_model, y_test, y_pred)
                st.pyplot(fig_1)
                st.pyplot(fig_2)
            # Gestion du pdf
            generate_report = st.button("Générer le rapport PDF")
            if generate_report:
                pdf_data = generate_pdf_report(selected_model, best_params, best_score, metrics, fig_1, fig_2)
                # Affichez le lien de téléchargement pour le PDF généré
                st.success("Le rapport PDF a été généré avec succès. Cliquez ci-dessous pour le télécharger.")
                st.markdown(get_binary_file_downloader_html(pdf_data, "Rapport", "Télécharger le PDF"),
                            unsafe_allow_html=True)
        except Exception:
            st.sidebar.error(f"Il y a une erreur de sélection. Sélectionnez une ou plusieurs colonnes, colonne 'target' en dernière.")


def get_algo(target_type):
    # Proposition des algorithmes selon le type de dataset
    selected_algo, model_family, selected_params = None, None, None
    if target_type == "int64":
        model_family = 'regression'
        list_algo_regression = constantes.get_algo_reg()
        selected_algo = st.sidebar.selectbox("Sélectionnez un algorithme", list_algo_regression, key='2')
    elif target_type == "object":
        model_family = 'classification'
        list_algo_classification = constantes.get_algo_class()
        selected_algo = st.sidebar.selectbox("Sélectionnez un algorithme", list_algo_classification, key='3')
    if selected_algo and model_family:
        selected_params, test_size = get_params(selected_algo)
        return selected_algo, model_family, selected_params, test_size


def get_params(selected_algo):
    selected_params = {}
    test_size = None
    st.sidebar.markdown('---')
    on_test_size = st.sidebar.toggle("Modifier la taille de test")
    if on_test_size:
        test_size = st.sidebar.slider(" ", 0.5, 0.95, 0.2, 0.05)
    st.sidebar.markdown('---')
    on_grid_search = st.sidebar.toggle("Désactiver GridSearchCV, modifier les hyperparamètres")
    if on_grid_search:
        # Afficher les hyperparamètres selon le modèle choisi
        hyperparameters = constantes.get_hyperparameters(selected_algo)
        for parametre, values in hyperparameters.items():
            if all(isinstance(value, (int, float)) for value in values):
                selected_value = st.sidebar.select_slider(parametre, options=values['values'], key=parametre, help=values['description'])
            elif all(isinstance(value, bool) for value in values):
                selected_value = st.sidebar.selectbox(parametre, options=values['values'], key=parametre, help=values['description'])
            else:
                selected_value = st.sidebar.selectbox(parametre, options=values['values'], key=parametre, help=values['description'])
            # Ajouter le paramètre au dictionnaire
            selected_params[parametre] = [selected_value]
            # print("Paramètres sélectionnés :", selected_params[parametre])
    st.sidebar.markdown('---')
    return selected_params, test_size


def get_binary_file_downloader_html(bin_data, file_label='File', btn_label='Download'):
    b64 = base64.b64encode(bin_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_label}.pdf">{btn_label}</a>'
    return href


def main():
    pass


if __name__ == '__main__':
    intro()
    header()
    sidebar()
    main()
