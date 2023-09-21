import os
import pandas as pd


# Fonction pour charger un fichier CSV et le transformer en DataFrame
def load_csv_to_dataframe(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Supprimer la colonne "id" si elle existe
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # Vérifier la présence d'une colonne "target"
    if "target" not in df.columns:
        return f"Votre dataset doit contenir une colonne 'target' pour entraîner et évaluer correctement votre modèle."
    # Inspecter le type de données de la colonne "target"
    target_type = df["target"].dtype
    return df, target_type


# Fonction pour lire les csv déjà présents
def list_csv_files(csv_directory):
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    return csv_files
