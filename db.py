from typing import Tuple, Union
import pandas as pd
import psycopg2


def create_connection():
    db_params = {
        'host': 'ec2-34-247-16-250.eu-west-1.compute.amazonaws.com',
        'database': 'd1fqoktf0gl90p',
        'user': 'xpfxvuvcndvbve',
        'port': '5432',
        'password': '43b5e0de771549a5cb3117f84603628575b85328a0aecd350b017dcbf4534ddb'
    }

    try:
        # Connexion à la base de données
        connection = psycopg2.connect(**db_params)
        return connection
    except psycopg2.Error as e:
        print("Erreur lors de la connexion à la base de données : {}".format(e))
        return None


def db_connection():
    try:
        connection = create_connection()
        if connection:
            # Créer un curseur
            cursor = connection.cursor()
            # Requête SQL pour obtenir la liste de toutes les tables
            table_list_query = """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    """
            # Exécutez la requête
            cursor.execute(table_list_query)
            # Récupérer la liste des noms de tables
            table_list = [table[0] for table in cursor.fetchall()]
            # Fermer le curseur
            cursor.close()
            # Fermer la connexion
            connection.close()
            # Retourner la liste des noms de tables
            return table_list
    except Exception as e:
        # Retourner l'erreur
        print("Erreur lors de la connexion à la base de données : {}".format(e))
        return None


def create_dataframe(table_name: str) -> Union[Tuple[pd.DataFrame, str], None]:
    """
    explication de la methode

    :param table_name: ce que c'est
    :type table_name: str
    :except decrire l'exception
    :return: ce que ca retourne
    :rtype: Union[Tuple[pd.DataFrame, str], pd.DataFrame, None]
    """
    try:
        connection = create_connection()
        if connection:
            # Créer un curseur
            cursor = connection.cursor()
            # Requête SQL pour sélectionner toutes les données de la table spécifiée
            query = f"SELECT * FROM {table_name}"
            # Exécutez la requête
            cursor.execute(query)
            # Récupérer les données dans un DataFrame
            df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

            # Supprimer la colonne "id" si elle existe
            if "id" in df.columns:
                df = df.drop(columns=["id"])

            if "target" not in df.columns:
                print("Votre dataset doit contenir une colonne 'target'")

            # Inspecter le type de données de la colonne "target"
            target_type = df["target"].dtype
            # Fermer le curseur
            cursor.close()
            # Fermer la connexion
            connection.close()
            # Retourner le dataframe et le type de données de "target"
            return df, target_type
    except Exception as e:
        print("Erreur lors de la récupération des données de la table : {}".format(e))
        return None
