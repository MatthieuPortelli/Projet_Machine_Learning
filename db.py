from typing import Tuple, Union

import pandas as pd
import psycopg2


def create_connection():
    db_params = {
        'host': 'ec2-34-247-94-62.eu-west-1.compute.amazonaws.com',
        'database': 'd4on6t2qk9dj5a',
        'user': 'nxebpjsgxecqny',
        'port': '5432',
        'password': '1da2f1f48e4a37bf64e3344fe7670a6547c169472263b62d042a01a8d08d2114'
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
        return "Erreur lors de la connexion à la base de données : {}".format(e)


def create_dataframe(table_name: str) -> Union[Tuple[pd.DataFrame, str], pd.DataFrame, None]:
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

            if df["target"]:
                # Inspecter le type de données de la colonne "target"
                target_type = df["target"].dtype
                # Fermer le curseur
                cursor.close()
                # Fermer la connexion
                connection.close()
                # Retourner le dataframe et le type de données de "target"
                return df, target_type
            return df
    except Exception as e:
        print("Erreur lors de la récupération des données de la table : {}".format(e))
        return None
