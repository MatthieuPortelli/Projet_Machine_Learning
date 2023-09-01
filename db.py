import psycopg2


def db_connection():
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
        print("Connexion réussie !")
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
    except psycopg2.Error as e:
        print("Erreur lors de la connexion à la base de données : {}".format(e))
