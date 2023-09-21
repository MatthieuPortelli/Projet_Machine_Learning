# Projet_Machine_Learning

**Projet_Machine_Learning** est une application de démonstration pour explorer et expérimenter avec des modèles de machine learning. L'application vous permet de charger des ensembles de données, de choisir parmi une variété d'algorithmes d'apprentissage automatique, de régler les hyperparamètres, de visualiser les résultats, et même de générer des rapports PDF pour documenter vos expériences.

## Fonctionnalités

- **Chargement des Données** : Sélectionnez un ensemble de données parmi deux CSV disponibles.

- **Sélection des Colonnes** : Choisissez les colonnes spécifiques du jeu de données à utiliser, y compris la colonne cible.

- **Hyperparamètres Optimaux** : Utilisez GridSearchCV pour trouver les meilleurs hyperparamètres pour votre modèle.

- **Visualisations** : Visualisez la matrice de corrélation, les données de la grille, les métriques de performance, et plus encore.

- **Rapports PDF** : Générez des rapports PDF contenant des informations sur le modèle sélectionné, les hyperparamètres optimaux, les métriques de performance et des visualisations.

## Comment Utiliser

1. Sélectionnez un jeu de données depuis la liste déroulante dans la barre latérale.

2. Choisissez les colonnes sur lesquelles appliquer le modèle, en n'oubliant pas de sélectionner la colonne cible.

3. Sélectionnez un algorithme d'apprentissage automatique parmi les options disponibles.

4. Utilisez GridSearchCV pour ajuster les hyperparamètres du modèle si nécessaire.

5. Explorez les visualisations et les métriques pour évaluer les performances du modèle.

6. Générez un rapport PDF en cliquant sur le bouton correspondant.

## Installation

1. Clonez ce référentiel sur votre machine locale :

git clone https://github.com/votre-utilisateur/ml-playground.git

2. Accédez au répertoire du projet :

cd ml-playground

3. Installez les dépendances nécessaires :

pip install -r requirements.txt

4. Exécutez l'application Streamlit :

streamlit run main.py

## Exigences

- Python 3.6+
- Dépendances répertoriées dans `requirements.txt`

---

**ML Playground** - Une application de démonstration pour l'apprentissage automatique.
