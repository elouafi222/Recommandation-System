import random as rd
import csv
import pandas as ps
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#### generate data for movie


def generate_Date_for_movie(nemberline=10, fichier_csv="data.csv"):
    "this function genearate data for list of users"
    with open(fichier_csv, "w", newline="") as csvfile:
        # Création d'un objet writer
        writer = csv.writer(csvfile)

        # Écriture de l'en-tête
        writer.writerow(["x1", "x2", "y"])

        # Écriture des données

        for _ in range(0, nemberline):
            x1 = round(rd.uniform(0, 1), 2)
            x2 = round(rd.uniform(0, 1), 2)
            a = 1
            b = 1
            if x1 > 0.5:
                a = 5
            if x2 > 0.5:
                b = 5
            y = (a * x1 + b * x2) / 2
            donnees = [x1, x2, y]
            writer.writerow(donnees)


# generate_Date_for_movie(nemberline=10000)

### Delete ligne how contains the same value


def delete_ligne_redandante(fichier_csv="data.csv"):
    """"""
    dataset = ps.read_csv(fichier_csv)

    return dataset.drop_duplicates()


datasetClening = delete_ligne_redandante()


def creating_model(dataset):
    """"""
    X = dataset[["x1", "x2"]]
    y = dataset["y"]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialiser le modèle de régression linéaire
    modele = LinearRegression()

    # Entraîner le modèle sur l'ensemble d'entraînement
    modele.fit(X_train, y_train)

    # Prédire les valeurs sur l'ensemble de test
    y_pred = modele.predict(X_test)

    # Calculer l'erreur quadratique moyenne
    mse = mean_squared_error(y_test, y_pred)
    return modele, mse


modele, mse = creating_model(datasetClening)

# Nouvelles features pour la prédiction
nouvelles_features = [
    [1, 1]
]  # Remplacez valeur_x1 et valeur_x2 par les valeurs réelles que vous voulez prédire

# Prédire la nouvelle valeur
# prediction = modele.predict(nouvelles_features)

# print(prediction, mse)


############# ploter le modele avec les donnes ##########

df = datasetClening


# Créer une figure en 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Tracer le nuage de points avec les données réelles
# Tracer le nuage de points avec les données réelles
scatter = ax.scatter(
    df["x1"],
    df["x2"],
    df["y"],
    cmap="viridis",
    label="Données réelles",
    alpha=0.7,
    c=df["y"],
)


# Tracer le modèle de régression linéaire en utilisant une surface
x1_values = np.linspace(df["x1"].min(), df["x1"].max(), 100)
x2_values = np.linspace(df["x2"].min(), df["x2"].max(), 100)
x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
y_pred_grid = modele.predict(np.c_[x1_grid.ravel(), x2_grid.ravel()])
y_pred_grid = y_pred_grid.reshape(x1_grid.shape)

surface = ax.plot_surface(
    x1_grid,
    x2_grid,
    y_pred_grid,
    cmap="plasma",
    alpha=0.5,
    label="Modèle de régression",
)

# Ajouter des étiquettes et une légende
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.set_title("Modèle de régression linéaire en 3D")
fig.colorbar(scatter, label="y (Variable cible)")
# fig.colorbar(surface, label="Prédictions du modèle")
ax.legend()

# Afficher le graphique
plt.show()
