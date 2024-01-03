import math
import numpy as np

## In this exemple i select the favorit between users and i sugjest
## a new topic for him

users_interests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"],
]

## une premiere methode pour connaitre les centres d'interet d'un utilisateur
## et rechercher d’autres utilisateurs qui présentent des similitudes avec lui et à lui
##suggérer les sujets qui intéressent ces utilisateurs.
## il faux danc un moyenne de calculé le degre de similtitude
## entre deux utilisateur on vat utiliser un indicateur appelé « similarité cosinus »
##


def chercherSujet(listE):
    listedesSujet = list()
    for user in listE:
        for users_interests in user:
            if users_interests not in listedesSujet:
                listedesSujet.append(users_interests)

    return listedesSujet


def cosine_similarity(v, w):
    return np.dot(v, w) / math.sqrt(np.dot(v, v) * np.dot(w, w))


listElmt = chercherSujet(users_interests)


def matriceDeFavorie():
    nbrcolone = len(listElmt)
    nbrligne = len(users_interests)
    matrice = [[0 for j in range(nbrcolone)] for i in range(nbrligne)]

    for i in range(nbrligne):
        for j in range(nbrcolone):
            if listElmt[j] in users_interests[i]:
                matrice[i][j] = 1

    return matrice


matrice = matriceDeFavorie()

## voir le centre d'interer de user 1 avec les autre utilisateur


def centreInterer():
    listFavori = list()
    for ligne in matrice:
        listFavori.append(cosine_similarity(matrice[0], ligne))
    return listFavori


listeFavorie = centreInterer()

print(listeFavorie)
