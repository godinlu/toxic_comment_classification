# ce script peremet de créer 4 fichiers de données à partir des données
# d'origine ces fichiers sont vectorizer puis réduit on terme de dimension

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.multioutput import MultiOutputClassifier
from functions import get_dataset


X_train, y_train, X_test, y_test = get_dataset()

# Création d'une instance de TfidfVectorizer
vectorizer = TfidfVectorizer()

# Transformation des donnée de tout les comment_text en vecteurs
vectorize_X_train = vectorizer.fit_transform(X_train)
vectorize_X_test = vectorizer.transform(X_test)

svd = TruncatedSVD(n_components=200)
svd_X_train = svd.fit_transform(vectorize_X_train)
svd_X_test = svd.fit_transform(vectorize_X_test)

np.savetxt('data/X_train.csv', svd_X_train, delimiter=',', fmt='%f')
np.savetxt('data/X_test.csv', svd_X_test, delimiter=',', fmt='%f')
