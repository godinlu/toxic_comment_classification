import argparse
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from src.functions import get_dataset
from sklearn.decomposition import TruncatedSVD

#############################################################################

# il faut lancer ce fichier pour pouvoir tester le modèle
# avec des phrases écrites dans le terminal.

#############################################################################
# on importe les données avec la fonction get_dataset()
X_train, y_train, X_test, y_test = get_dataset()

# Créer une instance de TfidfVectorizer
# et transformer les données en vecteurs TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_train)

# on réduit la dimension avec TruncatedSVD pour optimiser le temps de calcul
svd = TruncatedSVD(n_components=100)
X_svd = svd.fit_transform(X)

# on construit le modèle avec la regression logistique
model = MultiOutputClassifier(LogisticRegression())
model.fit(X_svd, y_train)

# Cette fonction va déterminer à l'aide du modèle,
# une prédiction de la classification du commentaire


def classify_text(vectorizer, svd, model, commentaire):
    # Transformer le commentaire en vecteur TF-IDF
    commentaire_vectorise = vectorizer.transform([commentaire])
    # Réduire la dimension avec TruncatedSVD
    commentaire_svd = svd.transform(commentaire_vectorise)

    # Prédire les étiquettes pour le commentaire
    classfication_commentaire = model.predict(commentaire_svd)
    return classfication_commentaire


def classification_result_to_string(result):
    # Dictionnaire pour stocker les categories d'injures
    categories = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ]

    # On passe d'un vecteur 2d en vecteur en 1d
    result_1d = result.ravel()
    # Utiliser numpy pour trouver les indices où la valeur est égale à 1
    indices = np.where(result_1d == 1)[0]

    # Récupérer les labels correspondant aux indice
    is_toxic = [categories[i] for i in indices]
    # Si le vecteur est vide alors le commentaire est bien
    if len(is_toxic) == 0:
        return "normal"
    else:
        return is_toxic


# Fonction qui lance la classification
def main():
    # On utilise argparse pour récuperer le commentaire à classifier
    parser = argparse.ArgumentParser(description="Classification en ligne")
    parser.add_argument("commentaire", nargs="?",
                        help="Commentaire en anglais à classifié")

    args = parser.parse_args()
    # Dans le cas où on lance le programme depuis le terminal
    # avec un commentaire en argument
    if args.commentaire:
        # Classer la phrase/commentaire fournie
        commentaire_categ = classify_text(vectorizer, svd, model, args.phrase)
        print("Classification Result:")
        print(commentaire_categ)
        print("le commentaire est",
              classification_result_to_string(commentaire_categ))
    else:
        # Lire depuis l'entrée standard
        print("Rentré une phrase en anglais (Ctrl+C pour finir):")
        try:
            # on parcoure toutes les lignes du terminal
            for line in sys.stdin:
                line = line.strip()
                # si la ligne n'est pas vide
                if line:
                    # on classifie le commentaire
                    commentaire_categ = classify_text(vectorizer,
                                                      svd, model, line)
                    print(f"phrase : {line}")
                    # on affiche le résultat de la classification
                    print("Classification Result :")
                    print(commentaire_categ)
                    print("le commentaire est",
                          classification_result_to_string(commentaire_categ))
        # si on appuie sur control C , le programme s'arrête
        except KeyboardInterrupt:
            sys.exit(0)


# On lance le programme
if __name__ == "__main__":
    main()
