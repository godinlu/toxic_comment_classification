import argparse
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from src.functions import get_dataset
from sklearn.decomposition import TruncatedSVD

X_train, y_train, X_test, y_test = get_dataset()

# Créer une instance de TfidfVectorizer et transformer les données en vecteurs TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_train)
    
# on réduit la dimension avec TruncatedSVD pour optimiser le temps de calcul
svd = TruncatedSVD(n_components=100)
X_svd = svd.fit_transform(X) 

model = MultiOutputClassifier(LogisticRegression())
model.fit(X_svd, y_train)

print("consigne : attendre les 5 premiers warning avant de rentrer la phrase")


    

#cette fonction va déterminer à l'aide du modèle , une prédiction de la classification de la phrase
def classify_text(vectorizer, svd, model, phrase):
    # Transformer la phrase en vecteur TF-IDF
    phrase_vectorise = vectorizer.transform([phrase])
    
    # Réduire la dimension avec TruncatedSVD
    phrase_svd = svd.transform(phrase_vectorise)
    
    # Prédire les étiquettes pour la phrase
    classfication_phrase = model.predict(phrase_svd)
    return classfication_phrase

def classification_result_to_string(result):
    # Dictionnaire pour mapper les positions à des labels
    genre = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ]
    
     # Utiliser numpy pour trouver les indices où la valeur est égale à 1
    indices = np.where(result == 1)[0]
    
    # Récupérer les labels correspondant aux indices
    is_toxic = [genre[i] for i in indices]
    #si le vecteur est vide alors le commentaire est bien
    if len(is_toxic) == 0:
        return "normal"
    else:
        return  is_toxic

def main():
    #On utilise argparse pour faire la classification
    parser = argparse.ArgumentParser(description="Classification en ligne")
    parser.add_argument("phrase", nargs="?", help="Phrase en anglais à classifié")

    args = parser.parse_args()

    if args.phrase:
        
        
        # Classer la phrase fournie
        result = classify_text(vectorizer, svd, model, args.phrase)
        print("Classification Result:")
        print(result)
    else:
        # Lire depuis l'entrée standard
        print("Rentré une phrase en anglais (Ctrl+C pour finir):")
        try:
            
            for line in sys.stdin:
                line = line.strip()
                if line:
                    result = classify_text(vectorizer, svd, model, line)
                    print(f"phrase : {line}")
                    print("Classification Result :")
                    print(result)
                    print("le commentaire est ",classification_result_to_string(result))
        except KeyboardInterrupt:
            sys.exit(0)

if __name__ == "__main__":
    main()