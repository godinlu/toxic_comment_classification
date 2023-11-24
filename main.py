import pandas as pd


data = pd.read_csv("data/train.csv")
print(data.shape)


# Création d'une instance de CountVectorizer
vectorizer = CountVectorizer()

# Transformation des documents en vecteurs
X = vectorizer.fit_transform(data["comment_text"])

# Affichage des noms des fonctionnalités (mots)
feature_names = vectorizer.get_feature_names_out()
print(feature_names.shape)

first_row = X[0].toarray()
print(first_row)
print(sum(first_row[0])) 


