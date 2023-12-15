import pandas as pd

def get_dataset()->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """this function return all of the dataSet of the chat like this
        (X_train, y_train, X_test, y_test)
    """
    # on commence par récupérer le jeu de données
    train_set = pd.read_csv("data/train.csv")

    X_train = train_set["comment_text"]  # Remplacez par les noms de vos colonnes cibles
    y_train = train_set.drop(["id","comment_text"], axis=1)

    # Jointure sur la colonne "ID"
    test_set = pd.merge(pd.read_csv("data/test.csv"), pd.read_csv("data/test_labels.csv"), on='id')
    test_set = test_set[test_set['toxic'] != -1]

    X_test = test_set["comment_text"]
    y_test = test_set.drop(["id","comment_text"], axis=1)

    return (X_train, y_train, X_test, y_test)

def get_pit()->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """this function return all of the dataSet of the chat like this
        (X_train, y_train, X_test, y_test)
    """
    # on commence par récupérer le jeu de données
    train_set = pd.read_csv("data/train.csv")




    # Sélectionner les lignes qui ont des valeurs non nulles dans au moins une colonne
    non_zero_rows = train_set[(train_set.iloc[:, 2:] != 0).any(axis=1)]

    # Sélectionner les lignes qui ont que des zéros et en prendre environ 50%
    only_zeros = train_set[(train_set.iloc[:, 2:] == 0).all(axis=1)].sample(frac=0.5)

    # Concaténer les deux ensembles de données
    result = pd.concat([non_zero_rows, only_zeros])
    print(result.shape)


    X_train = train_set["comment_text"]  # Remplacez par les noms de vos colonnes cibles
    y_train = train_set.drop(["id","comment_text"], axis=1)





    # Jointure sur la colonne "ID"
    test_set = pd.merge(pd.read_csv("data/test.csv"), pd.read_csv("data/test_labels.csv"), on='id')
    test_set = test_set[test_set['toxic'] != -1]

    X_test = test_set["comment_text"]
    y_test = test_set.drop(["id","comment_text"], axis=1)

    return (X_train, y_train, X_test, y_test)