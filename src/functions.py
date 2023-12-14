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
    test_set.loc(test_set['toxic'] != -1, inplace=True)

    X_test = train_set["comment_text"]
    y_test = train_set.drop(["id","comment_text"], axis=1)

    return (X_train, y_train, X_test, y_test)

