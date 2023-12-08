import pandas as pd

def get_dataset()->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """this function return all of the dataSet of the chat like this
        (X_train, y_train, X_test, y_test)
    """
    # on commence par récupérer le jeu de données
    train_set = pd.read_csv("data/train.csv")

    X_train = train_set["comment_text"]  # Remplacez par les noms de vos colonnes cibles
    y_train = train_set.drop(["id","comment_text"], axis=1)

    X_test = pd.read_csv("data/test.csv")["comment_text"]
    y_test = pd.read_csv("data/test_labels.csv").drop(['id'],axis=1)
    return (X_train, y_train, X_test, y_test)

