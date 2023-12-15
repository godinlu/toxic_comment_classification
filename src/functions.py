import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report,roc_curve

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

def get_preprocess_dataset()->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """this function return all of the dataSet with the vectorizing and the reduction of dimension 
        of the chat like this
        (X_train, y_train, X_test, y_test)
    """
    return (
        pd.read_csv("data/X_train.csv", header=None), 
        pd.read_csv("data/y_train.csv"),
        pd.read_csv("data/X_test.csv", header=None),
        pd.read_csv("data/y_test.csv")
    )

def classif_result(model, y_train, y_test, X_svd, X_test_svd,train = True):
    individual_estimators = model.estimators_
    y_pred_proba = []

    # Récuperer les probabilités pour chaque classe (données d'entraînement)
    if train :
        for estimator, target in zip(individual_estimators, y_train.columns):
            y_pred_proba.append(estimator.predict_proba(X_svd)[:, 1])

        y_pred_proba = np.array(y_pred_proba).T

    # Récuperer les probabilités pour chaque classe (données de test)
    y_test_pred_proba = model.predict_proba(X_test_svd)
    
    y_test_pred_proba = np.array([class_prob[:, 1] for class_prob in y_test_pred_proba]).T

    y_pred = model.predict(X_svd)
    y_test_pred = model.predict(X_test_svd)

    if train:
        accuracy = accuracy_score(y_train, y_pred)
        report = classification_report(y_train, y_pred, zero_division=1)
    
    accuracy_test = accuracy_score(y_test, y_test_pred)
    report_test = classification_report(y_test, y_test_pred, zero_division=1)
    if train:
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")


    print(f"Accuracy Test: {accuracy_test}")
    print(f"Classification Report Test:\n{report_test}")

    plt.figure(figsize=(12, 5))

    # Courbes ROC pour les données d'entraînement
    if train :
        plt.subplot(1, 2, 1)
        for i in range(6):
            fpr, tpr, thresholds = roc_curve(y_train.iloc[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'Courbe ROC de la classe {i} (Train)')

        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title('Courbes ROC de chaque classe (Train)')
        plt.legend()

    # Courbes ROC pour les données de test
    plt.subplot(1, 2, 2)
    for i in range(6):
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test.iloc[:, i], y_test_pred_proba[:, i])
        plt.plot(fpr_test, tpr_test, label=f'Courbe ROC de la classe {i} (Test)')

    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC de chaque classe (Test)')
    plt.legend()

    plt.tight_layout()
    plt.show()
