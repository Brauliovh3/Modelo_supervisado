import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar datos simples (ejemplo: datos de flores Iris)
def cargar_datos():
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

# Entrenamiento y evaluación de un árbol de decisión
def modelo_arbol_decision():
    X, y = cargar_datos()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Exactitud:', accuracy_score(y_test, y_pred))
    print('Matriz de confusión:\n', confusion_matrix(y_test, y_pred))
    print('Reporte de clasificación:\n', classification_report(y_test, y_pred))

if __name__ == '__main__':
    modelo_arbol_decision()
