import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Cargar datos (ejemplo con datos generados)
def cargar_datos():
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

# Preprocesamiento (placeholder)
def preprocesar(X):
    # Aquí puedes agregar pasos de preprocesamiento si es necesario
    return X

# Entrenamiento del modelo
def entrenar_modelo(X_train, y_train):
    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_train, y_train)
    return modelo

# Evaluación del modelo
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Guardar resultados
def guardar_resultados(y_test, y_pred):
    resultados = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    resultados.to_csv('resultados.csv', index=False)

if __name__ == '__main__':
    X, y = cargar_datos()
    X = preprocesar(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = entrenar_modelo(X_train, y_train)
    y_pred = modelo.predict(X_test)
    evaluar_modelo(modelo, X_test, y_test)
    guardar_resultados(y_test, y_pred)
    # Ejemplo de visualización
    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred)
    plt.title('Predicciones')
    plt.xlabel(X_test.columns[0])
    plt.ylabel(X_test.columns[1])
    plt.show()
