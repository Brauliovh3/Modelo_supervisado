"""
Modelo supervisado de machine learning con regresión logística sobre el dataset Iris.

Funcionamiento paso a paso:
1. Se cargan datos etiquetados (Iris), donde cada flor tiene características y una especie conocida.
2. Se dividen los datos en entrenamiento y prueba.
3. El modelo aprende la relación entre características y especie usando los datos de entrenamiento.
4. Se predicen las especies de las flores de prueba.
5. Se evalúa el desempeño del modelo con métricas y una matriz de confusión.

- Entrenamiento: El modelo ajusta sus parámetros para predecir correctamente la especie.
- Prueba: Se mide qué tan bien predice en datos nuevos.
- Visualización: Se muestra una matriz de confusión para ver aciertos y errores.

Este flujo es típico en problemas de clasificación supervisada.
"""
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
    # Visualización más entendible: matriz de confusión
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y), yticklabels=set(y))
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()
