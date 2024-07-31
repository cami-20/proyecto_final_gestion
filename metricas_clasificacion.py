import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

def load_classification_problem(filename):
    # Cargar el dataset
    df_clas = pd.read_csv(filename)
    
    # Convertir variables categóricas en numéricas
    label_encoders = {}
    for column in df_clas.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_clas[column] = le.fit_transform(df_clas[column])
        label_encoders[column] = le
    
    # Verificar las etiquetas codificadas
    print("Categorías en 'anomaly':", df_clas['anomaly'].unique())
    
    # Separar las características y la variable objetivo
    X_clas = df_clas.drop('anomaly', axis=1).values
    Y_clas = df_clas['anomaly'].values
    
    return X_clas, Y_clas

def accuracy(filename):
    X_clas, Y_clas = load_classification_problem(filename)
    
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    # Crear un pipeline que escala los datos y luego aplica LogisticRegression con OneVsRestClassifier
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=10000)))
    ])
    
    scoring = 'accuracy'
    
    # Evaluar el modelo usando validación cruzada
    results = cross_val_score(pipeline, X_clas, Y_clas, cv=kfold, scoring=scoring)
    
    print(f"Porcentaje de Exactitud / Desviación estándar: {results.mean()*100.0:.3f}% ({results.std()*100.0:.3f}%)")

def kappa_cohen(filename):
    test_size = 0.33
    seed = 7
    X_clas, Y_clas = load_classification_problem(filename)
    X_train, X_test, Y_train, Y_test = train_test_split(X_clas, Y_clas, test_size=test_size, random_state=seed)
    
    # Crear y entrenar el modelo
    model = OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=10000))
    model.fit(X_train, Y_train)
    
    # Hacer predicciones
    predicted = model.predict(X_test)
    
    # Calcular el porcentaje de acierto
    accuracy = accuracy_score(Y_test, predicted)
    
    # Calcular el Puntaje de Cohen (Kappa)
    cohen_kappa = cohen_kappa_score(Y_test, predicted)
    
    # Imprimir resultados
    print(f"Porcentaje de Acierto: {accuracy*100.0:.3f}%")
    print(f"Puntaje de Cohen: {cohen_kappa*100.0:.3f}%")

def confusion_matrix_example(filename):
    test_size = 0.33
    seed = 7
    X_clas, Y_clas = load_classification_problem(filename)
    X_train, X_test, Y_train, Y_test = train_test_split(X_clas, Y_clas, test_size=test_size, random_state=seed)
    
    # Crear y entrenar el modelo
    model = OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=10000))
    model.fit(X_train, Y_train)
    
    # Hacer predicciones
    predicted = model.predict(X_test)
    
    # Generar matriz de confusión
    matrix = confusion_matrix(Y_test, predicted)
    print("Matriz de Confusión:")
    print(matrix)
    
    # Visualizar la matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    disp.plot(cmap='Blues')

def classification_report_example(filename):
    test_size = 0.33
    seed = 7
    X_clas, Y_clas = load_classification_problem(filename)
    X_train, X_test, Y_train, Y_test = train_test_split(X_clas, Y_clas, test_size=test_size, random_state=seed)
    
    # Crear y entrenar el modelo usando OneVsRestClassifier
    model = OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=10000))
    model.fit(X_train, Y_train)
    
    # Hacer predicciones
    predicted = model.predict(X_test)
    
    # Generar el reporte de clasificación
    report = classification_report(Y_test, predicted, target_names=list(map(str, model.classes_)))
    print("Reporte de Clasificación:")
    print(report)
    
    # Generar la matriz de confusión
    matrix = confusion_matrix(Y_test, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    disp.plot(cmap='Blues')

# Llamar a las funciones con el archivo adecuado
filename = 'dataset/metaverse_transactions_dataset.csv'

print("Evaluación de Precisión:")
accuracy(filename)

print("\nKappa de Cohen:")
kappa_cohen(filename)

print("\nMatriz de Confusión:")
confusion_matrix_example(filename)

print("\nReporte de Clasificación:")
classification_report_example(filename)
