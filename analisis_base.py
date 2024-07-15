#Importar las librerias 
import numpy as np
import pandas as pd 
import seaborn as sns  
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix   


#Cargar la base 
filename = 'dataset\metaverse_transactions_dataset.csv'
data = pd.read_csv(filename) 
#print(data)

# Función para calcular el sesgo en columnas numéricas
def sesgo(data):
    numeric_data = data.select_dtypes(include=['number'])
    result = numeric_data.skew()
    print(result)


#sesgo(data)

#Ejercicio de correlación
def matriz_correlacion():
    correlacion = data.corr(method='pearson')
    plt.figure(figsize=(14,14))
    plt.title('Matriz de Correlación')
    sns.heatmap(correlacion, vmax=1, square=True, annot=True, cmap='viridis')
    plt.show()    
#matriz_correlacion()

#Distribución entre clases 
def distribution_in_classes():
    class_counts = data.groupby('anomaly').size()
    print(class_counts)
    
#distribution_in_classes()


#Resumen general del dataset
def summary():
    print(data.describe())
    
#summary()

#Calcular la cantidad de datos por clase 
def count_data_by_class():   
    class_counts = data.groupby('anomaly').size()
    print(class_counts)

#count_data_by_class()

#Imprimir los tipos de datos
def print_Data_types():
    print(data.dtypes)
    
#print_Data_types()

#Imprimir la dimension del df
def print_dimensions():
    print(data.shape)
    
#print_dimensions()


# Función para calcular la correlación en columnas numéricas
def correlation(data):
    numeric_data = data.select_dtypes(include=['number'])
    correlations = numeric_data.corr(method='pearson')
    print(correlations)


correlation(data)

