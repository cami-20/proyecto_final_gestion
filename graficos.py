# Importar las librerias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from pandas.plotting import scatter_matrix 

# Cargar los datos
filename = 'dataset/metaverse_transactions_dataset.csv'
data = pd.read_csv(filename)

# Función matriz densidad
def matriz_densidad():
    
    # Código básico para gráficar
    figura = plt.figure(figsize=(6,6))
    
    # Código básico para gráficar una matriz de dispersión
    data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)

    #Agregar un título general a la figura
    figura.suptitle("MATRIZ DE DENSIDAD", fontsize=16)

    # Mostrar la figura
    plt.show()

#Llamar a la función 
#matriz_densidad()

#Función de diagrama de caja
def diagrana_caja():

    # Código básico para gráficar
    figura = plt.figure(figsize=(8,8))

    #Código para gráfica una matriz de un diagrama de caja
    data.plot(kind='box', subplots=True, layout=(3,3), sharex=False)

      #Agregar un título general a la figura
    figura.suptitle("DIAGRAMA DE CAJA", fontsize=16)
    
    # Mostrar la figura
    plt.show()

 #Llamar a la función 
#diagrana_caja()

#Función para generar la matriz de correlación 
def matriz_correlacion():
    datos_numericos = data.select_dtypes(include=[np.number])
    correlacion = datos_numericos.corr(method='pearson')
    plt.figure(figsize=(6,6))
    plt.title('Matriz de Correlación')
    sns.heatmap(correlacion, vmax=1, square=True, annot=True, cmap='viridis')
    plt.show()    

#Llamar a la función 
#matriz_correlacion()

#Función para matriz de dispersión 
def matriz_dispersion():
    plt.rcParams['figure.figsize'] = (15,15)
    scatter_matrix(data)
    plt.show()
matriz_dispersion()



