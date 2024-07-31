#Importar las librerias 
import numpy as np
import pandas as pd 
import seaborn as sns  
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix   


#Cargar la base 
filename = 'dataset/metaverse_transactions_dataset.csv'
data = pd.read_csv(filename)
#print("\nLectura de datos") 
#print(data)

# Función para calcular el sesgo en columnas numéricas
def sesgo(data):
    numeric_data = data.select_dtypes(include=['number'])
    result = numeric_data.skew()
    print(result)


print("\nCálculo del Sesgo en Columnas Numéricas de un DataFrame")
sesgo(data)
print("\nhour_of_day: -0.005089: El sesgo es muy cercano a cero, lo que sugiere que la distribución de esta columna es casi simétrica.")
print("amount: 0.124223: Un sesgo positivo, aunque pequeño, lo que indica que la distribución está ligeramente sesgada hacia la derecha.")
print("ip_prefix: -1.438522: Un sesgo negativo significativo, sugiriendo que la distribución está sesgada hacia la izquierda")
print("login_frequency: 0.174246: Un sesgo positivo leve, indicando que la distribución está ligeramente sesgada hacia la derecha.")
print("session_duration: 0.660789: Un sesgo positivo notable, sugiriendo que la distribución está sesgada hacia la derecha.")
print("risk_score: 1.047827: Un sesgo positivo significativo, lo que indica una distribución muy sesgada hacia la derecha.")


#Ejercicio de correlación
def matriz_correlacion(data):
    # Seleccionar solo las columnas numéricas
    numeric_data = data.select_dtypes(include=['number'])
    # Calcular la matriz de correlación
    correlacion = numeric_data.corr(method='pearson')
    #Mostrar la matriz de correlación en la consola
    print("Matriz de Correlación: ")
    print(correlacion)
    print("La correlación más fuerte se observa entre login_frequency y session_duration (0.871915). Esto sugiere que los usuarios que inician sesión con más frecuencia tienden a tener sesiones más largas.")
    print("Existe una correlación moderada negativa entre hour_of_day y risk_score (-0.190985). Esto podría indicar que el riesgo tiende a ser ligeramente menor en horas más avanzadas del día.")
    print("La mayoría de las otras variables muestran correlaciones muy débiles entre sí (valores cercanos a 0), lo que indica que no hay una relación lineal fuerte entre ellas")
    print("La fuerte correlación entre login_frequency y session_duration podría ser útil para entender el comportamiento del usuario, aunque no parece estar fuertemente relacionada con el riesgo.")


matriz_correlacion(data)


def grafica_correlacion(data):
     # Seleccionar solo las columnas numéricas
    numeric_data = data.select_dtypes(include=['number'])
    # Calcular la matriz de correlación
    correlacion = numeric_data.corr(method='pearson')

    #Visualizar la matriz de correlación con un mapa de color 
    plt.figure(figsize=(14, 14))
    plt.title('Matriz de Correlación')
    sns.heatmap(correlacion, vmax=1, square=True, annot=True, cmap='viridis')
    plt.show()

#grafica_correlacion(data)

#Distribución entre clases 
def distribution_in_classes():
    class_counts = data.groupby('anomaly').size()
    print(class_counts)
print
distribution_in_classes()


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


#correlation(data)

