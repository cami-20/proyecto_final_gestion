import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

def cargar_data():
    # Cargar el dataset
    filename = 'dataset/metaverse_transactions_dataset.csv'
    df = pd.read_csv(filename)

    # Seleccionar las características y el objetivo
    X_reg = df[['hour_of_day', 'amount', 'ip_prefix', 'login_frequency', 'session_duration']]
    Y_reg = df['risk_score']
    
    return X_reg, Y_reg

# Error medio absoluto
def error_absoluto():
    test_size = 0.4
    seed = 8
    X_reg, Y_reg = cargar_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=test_size, random_state=seed)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # MAE: Error medio Absoluto
    MAE = mean_absolute_error(Y_test, predicted)
    print("Error Medio Absoluto: {}". format(MAE))

# Error Cuadrático Medio 
def error_cuadratico_medio():
    test_size = 0.4
    seed = 8
    X_reg, Y_reg = cargar_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=test_size, random_state=seed)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # MSE: Error Cuadrático Medio
    MSE = mean_squared_error(Y_test, predicted)
    print("Error Cuadrático Medio: {}". format(MSE))

# Coeficiente de Determinación conocido como R2
def coeficiente_determinacion():
    test_size = 0.4
    seed = 8
    X_reg, Y_reg = cargar_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=test_size, random_state=seed)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # Calculo del coeficiente de determinación
    R2 = r2_score(Y_test, predicted)
    print("Coeficiente de Determinación (R2): {}". format(R2))
    # valor 0 no tiene ajuste (modelo es ineficiente)
    # valor 1 tiene un ajuste perfecto

#Ejecución de las funciones
error_absoluto()
error_cuadratico_medio()
coeficiente_determinacion()
    