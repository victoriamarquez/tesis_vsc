from helpers import *
import numpy as np


from sklearn.linear_model import LinearRegression

def linearRegression(df, idUnique, emotion, verbose=True):

    # 1. Preparar los datos
    data = getByUniqueIdEmotion(df, idUnique, emotion)
    
    # Supongamos que tienes los siguientes datos:
    vectores_latentes = [getNPZ(data, index) for index in data.index] # Lista de vectores latentes, cada uno con forma (1, 18, 512)
    niveles_emocion = [1, 2, 3, 4]  # Lista de niveles de emocion asociados, cada uno es un escalar
    
    # Convertir los vectores latentes a una matriz de forma (n_samples, 18 * 512)
    X = np.array([vector.reshape(-1) for vector in vectores_latentes])  # Aplana cada vector latente a 1D
    y = np.array(niveles_emocion)  # Convertir los niveles de felicidad a un array numpy
    
    # 2. Ajustar el modelo de regresión lineal
    
    # Crear el modelo de regresión lineal
    modelo = LinearRegression()
    
    # Ajustar el modelo a los datos
    modelo.fit(X, y)
    
    # Obtener la dirección de emocion en el espacio latente
    direccion_emocion = modelo.coef_.reshape(1, 18, 512)

    optional_print(f'Executed Linear Regression for person: {idUnique}, emotion: {emotion}.', verbose)
    return direccion_emocion