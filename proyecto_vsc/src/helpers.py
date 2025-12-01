import subprocess
from numpy import load, savez
from IPython.display import Image
from IPython import display
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import os

# Funciones Auxiliares

def save_modified_npz(base_w, delta, out_path):
    """
    Aplica un vector diferencial (delta) a un vector latente base y guarda el resultado 
    como un archivo NPZ compatible con StyleGAN2-ADA.

    Usada en method_comparison.

    La función asume que tanto `base_w` como `delta` son vectores latentes 'w' 
    en el espacio $W$ o $W+$ de shape `(512,)`. Los combina y luego los replica 
    18 veces a través de las capas para crear el formato esperado por StyleGAN2-ADA: 
    `(1, 18, 512)`.

    Args:
        base_w (numpy.ndarray): Vector latente neutro base (shape (512,)).
        delta (numpy.ndarray): Vector de diferencia o dirección emocional a aplicar (shape (512,)).
        out_path (str): Ruta completa al archivo de salida donde se guardará el nuevo .npz.

    Returns:
        None: Guarda el archivo .npz en la ruta especificada.
    """
    modified_w = np.repeat((base_w + delta)[np.newaxis, :], 18, axis=0)[np.newaxis, :, :]
    np.savez(out_path, w=modified_w)

def getNPZ(name):
    """
    Carga el vector latente 'w' proyectado de un archivo NPZ y lo reduce al formato (512,).

    Usada en varios lugares.
    
    La función construye la ruta de archivo NPZ asumiendo un formato de nombre 
    estándar (`{base_name}_01_projected_w.npz`) dentro de un directorio fijo. 
    Carga el vector, que típicamente tiene un shape `(1, 18, 512)` (espacio W+), 
    y extrae un único slice para obtener el vector latente base de 512 dimensiones.

    Args:
        name (str): Nombre del archivo original de la imagen (con o sin extensión), 
            usado para construir el nombre del archivo .npz proyectado.

    Returns:
        numpy.ndarray: El vector latente 'w' base, limpio de la dimensión de capas 
            repetidas (shape (512,)).
    """

    # Quitar extensión si está presente
    base_name = name.split('.')[0]
    filename = f'/home/vicky/Documents/tesis_vsc/images/processed_images/{base_name}_01_projected_w.npz'
    
    # Cargar vector
    w = load(filename)['w']  # (1, 18, 512)
    
    # Extraer solo un slice (por ejemplo el primero), y sacar la dimensión repetida
    w_clean = w[0, 0, :]  # shape: (512,)
    
    return w_clean

def optional_print(text, verbose=True):
    """
    Imprime un texto solo si el flag 'verbose' es True.
    
    Usada en varios lugares, tengo muchas ganas de deprecarla y usar los logs.

    Es una función de utilidad simple para controlar la verbosidad de las salidas 
    de la consola sin tener que repetir el condicional `if verbose:` en el código principal.

    Args:
        text (str): El texto a imprimir.
        verbose (bool, optional): Si es True (por defecto), el texto se imprime. 
            Si es False, no ocurre nada.
    """
    if verbose:
        print(text)
    else:
        ...

def getByUniqueIdEmotion(df, idUnique, emotion):
    """
    Filtra y ordena un DataFrame para obtener todos los registros de una persona 
    para una emoción específica, ordenados por nivel de intensidad.

    Usada en los métodos que calculan las direcciones para PCA y Linear Regression.

    Asume que el DataFrame de metadatos contiene las columnas 'exp' (código de emoción), 
    'idUnique' (identificador de persona) y 'exp_level' (nivel de intensidad).

    Args:
        df (pandas.DataFrame): DataFrame de metadatos de las imágenes.
        idUnique (str): Identificador único de la persona (ej. 'F0001').
        emotion (str): Código de la emoción a filtrar (ej. 'HA', 'AN').

    Returns:
        pandas.DataFrame: Un DataFrame que contiene solo las filas que coinciden 
            con la persona y la emoción, ordenadas ascendentemente por el nivel 
            de intensidad (`exp_level`).
    """
    # Todos los registros donde ID está EMOCION
    gender = idUnique[-1]
    id = idUnique[:-1]
    res = df.loc[(df['exp'] == emotion) & (df['idUnique'] == idUnique)]
    
    # Todos los registros donde ID está EMOCION, ordenados de menos EMOCION a más EMOCION
    res = res.sort_values(by=['exp_level'],ascending=True, inplace=False)
    res.reset_index(drop=True, inplace=True)
    return res
