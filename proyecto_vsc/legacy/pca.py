import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from helpers import *

def executePCA(df, idUnique, emotion):
    logging.info(f'[executePCA] [→] Calculating PCA for person: {idUnique}, emotion: {emotion}.')
    
    # 1. Preparar los datos
    data = getByUniqueIdEmotion(df, idUnique, emotion)
    vectores_latentes = [getNPZ(data, index) for index in data.index] # Lista de vectores latentes, cada uno con forma (1, 18, 512)
    
    # Aplanar los vectores a una forma 2D (n_samples, 18*512)
    vectores_aplanados = np.array([v.reshape(18*512) for v in vectores_latentes])
    
    # Paso 2: Aplicar PCA
    pca = PCA(n_components=4)  # Solo necesitamos el primer componente
    pca.fit(vectores_aplanados)
    
    # Obtener la dirección de la emoción
    direccion_emocion = pca.components_[0]
    logging.info(f'[executePCA] [✔] Calculated PCA for person: {idUnique}, emotion: {emotion}.')

    return direccion_emocion