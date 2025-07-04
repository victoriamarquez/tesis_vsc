import os
import re
import numpy as np
import pandas as pd
from IPython.display import Image 
from numpy import load, savez
import torch
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
from sklearn.preprocessing import normalize
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from image_pre_processing import *
from image_processing import *
from proyecto_vsc.src.legacy.pca import *
from proyecto_vsc.src.legacy.lr import *
from proyecto_vsc.src.legacy.bootstrapping import *

gc.collect()
torch.cuda.empty_cache()
verbose = True

# Define the base directory
base_dir = "/mnt/discoAmpliado/viky/BU_3DFE"

images_data = create_image_dict(base_dir, verbose)

align_all_images(images_data[0:2], verbose)
##df = batch_processing(images_data[0:2], verbosity)
df = load_dataframe()

## df = pd.read_csv(f'/mnt/discoAmpliado/viky/dataframes/processed_dataframe_combined_fallback.csv')
df['idUnique'] = df['id'].astype(str) + df['gender']

# Crear lista de emociones y eliminar 'NE'
emociones = df.exp.unique().tolist()
emociones.remove("NE")

# Crear lista de IDs
ids = df.idUnique.unique().tolist()

# Crear un DataFrame vacío con IDs como índice y emociones como columnas
emociones_total_LR = pd.DataFrame(index=ids, columns=emociones)
emociones_total_PCA = pd.DataFrame(index=ids, columns=emociones)

# Calcular el vector de cada emoción para cada persona
for emocion in emociones:
    for idUnique in ids:
        # LR
        direccion_emocion = linearRegression(df, idUnique, emocion, False)
        emociones_total_LR.at[idUnique, emocion] = direccion_emocion.flatten()
        
        # PCA
        ##direccion_emocion = executePCA(df, idUnique, emocion, False)
        ##emociones_total_PCA.at[idUnique, emocion] = direccion_emocion
        
optional_print('PCA and Linear Regression completed successfully.', verbose)

# Normalizar PCA
##emociones_total_PCA_normalizado = normalizar_emociones(emociones_total_PCA)

# Normalizar LR
##emociones_total_LR_normalizado = normalizar_emociones(emociones_total_LR)
##optional_print('PCA and Linear Regression results normalized successfully.', verbosity)

# Bootstrapping y similitud coseno (func void solo printea)

#bootstrapping_todas_emociones(emociones_total_PCA_normalizado, emociones_total_LR_normalizado, emociones, 1000, False)

##optimos_LR = umbrales_optimos(emociones_total_LR, 1000, verbosity)
##optimos_LR.to_csv('umbrales_optimos_LR.csv', sep='\t')

# TODO: Los outputs de umbrales_optimos son medio raros?
# Idea: debuggear umbrales_optimos para entender si tiene sentido lo que está haciendo
# Cómo sigo? Qué falta? (además de hacer andar el docker)

# TODO: SIGUIENTE PASO: Visualizar los resultados para cada emoción???