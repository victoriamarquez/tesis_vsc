
import gc

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch

from helpers import check_alineadas, check_npz, load_dataframe, optional_print, restar_vectores_neutros_y_normalizar
from image_pre_processing import create_image_dict
from image_processing import align_all_images, generate_all_images, process_all_images

def metodo_2(df):
    """Método 2: Promedio y luego regresión lineal, por emoción."""
    results = {}
    emotions = ['DI', 'HA', 'SU', 'AN', 'SA', 'FE']
    
    for emotion in emotions:
        subset_emocion = df[df['exp'] == emotion]
        promedios_por_persona = [] ## no promediar
        for idUnique in subset_emocion['idUnique'].unique():
            subset_emocion_persona = subset_emocion[subset_emocion['idUnique'] == idUnique]
            ## print(subset_emocion_persona)
            promedios_por_persona.append(np.mean(subset_emocion_persona['adjusted_vector'], axis=0))
        if (len(promedios_por_persona) != 100):
            print(f'[ADVERTENCIA] El tamaño del vector de promedios por persona debería ser 100 y es {len(promedios_por_persona)}.')

        
        X = np.arange(len(promedios_por_persona)).reshape(-1, 1) 
        y = np.array(promedios_por_persona).reshape(np.array(promedios_por_persona).shape[0], -1)  # Aplanamos los vectores antes de la regresión

        #print(f'El vector X tiene la forma {X.shape} y el vector y tiene la forma {y.shape}.')
        #print(f'Para la emoción {emotion} el vector X es {X} y el vector y es {y}.')

        
        model = LinearRegression().fit(X, y)
        results[emotion] = model.coef_.reshape(1, 18, 512)  # Restauramos la forma original
            
        ## results[emotion] = promedio_de_promedios_emocion
    return results


def main(align=False, process=False, generate=False, verbose=True):
    optional_print("_____________[INICIANDO EJECUCIÓN]_____________", verbose)

    gc.collect()
    torch.cuda.empty_cache()

    # Define the base directory
    base_dir = "/mnt/discoAmpliado/viky/BU_3DFE"

    images_data = create_image_dict(base_dir, verbose)
    
    if(align==True):
        optional_print("_____________[ALINEANDO]_____________", verbose)
        align_all_images(images_data, verbose)
        optional_print("_____________[ALINEADO]_____________", verbose)
        df = pd.DataFrame(images_data)
        df = df.drop(columns=['race', 'attribute', 'ext'])
        df.to_csv("dataframe.csv", index=False)
    else:
        df = pd.read_csv(f'/mnt/discoAmpliado/viky/dataframe.csv')
        df = df.drop(columns=['race', 'attribute', 'ext'])
    
    if(process==True):
        optional_print("_____________[PROCESANDO]_____________", verbose)
        process_all_images(1000, verbose)
        optional_print("_____________[PROCESADO]_____________", verbose)
    if(generate==True):
        optional_print("_____________[GENERANDO]_____________", verbose)
        generate_all_images(verbose)
        optional_print("_____________[GENERADO]_____________", verbose)
    
    optional_print(f"_____________[ALINEADAS CHECK] {check_alineadas(images_data, False)}_____________", verbose)
    optional_print(f"_____________[NPZ CHECK] {check_npz(images_data, False)[0]}_____________", verbose)
    
    df['idUnique'] = df['id'].astype(str) + df['gender']

    # Crear lista de emociones y eliminar 'NE'
    emociones = df.exp.unique().tolist().remove("NE")  ## En este momento, esto  no se usa para nada.

    # Crear lista de IDs
    ids = df.idUnique.unique().tolist()  ## En este momento, esto  no se usa para nada.

    df_extreme = df[(df['exp_level']==4) | (df['exp']=='NE')]

    restar_vectores_neutros_y_normalizar(df_extreme)
    ## optional_print(df_extreme, verbose)

    metodo_2(df_extreme)

    optional_print("_____________[EJECUCIÓN FINALIZADA]_____________", verbose)

main(False, False, False, True)

