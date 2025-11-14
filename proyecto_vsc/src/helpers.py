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
    modified_w = np.repeat((base_w + delta)[np.newaxis, :], 18, axis=0)[np.newaxis, :, :]
    np.savez(out_path, w=modified_w)

# toma un dataframe y un índice y te da el npz correspondiente
def getNPZ_legacy(name):
    filename = f'/home/vicky/Documents/tesis_vsc/images/processed_images/{name}_01_projected_w.npz'
    return load(filename)['w']

def getNPZ(name):
    # Quitar extensión si está presente
    base_name = name.split('.')[0]
    filename = f'/home/vicky/Documents/tesis_vsc/images/processed_images/{base_name}_01_projected_w.npz'
    
    # Cargar vector
    w = load(filename)['w']  # (1, 18, 512)
    
    # Extraer solo un slice (por ejemplo el primero), y sacar la dimensión repetida
    w_clean = w[0, 0, :]  # shape: (512,)
    
    return w_clean

def optional_print(text, verbose=True):
    if verbose:
        print(text)
    else:
        ...

def combine_dataframes(total_batches, verbosity=True):
    all_dfs = []
    for batch_num in range(total_batches):
        try:
            df_batch = pd.read_csv(f'/mnt/discoAmpliado/viky/dataframes/processed_dataframe_batch_{batch_num + 1}.csv')
            all_dfs.append(df_batch)
        except FileNotFoundError:
            optional_print(f'Batch {batch_num + 1} CSV file not found, skipping.', verbosity)

    df_combined = pd.concat(all_dfs, ignore_index=True)
    #df_combined['idUnique'] = df_combined['id'].astype(str) + df_combined['gender']
    df_combined.to_csv('/mnt/discoAmpliado/viky/dataframes/processed_dataframe_combined.csv', index=False)
    return df_combined

def getByUniqueIdEmotion(df, idUnique, emotion):
    # Todos los registros donde ID está EMOCION
    gender = idUnique[-1]
    id = idUnique[:-1]
    res = df.loc[(df['exp'] == emotion) & (df['idUnique'] == idUnique)]
    
    # Todos los registros donde ID está EMOCION, ordenados de menos EMOCION a más EMOCION
    res = res.sort_values(by=['exp_level'],ascending=True, inplace=False)
    res.reset_index(drop=True, inplace=True)
    return res

# Normalizar emociones por separado
def normalizar_emociones(df_emociones):
    df_normalizado = df_emociones.copy()
    for emocion in df_emociones.columns:
        # Concatenar vectores, normalizar y volver a asignar
        vectores = np.stack(df_emociones[emocion].dropna())
        vectores_normalizados = normalize(vectores, norm='l2', axis=1)
        df_normalizado[emocion] = list(vectores_normalizados)
    return df_normalizado

def check_npz(images_data, verbose=False):
    npz_encontrados = []
    npz_no_encontrados = []
    for image in images_data:
        name = os.path.splitext(image['name'])[0]  
        if not os.path.exists(f'/home/vicky/Documents/tesis_vsc/images/processed_images/{name}_01_projected_w.npz'):
            npz_no_encontrados.append(name)
        else:
            npz_encontrados.append(name)
    optional_print(f"Npz no encontrados: {npz_no_encontrados}", verbose)
    optional_print("-------------", verbose)
    optional_print(f"Npz encontrados: {npz_encontrados}", verbose)
    if len(npz_no_encontrados) != 0:
        return "NO se encontraron todos los npzs.", npz_no_encontrados
    else:
        return "SI se encontraron todos los npzs.", npz_no_encontrados

def check_alineadas(images_data, verbose=False):
    alineadas_encontradas = []
    alineadas_no_encontradas = []
    for image in images_data:
        #name = os.path.splitext(image['name'])[0]  
        print(image)
        if not os.path.exists(f'/home/vicky/Documents/tesis_vsc/images/aligned_images/{name}_01.png'):
            alineadas_no_encontradas.append(name)
        else:
            alineadas_encontradas.append(name)
    optional_print(f"Imagenes alineadas no encontradas: {alineadas_no_encontradas}", verbose)
    optional_print("-------------", verbose)
    optional_print(f"Imagenes alineadas encontradas: {alineadas_encontradas}", verbose)
    if len(alineadas_no_encontradas) != 0:
        return "NO se encontraron todas las imagenes alineadas."
    else:
        return "SI se encontraron todas las imagenes alineadas."
    
def normalize_vectors(vectors):
    """Normaliza una lista de vectores usando norma L2.""" ## acá usar solo el 2ro de los 18 vectores
    original_shape = vectors.shape  # Guardamos la forma original

    ## vectors = vectors.reshape(vectors.shape[0], -1)  # Aplanamos a (N, D)
    vectors = normalize(vectors, axis=1)  # Normalizamos en la dimensión correcta
    ##return vectors.reshape(original_shape)  # Restauramos la forma original
    return vectors

def restar_vectores_neutros_y_normalizar(df):
    """Resta el vector neutro de cada persona a sus otras imágenes"""
    adjusted_vectors = []
    
    for person_id, group in df.groupby('idUnique'):
        neutral_vector = None
        
        # Buscar la imagen neutra
        for _, row in group.iterrows():
            if row['exp'] == 'NE':
                neutral_vector = getNPZ(row['name'])[0]
                break
        
        # Si no hay imagen neutra, dejamos los vectores como están
        if neutral_vector is None:
            print(f"Advertencia: No se encontró imagen neutra para {person_id}")
            adjusted_vectors.extend([getNPZ(row['name']) for _, row in group.iterrows()])
            continue
        
        # Restar el vector neutro de las otras imágenes
        for _, row in group.iterrows():
            vector = getNPZ(row['name'])[0][0]
            print("Vector shape", vector.shape)
            print("Vector: ", vector)
            adjusted_vectors.append(normalize(vector)-normalize(neutral_vector)) # Versión en la que normalizo primero
            ##adjusted_vectors.append(normalize_vectors(vector - neutral_vector)) #Switch para normalizar o no
            ##adjusted_vectors.append(vector - neutral_vector)
    
    df['adjusted_vector'] = adjusted_vectors
    return df


# LEGACY

# carga un npz desde un archivo
def cargar_npz(filename):
    return load(f'images/results/{filename}/projected_w.npz')['w']

# creando la imagen npz->bmp
def generateImage(outdir, npz_path, verbosity=True):
    optional_print(f"Now processing {npz_path}", verbosity)
    command = f"./stylegan2-ada-pytorch/docker_run.sh python ./stylegan2-ada-pytorch/generate.py --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --outdir={outdir} --projected-w {npz_path}"
    run_command(command, verbosity)
    optional_print(f'Image can be found in {outdir}.', verbosity)

# Guarda data en formato npz
def generateNPZ(path, data):
    savez(f'images/{path}.npz', w=data)

# toma un array de npz y genera todas las imágenes, borador
def generateImagesAndDisplay(arr):
    for i in range(0, len(arr)):
        generateNPZ(f'borrador/{i}', arr[i])
        generateImage('borrador', f'images/borrador/{i}.npz')
        display(Image(filename='images/borrador/proj00.png'))

def run_command(command, verbosity=True):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        optional_print("Salida estándar:", verbosity)
        optional_print(result.stdout, verbosity)
    except subprocess.CalledProcessError as e:
        optional_print("Error al ejecutar el comando:", verbosity)
        optional_print(e.stderr, verbosity)

def load_dataframe():
    df = pd.read_csv(f'/mnt/discoAmpliado/viky/dataframes/processed_dataframe_combined.csv')
    return df