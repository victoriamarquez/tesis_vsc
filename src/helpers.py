import subprocess
from numpy import load, savez
from IPython.display import Image
from IPython import display
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

# Funciones Auxiliares

# toma un dataframe y un índice y te da el npz correspondiente
def getNPZ(df, index):
    # filename = df['projected_file'][index]
    # TODO: volver a load(filename)['w']
    fileName = df['name'][index]
    filenameFinal = f'/mnt/discoAmpliado/viky/images/results_BU_3DFE/{fileName}/projected_w.npz'
    return load(filenameFinal)['w']

def optional_print(text, verbosity=True):
    if verbosity:
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