import os
import re

from pathlib import Path
from helpers import *


def parse_file_name(file_name):
    try:
        gender = file_name[0]  # F o M
        id_num = file_name[1:5]  # '0001'
        person_id = f"{gender}{id_num}"  # 'F0001'
        
        emotion = file_name[6:8]  # 'DI'
        intensity = file_name[8:10]  # '04'
        race = file_name[10:12]

        is_neutral = emotion == "NE"

        img_type = file_name.split("_")[-1].split(".")[0]  # 'F2D'

        return {
            "file_name": file_name,
            "person_id": person_id,
            "gender": gender,
            "race": race,
            "id_num": id_num,
            "emotion": emotion,
            "intensity": intensity,
            "is_neutral": is_neutral,
            "img_type": img_type,
        }
    except Exception as e:
        print(f"Error al parsear {file_name}: {e}")
        return None
    

def construir_dataframe_imagenes(ruta_base):
    ruta_base = Path(ruta_base)
    data = []

    for carpeta_persona in ruta_base.iterdir():
        if carpeta_persona.is_dir():
            for imagen_path in carpeta_persona.glob("*.bmp"):
                nombre = imagen_path.name
                parsed = parse_file_name(nombre)
                if parsed and parsed["img_type"] == "F2D":  # Filtramos solo imágenes F2D
                    parsed["original_path"] = str(imagen_path)
                    data.append(parsed)

    df = pd.DataFrame(data)
    return df



############ Legacy

def create_image_dict(base_dir, verbose=True):
    if not os.path.exists(base_dir):
        optional_print("La ruta no existe:" + base_dir, verbose)
    else:
        optional_print("Ruta encontrada:" + base_dir, verbose)

    # Regex pattern to match the filenames
    pattern = re.compile(r'^(?P<gender>[MF])(?P<id>\d{4})_(?P<exp>NE|AN|DI|FE|HA|SA|SU)(?P<exp_level>00|01|02|03|04)(?P<race>WH|BL|IN|AE|AM|LA)_(?P<attribute>F2D)\.(?P<ext>bmp)$')
    # Dictionary to hold the image data
    images_data = []

    # Walk through the base directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            match = pattern.match(file)
            if match and match.group('ext') == 'bmp':
                image_data = match.groupdict()
                image_data['name'] = os.path.splitext(file)[0]
                image_data['raw_image_folder'] = root
                #image_data['file'] = file
                image_data['idUnique'] = image_data['id'] + image_data['gender']
                images_data.append(image_data)
    return images_data
