import os
import re
from helpers import *

def create_image_dir(base_dir, verbosity=True):
    if not os.path.exists(base_dir):
        optional_print("La ruta no existe:" + base_dir, verbosity)
    else:
        optional_print("Ruta encontrada:" + base_dir, verbosity)

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
                image_data['file'] = file
                image_data['projected_npz'] = "N"
                image_data['projected_file'] = "N"
                image_data['idUnique'] = image_data['id'] + image_data['gender']
                images_data.append(image_data)
    return images_data