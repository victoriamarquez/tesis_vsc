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
from pca import *
from lr import *
from bootstrapping import *
import subprocess


gc.collect()
torch.cuda.empty_cache()
verbosity = True

# Define the base directory
base_dir = "/mnt/discoAmpliado/viky/BU_3DFE"

images_data = create_image_dir(base_dir, verbosity)

# TODO: Hacer andar el docker que proyecta
align_images(images_data[0:2], verbosity) # Esto sí funciona
print("Align OK.")
##df = batch_processing(images_data[0:2], verbosity)

# Ruta al script que deseas ejecutar
script_path = "/mnt/discoAmpliado/viky/projectar_todo.py"

# Llamar al script projectar_todo.py
subprocess.run(['python', script_path], check=True)


##print(df)