import os
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
import subprocess


gc.collect()
torch.cuda.empty_cache()
verbose = True 

# Define the base directory
base_dir = "/mnt/discoAmpliado/viky/BU_3DFE"

images_data = create_image_dict(base_dir, verbose)
df = pd.DataFrame(images_data)
##df = pd.DataFrame.from_dict(images_data, orient='index')
df.to_csv("dataframe.csv", index=False)

##print("_____________[ALINEANDO]_____________")

##align_all_images(images_data, verbose) # Es necesario que tome los valores de images_data? No podría hacer "todas"?

##print("_____________[ALINEADO]_____________")
##print("_____________[PROCESANDO]_____________")

##process_all_images(1000, verbose)

##print("_____________[PROCESADO]_____________")
##print("_____________[GENERANDO]_____________")

##generate_all_images(verbose)

##print("_____________[GENERADO]_____________")
##print("_____________[FINALIZADO]_____________")


##print(f"_____________[ALINEADAS CHECK] {check_alineadas(images_data, False)}_____________")
##print(f"_____________[NPZ CHECK] {check_npz(images_data, False)[0]}_____________")

generate_one_image_from_npz(f"images/processed_images/AA_prueba_ne.np   z", "AA_prueba_ne")
generate_one_image_from_npz(f"images/processed_images/AA_prueba_sa.npz", "AA_prueba_sa")
#generate_one_image_from_npz(f"images/processed_images/AA_prueba_ha.npz", "AA_prueba_ha")
generate_one_image_from_npz(f"images/processed_images/AA_prueba_di.npz", "AA_prueba_di")




# Puedo obligarlo a funcionar loopeando esto?
#df_reproceso = df[df['name'].isin(check_npz(images_data, False)[1])]
#process_all_images_in_df(df_reproceso, 1000, True)






#### Para trabajar con una sola imagen
#process_one_image("F0030_AN01BL_F2D", 1000, verbose)
#generate_one_image("F0030_AN01BL_F2D", verbose)
