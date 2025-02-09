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
verbose = True 

# Define the base directory
base_dir = "/mnt/discoAmpliado/viky/BU_3DFE"

images_data = create_image_dict(base_dir, verbose)
df = pd.DataFrame(images_data)
df.to_csv("imagenes.csv", index=False)
print(df.head())
#print(getNPZ("F0030_AN01BL_F2D"))

align_all_images(images_data[0:1], verbose) # Es necesario que tome los valores de images_data? No podría hacer "todas"?

print("_________ALINEADO_____________")
#process_all_images(1000, verbose)

#generate_all_images(verbose)





#### Para trabajar con una sola imagen
#process_one_image("F0030_AN01BL_F2D", 1000, verbose)
#generate_one_image("F0030_AN01BL_F2D", verbose)