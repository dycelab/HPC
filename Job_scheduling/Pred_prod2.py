#!/usr/bin/env python
# coding: utf-8

# In[6]:


import argparse
import os
import random
import numpy as np
import time
import geopandas as gpd

os.chdir('/uufs/chpc.utah.edu/common/home/dycelab/data/Lidar/ABoVE/data_code')
from New_Pred_AGB_funcs import *  # this is a python file with all the functions to derive AGB maps from predictors


# In[ ]:


# Argument parser for SLURM job submission
parser = argparse.ArgumentParser(description="Run AGB prediction with configurable years and ecos.")
parser.add_argument("--years", type=str, required=True, help="Comma-separated list of years (e.g., 2020,2021,2022)")
parser.add_argument("--ecos", type=str, required=True, help="Comma-separated list of ecoregions (e.g., TUNDRA,TAIGA)")
parser.add_argument("--process_tile", type=int, required=True, help="Number of parallel processes")
parser.add_argument("--nrep", type=int, required=True, help="Number of repetitions")
parser.add_argument("--model", type=str, required=True, help="Model version")

args = parser.parse_args()

# Convert arguments to proper types
years = [int(y) for y in args.years.split(",")]  # Convert to a list of integers
ecos = args.ecos.split(",")  # Convert to a list of strings
process_tile = args.process_tile  # Number of parallel processes
nrep = args.nrep  # Number of repetitions
model = args.model

print("Years:", years)
print("Ecoregions:", ecos)
print("Process Tile:", process_tile)
print("Number of Repetitions:", nrep)
print('Model: ', model)


# In[8]:


# Set working directories
os.chdir('/uufs/chpc.utah.edu/common/home/dycelab/data/Lidar/ABoVE/model_calibration/round2/Ground_data/')
aois = gpd.read_file('/uufs/chpc.utah.edu/common/home/dycelab/data/Lidar/ABoVE/model_calibration/AOI_ABoVE_gridB_Ecoregion.shp')
aois['NA_L1NAME'][aois['NA_L1NAME'].isin(['ARCTIC TUNDRA','TUNDRA','ARCTIC CORDILLERA'])] = 'TUNDRA'
aois['NA_L1NAME'][aois['NA_L1NAME'].isin(['NORTH AMERICAN DESERTS'])] = 'GREAT PLAINS'
aois['NA_L1NAME'][aois['NA_L1NAME'].isin(['MARINE WEST COAST FOREST'])] = 'NORTHWESTERN FORESTED MOUNTAINS'
aois['NA_L1NAME'][aois['NA_L1NAME'].isin(['EASTERN TEMPERATE FORESTS'])] = 'NORTHERN FORESTS'
aois['Ecoregion'] = aois['NA_L1NAME']
aois = aois[['grid_level', 'grid_id','geometry', 'NA_L1NAME', 'Ecoregion']].drop_duplicates()

ecos_dict = {
    'NF': 'NORTHERN FORESTS',
    'NWFM':  'NORTHWESTERN FORESTED MOUNTAINS',
    'TG': 'TAIGA',
    'AT': 'TUNDRA',
    'GP': 'GREAT PLAINS',
    'HP': 'HUDSON PLAIN',
    'All': ['NORTHERN FORESTS', 'NORTHWESTERN FORESTED MOUNTAINS','TAIGA','GREAT PLAINS','HUDSON PLAIN']
}
ecos = [ecos_dict[eco] for eco in ecos]


skipping = True
use_lc = False   
bands = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]  

# process_tile = 1
# nrep = 5
# model = 'F17'
# years = [2022]
# ecos = ['TUNDRA','NORTHWESTERN FORESTED MOUNTAINS', 'GREAT PLAINS','NORTHERN FORESTS','TAIGA','HUDSON PLAIN']
# ecos = ['GREAT PLAINS']

for eco in ecos:
    aoi_sb = aois.loc[aois['Ecoregion'] == eco]
    tiles = list(aoi_sb['grid_id'].unique())
    #tiles = ['Bh020v002', 'Bh009v006']
    print(f"{eco}: {len(tiles)} tiles")

    # random.seed(6)
    tiles = random.sample(tiles, len(tiles))
    #tiles = ['Bh024v019','Bh030v017']

    tiles_years = [tile + "_" + str(year) + '_' + str(nrep) + '_' + model + '_' + eco for tile in tiles for year in years]
    print(f"{eco}: {len(tiles_years)} tile-year combinations")

    st = time.time()
    
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(min(process_tile, len(tiles_years))) as executor:
        executor.map(pred_landsat, tiles_years)

    print(f'Time used: {(time.time() - st) / 60:.2f} minutes')


# In[ ]:




