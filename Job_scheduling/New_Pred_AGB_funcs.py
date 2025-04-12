skip = True
import os
import numpy as np, pandas as pd, geopandas as gpd
import rasterio
import time
import gzip
import shutil
from rasterstats import zonal_stats
from xgboost import  XGBRegressor
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import random
from cubist import Cubist   
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import concurrent.futures

def calculate_VIs_raster(image):
    eps = 0.00001
    ##### NDVI #####
    ndvi = (image[3,:,:] - image[2,:,:])/(image[3,:,:] + image[2,:,:]+ eps)
    nbr = (image[3,:,:] - image[5,:,:])/(image[3,:,:] + image[5,:,:]+ eps)
    # gndvi = (image[3,:,:] - image[1,:,:])/(image[3,:,:] + image[1,:,:]+ eps)
    # ndwi = (image[1,:,:] - image[3,:,:])/(image[1,:,:] + image[3,:,:]+ eps)
    msi = image[4,:,:]/(image[3,:,:]+ eps)
    sr = image[3,:,:]/(image[2,:,:]+ eps)
    # evi = 2.5*(image[3,:,:] - image[2,:,:]) / (image[3,:,:] + (6 * image[2,:,:]) - (7.5 * image[0,:,:]) + 1)
    nbr2 = (image[4,:,:] - image[5,:,:])/(image[4,:,:] + image[5,:,:]+ eps)
    #ndmi = (image[3,:,:] - image[4,:,:])/(image[3,:,:] + image[4,:,:])
    ###### Tasseled Cap #####
    coefficients = {
          'brightness': [0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303],
          'greenness': [-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446],
          'wetness': [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]
       }
    # Applying the coefficients to the bands and summing them
    brightness = np.sum([image[i,:,:] * coefficients['brightness'][i] for i in range(6)], axis=0)
    greenness = np.sum([image[i,:,:] * coefficients['greenness'][i] for i in range(6)], axis=0)
    wetness = np.sum([image[i,:,:] * coefficients['wetness'][i] for i in range(6)], axis=0)
    image = np.concatenate((image,  ndvi.reshape(1, ndvi.shape[0], ndvi.shape[1]), 
                      msi.reshape(1, ndvi.shape[0], ndvi.shape[1]),  sr.reshape(1, ndvi.shape[0], ndvi.shape[1]), 
                      nbr2.reshape(1, ndvi.shape[0], ndvi.shape[1]),  nbr.reshape(1, ndvi.shape[0], ndvi.shape[1]),
                      brightness.reshape(1, ndvi.shape[0], ndvi.shape[1]), greenness.reshape(1, ndvi.shape[0], ndvi.shape[1])), axis=0)
    return image
    


def simulate_correlated_errors_vectorized(df_std_devs, corr_matrix):
    """
    Simulates correlated random errors for a DataFrame for a single simulation, using vectorization and a shared correlation matrix, with optimized scaling.

    Args:
        df_data: DataFrame containing the original data.
        df_std_devs: DataFrame containing the standard deviations for each observation and variable.
        corr_matrix: A NumPy array representing the shared correlation matrix.

    Returns:
        A 2D NumPy array containing the simulated errors for each observation and variable.
    """

    num_obs, num_vars = df_std_devs.shape

    # Generate uncorrelated standard normal random variables for all variables
    uncorrelated_errors = np.random.randn(num_obs, num_vars)

    # Cholesky decomposition of the shared correlation matrix
    L = np.linalg.cholesky(corr_matrix)

    # Correlate the uncorrelated errors for all observations
    correlated_errors = np.dot(uncorrelated_errors, L.T)

    # Scale the correlated errors using broadcasting
    simulated_errors = correlated_errors * df_std_devs.values

    return simulated_errors
        

def load_and_process_raster(file_path, name_suffix, anci_feas, spring_feas, early_summer_feas, mid_summer_feas, late_summer_feas, fall_feas):
    with rasterio.open(file_path) as src:
        print(name_suffix)
        if not name_suffix == '_ancillary':
            data = src.read() / 10000
            data = calculate_VIs_raster(data)
            feature_names = list(src.descriptions)
            feature_names_n =[ 'NDVI', 'MSI', 'SR', 'NBR2', 'NBR', 'brightness', 'greenness']
            feature_names = feature_names + feature_names_n
            if not name_suffix == '_mid_summer':
                feature_names = [nm + name_suffix for nm in feature_names]
                
            print(feature_names)
            if name_suffix == '_spring':
                keep_indexes = [feature_names.index(x) for x in spring_feas]
                feas_nms = [feature_names[x] for x in keep_indexes]
            elif name_suffix == '_early_summer':
                keep_indexes = [feature_names.index(x) for x in early_summer_feas]      
                feas_nms = [feature_names[x] for x in keep_indexes]
            elif name_suffix == '_late_summer':
                keep_indexes = [feature_names.index(x) for x in late_summer_feas]    
                feas_nms = [feature_names[x] for x in keep_indexes]
            elif name_suffix == '_fall':
                keep_indexes = [feature_names.index(x) for x in fall_feas]  
                feas_nms = [feature_names[x] for x in keep_indexes]
            else:
                keep_indexes = [feature_names.index(x) for x in mid_summer_feas]       
                feas_nms = [feature_names[x] for x in keep_indexes]
        else:
            data = src.read() / 10000
            anci_nms = list(src.descriptions) 
            keep_indexes = [anci_nms.index(x) for x in anci_feas]
            feas_nms = [anci_nms[x] for x in keep_indexes]
            
        selected_data = data[keep_indexes,:,:]
        del data
        return selected_data, feas_nms
    
# Function to load model and make predictions
def load_and_predict(eco_model_index, data_2d):
    eco = eco_model_index.split('__')[0]
    model_index = eco_model_index.split('__')[1]
    model_path = f'/uufs/chpc.utah.edu/common/home/dycelab/data/Lidar/ABoVE/model_calibration/round2/Ground_data/trained_MLs/{eco}_Nccdc_uncert_{model_index}.json'
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model.predict(data_2d)*100
        
def pred_landsat(tile_year_nrep_model):
    tile = tile_year_nrep_model.split('_')[0]
    year = tile_year_nrep_model.split('_')[1]
    n_reps = int(tile_year_nrep_model.split('_')[2])
    model = tile_year_nrep_model.split('_')[3]
    eco = tile_year_nrep_model.split('_')[4]
    out_name = f'{eco}_AGB_Pred_Year_{year}_{tile}_{model}.tif'
    
    out_path = '/uufs/chpc.utah.edu/common/home/dycelab/data/Lidar/ABoVE/model_calibration/Landsat_pred/'
    output_path = os.path.join(out_path, out_name)

    def load_feas(eco, model):
        model_path = f'/uufs/chpc.utah.edu/common/home/dycelab/data/Lidar/ABoVE/model_calibration/round2/Ground_data/trained_MLs/{eco}_Nccdc_uncert_0{model}.json'
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return list(model.get_booster().feature_names)
        
    top_feature_names = load_feas(eco, model)
    anciA = ['TEMPERATURE', 'maxTEMPERATURE', 'minTEMPERATURE', 'RAINFALL', 'maxRAINFALL', 'minRAINFALL', 'maxTemp', 'minTemp', 'precip', 'waterDef', 'DEM_GLO30', 'slope_GLO30', 'aspect_GLO30', 'tsri_GLO30', 'tpi_GLO30']
    anci_feas = [nm for nm in top_feature_names if nm in anciA]
    spring_feas = [nm for nm in top_feature_names if '_spring' in nm]
    early_summer_feas = [nm for nm in top_feature_names if '_early_summer' in nm]
    late_summer_feas = [nm for nm in top_feature_names if '_late_summer' in nm]
    fall_feas = [nm for nm in top_feature_names if '_fall' in nm]
    mid_summer_feas = list(set(top_feature_names) - set(anci_feas + spring_feas + early_summer_feas + late_summer_feas+fall_feas))
    
    mid_summer = f'/uufs/chpc.utah.edu/common/home/dycelab2/data/ABoVE/Seasonal_bands/mid_summer_bands/Summer_Year_{year}_Bands_{tile}_Merged.tif'
    seasons = {
        "ancillary": f'/uufs/chpc.utah.edu/common/home/dycelab2/data/ABoVE/Ancillary/Ancillary_{tile}_Merged.tif',
        "spring": f'/uufs/chpc.utah.edu/common/home/dycelab2/data/ABoVE/Seasonal_bands/spring_bands/Spring_Year_{year}_Bands_{tile}_Merged.tif',
        "early_summer": f'/uufs/chpc.utah.edu/common/home/dycelab2/data/ABoVE/Seasonal_bands/early_summer_bands/Early_Summer_Year_{year}_Bands_{tile}_Merged.tif',
        "mid_summer": f'/uufs/chpc.utah.edu/common/home/dycelab2/data/ABoVE/Seasonal_bands/mid_summer_bands/Summer_Year_{year}_Bands_{tile}_Merged.tif',
        "late_summer": f'/uufs/chpc.utah.edu/common/home/dycelab2/data/ABoVE/Seasonal_bands/late_summer_bands/Late_Summer_Year_{year}_Bands_{tile}_Merged.tif',
        "fall": f'/uufs/chpc.utah.edu/common/home/dycelab2/data/ABoVE/Seasonal_bands/fall_bands/Fall_Year_{year}_Bands_{tile}_Merged.tif'
    }
    
    lc = f'/uufs/chpc.utah.edu/common/home/dycelab2/data/ABoVE/LC/Year_{year}_{tile}_Merged.tif'
    

    if os.path.exists(output_path)& skip:
        print('skipping')
    else:
        print("Reading and processing ancillary and seasonal rasters")        
        with ThreadPoolExecutor(6) as executor:
            future_results = {season: executor.submit(load_and_process_raster,  file_path,  f"_{season}", anci_feas, spring_feas , early_summer_feas, mid_summer_feas, late_summer_feas, fall_feas)
                              for season, file_path in seasons.items()}
            seasonal_data = {season: future.result() for season, future in future_results.items()}

        seasonal_arrays = [data for data, _ in seasonal_data.values()]    
        seasonal_names = [name for _, name in seasonal_data.values()]
        seasonal_names = [name for sublist in seasonal_names for name in sublist]

        indices = [seasonal_names.index(item) for item in top_feature_names]

        geo_array = np.concatenate(seasonal_arrays, axis=0)
        del seasonal_arrays
        geo_array = geo_array[indices,:,:]
                    
        #print(len(top_feature_names))
        # Reshape for model input
        height, width = geo_array.shape[1], geo_array.shape[2]
        data_2d = geo_array.transpose(1, 2, 0).reshape(-1, len(top_feature_names))
        del geo_array
        # Predict using pre-trained models
        all_predictions = np.zeros((n_reps, data_2d.shape[0]), dtype=np.int32)
        
        
        # Use ThreadPoolExecutor to parallelize the prediction for each model
        model_indices = [eco + '__' + str(i) + model for i in range(n_reps)]
        with ThreadPoolExecutor(max_workers= 10) as executor:
            futures = {executor.submit(load_and_predict, j, data_2d): j for j in model_indices}
            for future in as_completed(futures):
                j = futures[future]
                j = int(j.split('__')[1][:-3])
                try:
                    all_predictions[j, :] = future.result()
                except Exception as exc:
                    print(f"Model {j} generated an exception: {exc}")
        
        # Calculate outputs and uncertainties
        predicted_rasters = np.mean(all_predictions, axis=0).reshape(height, width)
        predicted_std = np.std(all_predictions, axis=0).reshape(height, width)
        #percent_uncertainty = (predicted_std / np.maximum(predicted_rasters, 1e-6)) * 100
        out_all = np.stack((predicted_rasters, predicted_std), axis=0)
        #out_all = out_all*100

        print('post processing')
        if os.path.exists(lc)&(eco!='TUNDRA'):
            lc_src = rasterio.open(lc)
            lc_ra = lc_src.read(1)
            nodata_value_raster = lc_src.nodata
            nodata_locations = np.where(lc_ra == nodata_value_raster, True, False)
            out_all[:,(lc_ra == 1) | (lc_ra == 7)| (lc_ra == 9)] = 0
            out_all[:,lc_ra==0] = -999
        out_all[(out_all < 0) & (out_all > -990)] = 0

        # Write output raster
        print("Writing output raster")
        meta = rasterio.open(mid_summer).meta
        meta.update(count=out_all.shape[0], nodata=-999)
        meta.update({
          "driver": "COG",
          "dtype": "int32",
          'compress': 'lzw',
          'blockxsize': 512,
          'blockysize': 512,
          "tiled": True,
          "interleave": "band"})
        output_path = os.path.join(out_path, out_name)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(out_all)
            dst.descriptions = ['pred_AGB', 'pred_std']
        print("Done!")
