# -*- coding: utf-8 -*-

"""
Created on Fri Mar 15 11:38:27 2024

@author: Frederike Arp
"""
"""
this script makes the regressions of all different defined models that are loaded from regression_models_classes.py
then calcualtes rmse and r2 values
based on that matches every regino with the accordingly best fitting model
and then projects data into the future based out of this regression
"""

import pandas as pd
import numpy as np

from imagematerials.rest_of.const import REGION_TO_CLASS_DICT, models_output_dict

from imagematerials.rest_of.regression_models_classes import (Log_Log_Model, Semi_Log_Model, 
                                       Log_Inverse_Model, Log_Log_Inverse_Model, 
                                       Log_Log_Square_Model, NLI_Model, 
                                       GOMPERTZ_Model, LG_Model, BW_Model, 
                                       Log_Gauss_Saturate_Model)

#%% Estimate models

def estimate_models(cons_capita: pd.DataFrame, gdp_pc: pd.DataFrame, bounds:tuple):
    """
    Calcualte regression for every mathematical model and every region(group).
    
    Parameters
    ----------
    cons_capita : pd.DataFrame
       consumption pc per groups of IMAGE regions.
    gdp_pc : pd.DataFrame
        gdp pc per groups of IMAGE regions.

    Returns
    -------
    log_log_model : regression model 
        DESCRIPTION.
    semi_log_model : regression model
        DESCRIPTION.
    log_inverse_model : regression model
        DESCRIPTION.
    log_log_inverse_model : regression model
        DESCRIPTION.
    log_log_square_model : regression model
        DESCRIPTION.
    non_linerar_inv_model : regression model
        DESCRIPTION.
    gompertz_model : regression model
        DESCRIPTION.
    logistic_growth_model : regression model
        DESCRIPTION.
    bw_model : regression model
        DESCRIPTION.

    """
    # estimate every model
    # log_log_model = Log_Log_Model(cons_capita, gdp_pc)
    # semi_log_model = Semi_Log_Model(cons_capita, gdp_pc)
    # log_inverse_model = Log_Inverse_Model(cons_capita, gdp_pc)
    # log_log_inverse_model = Log_Log_Inverse_Model(cons_capita, gdp_pc)
    # log_log_square_model = Log_Log_Square_Model(cons_capita, gdp_pc)
    non_linerar_inv_model = NLI_Model(cons_capita, gdp_pc, bounds = bounds)

    # try and except for these models, as they might have a runtime error and not produce results
    # if runtime errer: model is given None as output

    
    try:
        bw_model = BW_Model(cons_capita, gdp_pc, bounds = bounds)
    except RuntimeError as e:
        print('limited growth model', e)
        bw_model = None
        
    try:
        gompertz_model = GOMPERTZ_Model(cons_capita, gdp_pc, bounds = bounds)
    except RuntimeError as e:
        print('Gompertz model', e)
        gompertz_model = None   
        
    try:
        logistic_growth_model = LG_Model(cons_capita, gdp_pc, bounds = bounds)
    except RuntimeError as e:
        print('logistic growth model', e)
        logistic_growth_model = None

    try:
        log_gauss_saturate_model = Log_Gauss_Saturate_Model(cons_capita, gdp_pc, bounds = bounds)
    except RuntimeError as e:
        print('log-gauss-saturate model', e)
        log_gauss_saturate_model = None



    return (non_linerar_inv_model, gompertz_model, 
            logistic_growth_model, bw_model, log_gauss_saturate_model)

    

#%% Make statistical analysis

def rmse_r2_models(models_output: tuple) -> pd.DataFrame:
    """
    Take list of regression models for selection of IMAGE grouped regions (output of estimate_models()).
    
    Parameters
    ----------
    models_output : tuple
        DESCRIPTION.

    Returns
    -------
    rmse_r2 : TYPE
        Return a DF with all RMSE and R2 values for all models for the specific region.

    """
    # from tuple that is given as input create a dictionary that assigns the models to names
    models_output_dict = {
        # models_output[0]: 'log-log model',
        # models_output[1] : 'semi-log model',
        # models_output[2] : 'log-inverse model',
        # models_output[3] : 'log-log-inverse model',
        # models_output[4] : 'log-log-square model',
        models_output[0] : 'non-linear inverse model',
        models_output[1] : 'gompertz model',
        models_output[2] : 'logistic growth model',
        models_output[3] : 'limited growth model', # (beschraenktes Wachstum)
        models_output[4] : 'log gauss saturate model'
        }
    
    #loop over models in dict to calculate RMSE and R2
    # put in None if no model could be calculated
    rmse_r2 = {'Model': models_output_dict.values(),
            'RMSE': [model.rmse if model is not None else np.nan for model in models_output_dict],
            'R^2' : [model.r2 if model is not None else np.nan for model in models_output_dict]}
    
    # make output a pd.DF
    rmse_r2 = pd.DataFrame(rmse_r2)
    rmse_r2.set_index('Model', inplace=True)
    
    return rmse_r2


#%% Estimate models for groups and perform statistical analysos

def estimate_models_per_region_group(regions_groups_dict: dict, 
                                     cons_pc_groups: dict,
                                     gdp_pc_groups: dict,
                                     bounds: dict) -> tuple:
    """
    Estimate all regression models for all groups of regions. Builds on estimate_models and rmse_r2_models

    Parameters
    ----------
    regions_groups_dict : dict{"region_group": [USA, China],
                               "region group 2": [Europe]}
        collection (list) of similar image regions in terms of per capita consumption and gdp that are stored in dict.
    cons_pc_groups : dict{'region_group': pd.DataFrame}
        consumption per capia for similar regions stored in .
    gdp_pc_groups : dict{'region_group': pd.DataFrame}
        DESCRIPTION.

    Returns
    -------
    model_groups : TYPE
        DESCRIPTION.
    rmse_r2_groups : TYPE
        DESCRIPTION.
    merged_rmse_r2 : TYPE
        DESCRIPTION.
    """
    model_groups = {}
    rmse_r2_groups = {}

    for region_group in regions_groups_dict.keys():
        if isinstance(bounds, dict):
            group_bounds = bounds.get(region_group, bounds)
        else:
            group_bounds = bounds
        model_groups[region_group] = estimate_models(
            cons_pc_groups.get(region_group),
            gdp_pc_groups.get(region_group),
            group_bounds
        )
        rmse_r2_groups[region_group] = rmse_r2_models(model_groups[region_group])
        
    merged_rmse_r2 = pd.concat(rmse_r2_groups.values(), axis=1, keys=rmse_r2_groups.keys())
    
    return model_groups, rmse_r2_groups, merged_rmse_r2



def match_regions_to_best_model(rmse_r2_groups: dict, 
                                model_groups: dict,
                                models_output_dict: dict,
                                regions_groups_dict: dict[str, list[str]], 
                                best_rmse_models: dict = None):
    """
    Match regions to best fitting model based on RMSE values.
    Parameters
    ----------
    rmse_r2_groups : dict
        DESCRIPTION.
    model_groups : dict
        DESCRIPTION.
    all_regions : list
        DESCRIPTION.
    regions_groups_dict : dict[str, list[str]]
        DESCRIPTION.
    drop : set[str], optional
        DESCRIPTION. The default is None.

    Returns
    -------
    best_rmse_models : TYPE
        DESCRIPTION.
    region_model_match : TYPE
        DESCRIPTION.

    """
    if best_rmse_models == None:
        best_rmse_models = {}
    
        for region_group in rmse_r2_groups.keys():
            lowest_rmse_idx = rmse_r2_groups[region_group].loc[:, 'RMSE'].idxmin()
            best_rmse_models[region_group] = lowest_rmse_idx
    
    # match all regions to best fitting mathematical model model
    region_model_match = {}
    for group_name, region_list in regions_groups_dict.items():

        if isinstance(region_list, str):
            # if only one region is given, make it a list to proceed
            region_list = [region_list]
        
        if group_name == 'all_regions':
            # not directly assigned to regions that are not in the group
            region_model_match["class_ 27"] = model_groups[group_name][models_output_dict[best_rmse_models[group_name]]]
        else:
            for region in region_list:
                region_model_match[region] = model_groups[group_name][models_output_dict[best_rmse_models[group_name]]]
        
    return best_rmse_models, region_model_match


#%%

def projection_data_region_capita(regions_list: list, 
                                  region_model_match: dict, 
                                  years: np.ndarray, 
                                  limit_dict: dict, 
                                  gdp_pc_100: pd.DataFrame, 
                                  pop: pd.DataFrame, 
                                  start_year: int, 
                                  end_year: int) -> pd.DataFrame:
    '''
    takes best fitting mathematical projection model and projects sand consumption per region
    returns DataFrame for each region
    '''
    projection_per_region = []
    path_per_region = []
    projections_and_path = []
    
    # loop over every region
    for region in regions_list:
        # get gdp_pc as predictor
        gdp_region = gdp_pc_100.loc[:, REGION_TO_CLASS_DICT.get(region)]
        
        #get index from first year for pathway adaptation
        start_index = list(gdp_region.index).index(years[0])
        
        # reshape gdp_pc
        gdp_region = gdp_region.to_numpy().reshape(len(gdp_region), 1)
       
        # use predict function and model for projection
        if region_model_match.get(region) is None:
            print(region, 'could not be projected')
            continue
        region_projected_data = region_model_match.get(region).predict(gdp_region)

        
        # numpy array from nd array to 1d array
        region_projected_data = region_projected_data.ravel()
        projection_per_region.append(region_projected_data)

            
    # list of projections of all regions to DataFrame
    projection_per_region = pd.DataFrame(projection_per_region).transpose()
    projection_per_region.columns = pop.columns
    projection_per_region.index = np.arange(start_year, 2101) 
    
    if end_year < 2100:
        # pathways of all regions ot DataFrame
        path_per_region = pd.DataFrame(path_per_region).transpose()
        path_per_region.columns = pop.columns
        path_per_region.index = np.arange(end_year, 2101) 
    
        # concat projection and pathway adaptation together
        projections_and_path = pd.concat([projection_per_region.loc[:end_year+1], path_per_region])

    return projection_per_region, path_per_region, projections_and_path

