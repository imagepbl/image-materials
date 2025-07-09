
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter('ignore', OptimizeWarning)

from imagematerials.rest_of.resource_model import ResourceModel
from imagematerials.rest_of.const import (all_regions_list_class, 
                                          models_output_dict,
                                          CLASS_TO_REGION_DICT)


def cement_projection(scenario: str):
    # cement
    cement = ResourceModel(resource_group = 'nmm', resource = 'cement', 
                        image_mat_available = True, start_year = 1971, 
                        scenario=scenario,
                        convert_image=True, end_year = 2012, convert_to_tons = 1/1000_000, 
                        trade_data=True)
    # cement net trade
    # Historical export per region for Cement (Mtonne), 1970-2000 + 2100 (constant from 2000 on) 
    # (because export and import did not add up to 0, import has been increased by 25%, see Roorda, page 13)

    high = ['class_ 16', 'class_ 20']

    medium = ['class_ 1', 'class_ 23',
            'class_ 2',  'class_ 3', 'class_ 11' , 'class_ 24'] 


    # trajectory not to forseen, will be fitted with global regression

    # what is in rset will not be fitted because of outliers - will follow global projections       
    rest = all_regions_list_class[:-1]
    rest = [r for r in rest if r not in (high+medium)]

    # for these models a regression will be made
    # all reginos that are not in the high, medium, low will be fitted with the global regression
    cement_grouping = {'all' : all_regions_list_class[:-1],
                    'high': high,
                    'medium': medium,
                    }

    #cement_grouping = {'all' : all_regions_list_class[:-1]}
    cement.data_grouped_regions(regions_grouping = cement_grouping)

    # get drivers for fitting (regions dont need to be summed, regions dict is none)
    cement.sum_IMAGE_drivers_regions(regions_dict=None)

    cement.match_MAT_data_to_regions_year(match_external_regions=False)
    cement.calculate_historic_other_fraction()

    # deal with single negative numbers by removing them from dataset
    cement.historic_other_fraction_consumption[cement.historic_other_fraction_consumption < 0] = np.nan

    # Fit models 
    cement.calculate_regressors(cement.historic_other_fraction_consumption)

    best_rmse_models={
        'all' : 'gompertz model',
        'high': 'gompertz model',
        'medium': 'gompertz model'}

    bounds = {
        'all' : ([0, 0, 0], [10, 10, 10]),
        'high': ([0, 0, 0], [0.8, 10, 10]),
        'medium': ([0, 5, 0], [10, 10, 10]),
    }

    cement.fit_models(best_rmse_models, bounds)

    # project based on best model
    cement.project_on_total(all_regions_list_class[:-1])
    return cement


def limestone_projection(scenario: str):

    # limestone
    limestone = ResourceModel(resource_group = 'nmm', resource = 'limestone', 
                        image_mat_available = False, start_year = 1970, scenario=scenario)

    # collect these above defined groups in a dictionary

    high = ['class_ 2', 'class_ 20']

    low = ['class_ 1']

    medium = ['class_ 11' , 'class_ 24',] 

    medium_high = ['class_ 23', 'class_ 19']

    # trajectory not to forseen, will be fitted with global regression

    # what is in rset will not be fitted because of outliers - will follow global projections       
    rest = all_regions_list_class[:-1]
    rest = [r for r in rest if r not in (high+medium+low+medium_high)]

    limestone_grouping = {'all' : all_regions_list_class[:-1],
                        'high': high,
                        'medium': medium,
                        'medium_high': medium_high,
                        'low': low,
                        }

    limestone.data_grouped_regions(regions_grouping = limestone_grouping) 
    limestone.sum_IMAGE_drivers_regions(regions_dict=None)

    # calculate regressors
    limestone.calculate_regressors(limestone.historic_consumption_data)

    best_rmse_models = {
    'all': 'gompertz model',
    'high': 'gompertz model',
    'medium': 'gompertz model',
    'low': 'gompertz model',
    'medium_high': 'gompertz model'
    }

    bounds = {
        'all': ([0, 0, 0], [1, 10, 10]),
        'high': ([0, 0, 0], [10, 10, 10]),
        'medium': ([0, 2, 2], [10, 10, 10]),
        'low': ([0, 0, 0], [10, 10, 10]),
        'medium_high': ([0, 0, 0], [10, 10, 10])}

    limestone.fit_models(best_rmse_models, bounds)
    limestone.project_on_total(all_regions_list_class[:-1])

    return limestone


def sand_projections(scenario: str):
    # sand
    sand = ResourceModel(resource_group = 'nmm', resource = 'sand_gravel_crushed_rock', 
                        image_mat_available = True, start_year = 1970, scenario=scenario)

    # collect these above defined groups in a dictionary
    SAND_GROUPING_REGIONS = {
        'all_regions': [k for k in CLASS_TO_REGION_DICT.keys() if k != 'class_ 27'],
        'Canada':  ['class_ 1'],
        'China':   ['class_ 20'],
        'Average': ['class_ 5', 'class_ 12', 'class_ 13', 'class_ 14','class_ 15', 
                    'class_ 16', 'class_ 17', 'class_ 18', 'class_ 19', 
                    'class_ 22', 'class_ 7', 'class_ 21'],
        'Lower':    ['class_ 3', 'class_ 4', 'class_ 6', 'class_ 9', 'class_ 10', 
                    'class_ 8', 'class_ 25', 'class_ 26'],
        'Japan':    ['class_ 23'],
        'High' : ['class_ 2', 'class_ 24', 'class_ 11']
        }

    sand.data_grouped_regions(regions_grouping = SAND_GROUPING_REGIONS) #list(sand_AVERAGE_REGIONS_TO_IMAGE.keys()
    sand.sum_IMAGE_drivers_regions(regions_dict=None)
    sand.match_MAT_data_to_regions_year(match_external_regions=False)
    sand.calculate_historic_other_fraction()

    # clean up negative data
    sand.historic_other_fraction_consumption[sand.historic_other_fraction_consumption < 0] = np.nan
    # calculate regressors
    sand.calculate_regressors(sand.historic_other_fraction_consumption)

    # Fit models 
    rmse_models = {'all_regions': 'gompertz model',
    'Canada': 'log gauss saturate model',
    'China': 'gompertz model',
    'Average': 'gompertz model',
    'Lower': 'gompertz model',
    'Japan': 'log gauss saturate model',
    'High': 'gompertz model'}

    bounds = {
        'all_regions' : ([0, 0, 0], [10, 10, 10]),
        'Canada' : ([0, 0, 0, 0], [10, 10, 10, 8]),
        'China' : ([0, 0, 0], [8, 10, 10]),
        'Average' : ([0, 0, 0], [10, 10, 10]),
        'Lower' : ([0, 0, 0], [10, 10, 10]),
        'Japan' : ([1, 0, 0, 2], [10, 10, 10, 10]),
        'High' : ([1, 5, 0], [10, 10, 10])
    }

    sand.fit_models(best_rmse_models=rmse_models, bounds=bounds)

    # project based on best model
    sand.project_on_total(all_regions_list_class[:-1])
    return sand

def clay_projections(scenario: str):
    # clay
    clay = ResourceModel(resource_group = 'nmm', resource = 'clays', 
                        image_mat_available = False, start_year = 1970, 
                        scenario=scenario)

    # collect these above defined groups in a dictionary

    high = ['class_ 20']

    low = ['class_ 1', 'class_ 2', 'class_ 23', 'class_ 19',
        'class_ 7', 'class_ 13']

    medium = ['class_ 11' , 'class_ 24',] 



    # trajectory not to forseen, will be fitted with global regression

    # what is in rset will not be fitted because of outliers - will follow global projections       
    rest = all_regions_list_class[:-1]
    rest = [r for r in rest if r not in (high+medium+low)]

    clay_grouping = {'all' : all_regions_list_class[:-1],
                    'high': high,
                    'medium': medium,
                        'low': low,
                    }

    clay.data_grouped_regions(regions_grouping = clay_grouping) 
    clay.sum_IMAGE_drivers_regions(regions_dict=None)

    # calculate regressors
    clay.calculate_regressors(clay.historic_consumption_data)

    best_rmse_models = {
    'all': 'gompertz model',
    'high': 'gompertz model',
    'medium': 'gompertz model',
    'low': 'gompertz model',
    }

    bounds = {
        'all': ([0, 0, 0], [10, 10, 10]),
        'high': ([0, 0, 0], [10, 10, 10]),
        'medium': ([0, 2, 2], [10, 10, 10]),
        'low': ([0, 0, 0], [10, 10, 10]),}

    clay.fit_models(best_rmse_models, bounds)
    clay.project_on_total(all_regions_list_class[:-1])

    return clay