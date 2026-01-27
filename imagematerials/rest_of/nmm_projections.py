
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


def cement_projection(scenario: str, path_input_data, path_input_data_image):
    # cement
    cement = ResourceModel(resource_group = 'nmm', resource = 'cement', 
                        image_mat_available = True, start_year = 1971, 
                        scenario=scenario,
                        convert_image=True, end_year = 2023, convert_to_tons = 1/1000_000, 
                        trade_data=False, 
                        path_input_data=path_input_data,
                        path_input_data_image=path_input_data_image)
    # cement net trade
    # Historical export per region for Cement (Mtonne), 1970-2000 + 2100 (constant from 2000 on) 
    # (because export and import did not add up to 0, import has been increased by 25%, see Roorda, page 13)

    group_1 = ['class_ 1']
    group_2 = ["class_ 3"]
    group_3 = ["class_ 4"]
    group_5 = ['class_ 12']
    group_6 = ['class_ 13']
    group_8 = ['class_ 19']
    group_9 = ['class_ 23']
    group_10 = ['class_ 24']
    group_11 = ['class_ 2']
    group_12 = ['class_ 5']
    group_13 = ['class_ 6']
    group_14 = ['class_ 7']
    group_15 = ['class_ 10'] 
    group_17 = ['class_ 11']
    china = ['class_ 20']

    scattered_global_fit = ['class_ 8', 'class_ 9', 'class_ 10', 'class_ 14', 
                            'class_ 15', 'class_ 16', 'class_ 17', 'class_ 18', 
                            'class_ 21', 'class_ 22', 'class_ 25', 'class_ 26'] # get global fit assigned'
    exclude = ['class_ 8', 'class_ 9', 'class_ 26'] # to little and scattered data--> get average data asssigned


    # trajectory not to forseen, will be fitted with global regression

    # for these models a regression will be made
    # all reginos that are not in the high, medium, low will be fitted with the global regression
    cement_grouping = {'all' : all_regions_list_class[:-1],
                    'group_1': group_1,
                    'group_2': group_2,
                    'group_3': group_3,
                    'group_5': group_5,
                    'group_6': group_6,
                    'group_8': group_8,
                    'group_9': group_9,
                    'group_10': group_10,
                    'group_11': group_11,
                    'group_12': group_12,
                    'group_13': group_13,
                    'group_14': group_14,
                    'group_15': group_15,
                    'group_17': group_17,
                    'china': china, 
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
        'group_1' : 'gompertz model',
        'group_2' : 'gompertz model',
        'group_3' : 'gompertz model',
        'group_4' : 'gompertz model',
        'group_5' : 'gompertz model',
        'group_6' :  'gompertz model',
        'group_8' : 'gompertz model',
        'group_9' : 'gompertz model',
        'group_10' : 'gompertz model',
        'group_11' : 'gompertz model',
        'group_12' : 'gompertz model',
        'group_13' : 'gompertz model',
        'group_14' : 'gompertz model',
        'group_15' : 'gompertz model',
        'group_17' : 'gompertz model',
        'china': 'gompertz model',
    }

    bounds = {
        'all' : ([0, 0, 0], [1, 20, 100]),
        'group_1' : ([0, 0, 0], [1, 20, 100]),
        'group_2' : ([0.01, 1, 25], [20, 10, 100]),
        'group_3' : ([0.01, 1, 0], [20, 10, 100]),
        'group_4' : ([0, 0, 0], [1, 20, 100]),
        'group_5' : ([0, 0, 0], [1, 20, 100]),
        'group_6' : ([0, 0, 20], [1, 10, 100]),
        'group_8' : ([0, 0, 0], [1, 20, 100]),
        'group_9' : ([0, 0, 0], [0.35, 20, 100]),
        'group_10' : ([0, 0, 0], [1, 20, 100]),
        'group_11' : ([0, 0, 0], [1, 20, 100]),
        'group_12' : ([0, 0, 0], [0.5, 20, 100]),
        'group_13' : ([0, 0, 0], [0.5, 20, 100]),
        'group_14' : ([0, 0, 0], [0.5, 20, 100]),
        'group_15' : ([0, 0, 0], [1, 20, 100]),
        'group_17' : ([0, 0, 0], [0.3, 20, 100]),
        'china': ([0.9, 0, 30], [1, 20, 100]),
    }

    cement.get_X_max_scaling_factor()
    cement.fit_models(best_rmse_models, bounds)

    # project based on best model
    cement.project_on_total(all_regions_list_class[:-1])
    cement.smooth_out_interpolation_all(10, 2012)
    cement.adjust_alpha_and_project(all_regions_list_class[:-1], 
                        start_year_adjust=2025, 
                        end_year_adjust=2100, 
                        min_alpha=None)
    
    cement.remove_regions_with_no_good_fit_from_region_model_match(exclude)

    return cement


def limestone_projection(scenario: str, path_input_data, path_input_data_image):

    # limestone
    limestone = ResourceModel(resource_group = 'nmm', resource = 'limestone', 
                        image_mat_available = False, start_year = 1970, scenario=scenario,
                        path_input_data=path_input_data,
                        path_input_data_image=path_input_data_image)

    # collect these above defined groups in a dictionary

    limestone.historic_other_fraction_consumption = limestone.historic_consumption_data.copy()

    group_1 = ['class_ 1']
    group_2 = ['class_ 2']
    group_3 = [ 'class_ 7', 'class_ 17', 
                'class_ 24', 'class_ 26']
    group_4 = ['class_ 11', 'class_ 12', 'class_ 21'] 
    group_5 = ['class_ 19', 'class_ 23']
    group_6 = ['class_ 4', 'class_ 6', 'class_ 22']
    group_7 = ['class_ 3', 'class_ 14', 'class_ 15']
    group_8 = ['class_ 20']
    group_9 = ['class_ 18']

    diff = ['class_ 5', 'class_ 10', 'class_ 13', 'class_ 16', 'class_ 25', 'class_ 26']
    other = ['class_ 8', 'class_ 9', 'class_ 25', 'class_ 26']
    exclude = []

    # what is in rest will not be fitted because of outliers - will follow global projections
    rest = all_regions_list_class[:-1]
    rest = [r for r in rest if r not in (group_1+group_2+group_3+group_4+group_5+group_6+group_7+group_8+exclude+diff)]

    limestone_grouping = {'all_regions' : all_regions_list_class[:-1],
                        'group_1': group_1,
                        'group_2': group_2,
                        'group_3': group_3,
                        'group_4': group_4,
                        'group_5': group_5,
                        'group_6': group_6,
                        'group_7': group_7,
                        'group_8': group_8,
                        'group_9': group_9,
                        }


    limestone.data_grouped_regions(regions_grouping = limestone_grouping) 
    limestone.sum_IMAGE_drivers_regions(regions_dict=None)

    # calculate regressors
    limestone.calculate_regressors(limestone.historic_consumption_data)
    best_rmse_models = {
        'all_regions': 'gompertz model',
        'group_1': 'gompertz model',
        'group_2': 'gompertz model',
        'group_3': 'gompertz model',
        'group_4': 'gompertz model',
        'group_5': 'gompertz model',
        'group_6': 'gompertz model',
        'group_7': 'gompertz model',
        'group_8': 'gompertz model',
        'group_9': 'gompertz model'
    }

    bounds = {
        'all_regions': ([0, 0, 0], [15, 20, 100]),
        'group_1': ([0, 0, 0], [15, 20, 100]),
        'group_2': ([0, 0, 0], [15, 20, 100]),
        'group_3': ([0, 0, 0], [15, 20, 100]),
        'group_4': ([0, 0, 0], [15, 20, 100]),
        'group_5': ([0, 0, 0], [15, 20, 100]),
        'group_6': ([0, 0, 0], [15, 20, 100]),
        'group_7': ([0, 0, 0], [15, 20, 100]),
        'group_8': ([0, 0, 0], [15, 20, 100]),
        'group_9': ([0, 0, 0], [15, 10, 100]),
    }

    limestone.get_X_max_scaling_factor()
    limestone.fit_models(best_rmse_models, bounds)

    # limestone.assign_fit_to_groups_not_fitted(['class_ 5', 'class_ 10', 'class_ 18'], 
    #                             assign_model='group_6', 
    #                             model_nr=6)
                                
    limestone.assign_fit_to_groups_not_fitted(['class_ 13', 'class_ 16', 'class_ 5', 'class_ 10', 'class_ 18'], 
                            assign_model='all_regions', 
                            model_nr=6)
    
    # limestone.assign_fit_to_groups_not_fitted(other, 
    #                     assign_model='group_9', 
    #                     model_nr=6)

    limestone.assign_fit_to_groups_not_fitted(other, 
                            assign_model='all_regions', 
                            model_nr=6)

    limestone.remove_regions_with_no_good_fit_from_region_model_match(exclude)

    return limestone


def sand_projections(scenario: str, path_input_data, path_input_data_image):
    # sand
    sand = ResourceModel(resource_group = 'nmm', resource = 'sand_gravel_crushed_rock', 
                        image_mat_available = True, start_year = 1970, scenario=scenario,
                        path_input_data=path_input_data,
                        path_input_data_image=path_input_data_image)

    # collect these above defined groups in a dictionary

    group_1 = ['class_ 1']  # Canada
    group_2 = ['class_ 20']  # China
    group_3 = ['class_ 7']  # Average ASSIGN GLOBAL FIT
    group_4 = ['class_ 3', 'class_ 6', 'class_ 10']  # Lower
    group_5 = ['class_ 23']  # Japan
    group_6 = ['class_ 2', 'class_ 24', 'class_ 11']  # High
    group_7 = ['class_ 14', 'class_ 15']  # lower average
    group_8 = ['class_ 18', 'class_ 22']  # indonesia
    group_9 = ['class_ 4', 'class_ 10']  # South africa
    group_10 = ['class_ 12', 'class_ 13', 'class_ 16', 'class_ 19']
    group_11 = ['class_ 17']

    # collect these above defined groups in a dictionary
    SAND_GROUPING_REGIONS = {
        'all_regions': [k for k in CLASS_TO_REGION_DICT.keys() if k != 'class_ 27'],
        'group_1':  group_1,
        'group_2':  group_2,
        'group_3':  group_3,
        'group_4':  group_4,
        'group_5':  group_5,
        'group_6':  group_6,
        'group_7':  group_7,
        'group_8':  group_8,
        'group_9':  group_9,
        'group_10': group_10,
        'group_11': group_11
    }
    
    # assign to other fit
    scattered_regions = ['class_ 5', 'class_ 15', 'class_ 21']  # 'Average' region

    # no projection, take average
    exclude = ['class_ 8', 'class_ 9','class_ 25', 'class_ 26']  # Exclude 'Rest of World' region

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
    'group_1': 'gompertz model',
    'group_2': 'gompertz model',
    'group_3': 'gompertz model',
    'group_4': 'gompertz model',
    'group_5': 'gompertz model',
    'group_6': 'gompertz model',
    'group_7': 'gompertz model',
    'group_8': 'gompertz model',
    'group_9': 'gompertz model',
    'group_10': 'gompertz model',
    'group_11': 'gompertz model'}

    bounds = {
        'all_regions' : ([0, 0, 0], [15, 20, 100]),
        'group_1' : ([0, 0, 0], [15, 20, 100]),
        'group_2' : ([0, 0, 0], [15, 20, 100]),
        'group_3' : ([0, 0, 0], [15, 20, 100]),
        'group_4' : ([0, 0, 0], [15, 20, 100]),
        'group_5' : ([0, 0, 0], [15, 20, 100]),
        'group_6' : ([0, 0, 0], [15, 20, 100]),
        'group_7' : ([0, 0, 0], [15, 20, 100]),
        'group_8' : ([0, 0, 0], [15, 20, 100]),
        'group_9' : ([0, 0, 0], [15, 20, 100]),
        'group_10' : ([0, 0, 0], [15, 20, 100]),
        'group_11' : ([0, 0, 0], [15, 20, 100]),
    }


    sand.get_X_max_scaling_factor()
    sand.fit_models(best_rmse_models=rmse_models, bounds=bounds)

    sand.assign_fit_to_groups_not_fitted(scattered_regions, 
                                assign_model='all_regions', 
                                model_nr=6)
    
    sand.remove_regions_with_no_good_fit_from_region_model_match(exclude)

    return sand

def clay_projections(scenario: str, path_input_data, path_input_data_image):
    # clay
    clay = ResourceModel(resource_group = 'nmm', resource = 'clays', 
                        image_mat_available = False, start_year = 1970, 
                        scenario=scenario,
                        path_input_data=path_input_data,
                        path_input_data_image=path_input_data_image)

    clay.historic_other_fraction_consumption = clay.historic_consumption_data.copy()

    # collect these above defined groups in a dictionary
    low_steady = ['class_ 1', 'class_ 2', 'class_ 23']
    high_steady = ['class_ 11', 'class_ 17', 'class_ 24']
    medium = ['class_ 7', 'class_ 13', 'class_ 19', 'class_ 21'] 
    china = ['class_ 20']

    # not fitted
    low_gdp = ['class_ 3', 'class_ 4', 'class_ 5', 'class_ 6', 
               'class_ 8', 'class_ 9', 'class_ 15', 'class_ 22', 'class_ 25', 'class_ 26']
    
    no_trajectory_forseen = ['class_ 10', 'class_ 12',
                            'class_ 14', 'class_ 16']

    outliers = ['class_ 18']  # 18 : high per capita consumption on a very low gdp per capita
    
    exclude = []


    # trajectory not to forseen, will be fitted with global regression

    # what is in rest will not be fitted because of outliers - will follow global projections       
    rest = all_regions_list_class[:-1]
    rest = [r for r in rest if r not in (high_steady+medium+low_steady+china)]

    clay_grouping = {'all_regions' : all_regions_list_class[:-1],
                    'low_steady' : low_steady,
                    'high_steady' : high_steady,
                    'medium' : medium,
                    'china' : china
                    }

    clay.data_grouped_regions(regions_grouping = clay_grouping) 
    clay.sum_IMAGE_drivers_regions(regions_dict=None)

    # calculate regressors
    clay.calculate_regressors(clay.historic_consumption_data)

    best_rmse_models = {
        'all_regions': 'gompertz model',
        'low_steady' : 'gompertz model',
        'high_steady' : 'gompertz model',
        'medium' : 'gompertz model',
        'china' : 'gompertz model'
    }

    bounds = {
        'all_regions': ([0, 0, 0], [10, 20, 100]),
        'low_steady': ([0, 0, 0], [10, 20, 100]),
        'high_steady': ([0, 0, 0], [10, 20, 100]),
        'medium' : ([0, 0, 0], [10, 20, 100]),
        'china' : ([0, 0, 0], [10, 20, 100])
    }

    clay.get_X_max_scaling_factor()
    clay.fit_models(best_rmse_models, bounds)
    clay.assign_fit_to_groups_not_fitted(low_gdp, 
                                     assign_model='all_regions', 
                                     model_nr=6)

    clay.assign_fit_to_groups_not_fitted(no_trajectory_forseen+outliers, 
                                        assign_model='all_regions', 
                                        model_nr=6)
    # will take historic average instead
    clay.remove_regions_with_no_good_fit_from_region_model_match(exclude+low_gdp)

    return clay