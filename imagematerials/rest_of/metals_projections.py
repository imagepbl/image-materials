import numpy as np

import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter('ignore', OptimizeWarning)
warnings.simplefilter('ignore', RuntimeWarning)


from imagematerials.rest_of.resource_model import ResourceModel
from imagematerials.rest_of.const import (REGION_TO_CLASS_DICT,
                                          COPPER_AVERAGE_REGIONS_TO_IMAGE, 
                                          IAI_TO_IMAGE_CLASSES,
                                          all_regions_list_class)


# COPPER
def copper_projection(scenario: str, path_input_data, path_input_data_image):
    copper = ResourceModel(resource_group = 'metals', resource = 'copper', 
                        image_mat_available = True, start_year = 1990,
                        scenario= scenario, end_year = 2011, 
                        path_input_data=path_input_data,
                        path_input_data_image=path_input_data_image
                        )

    group_1 = ["class_ 1", "class_ 24"]
    group_2 = ["class_ 11", "class_ 23"]
    group_3 = ["class_ 3", "class_ 13", "class_ 15", "class_ 16"]
    group_4 = ["class_ 20"]
    group_5 = ["class_ 2"]
    group_6 = ["class_ 19"]

    # not enough data to make a projection, exclude
    exclude = ["class_ 4", "class_ 5", "class_ 6", "class_ 7", "class_ 8",  
                "class_ 9", "class_ 14", "class_ 17", "class_ 25", "class_ 26"]

    scattered = ["class_ 10", "class_ 12"] # assign global fit
    low = ['class_ 18', "class_ 21",  "class_ 22"] # assign lowest fit


    # for these models a regression will be made
    # all reginos that are not in the high, medium, low will be fitted with the global regression
    copper_regions = {'all_regions' : all_regions_list_class[:-1],
                'group_1': group_1,
                'group_2': group_2,
                'group_3': group_3,
                'group_4': group_4,
                'group_5': group_5,
                'group_6': group_6
                }

    copper.data_grouped_regions(regions_grouping = copper_regions) #list(COPPER_AVERAGE_REGIONS_TO_IMAGE.keys()
    copper.sum_IMAGE_drivers_regions(regions_dict=None)
    copper.match_MAT_data_to_regions_year(match_external_regions=False)
    copper.calculate_historic_other_fraction()
    copper.historic_other_fraction_consumption[copper.historic_other_fraction_consumption < 0] = np.nan

    # Fit models 
    best_rmse_models= {
        'all_regions' : 'gompertz model',
        'group_1' : 'gompertz model',
        'group_2': 'gompertz model',
        'group_3': 'gompertz model',
        'group_4': 'gompertz model',
        'group_5': 'gompertz model',
        'group_6': 'gompertz model'
    }

    bounds = {
        'all_regions' : ([0, 0, 0], [1, 20, 100]),
        'group_1': ([0, 0, 0], [1, 20, 100]),
        'group_2': ([0, 0, 0], [1, 20, 100]),
        'group_3': ([0, 0, 0], [1, 20, 100]),
        'group_4': ([0, 0, 0], [1, 20, 100]),
        'group_5': ([0, 0, 0], [1, 20, 100]),
        'group_6': ([0, 0, 0], [1, 20, 100]),
    }

    copper.calculate_regressors(copper.historic_other_fraction_consumption)
    copper.get_X_max_scaling_factor()
    copper.fit_models(best_rmse_models=best_rmse_models, bounds=bounds)
    
    copper.assign_fit_to_groups_not_fitted(list_regions=low, 
                                        assign_model='group_3', model_nr=1)
    copper.assign_fit_to_groups_not_fitted(list_regions=scattered, 
                                           assign_model='all_regions', model_nr=1)

    copper.remove_regions_with_no_good_fit_from_region_model_match(exclude)

    return copper, copper_regions


# Steel projections
def steel_projection(scenario: str, path_input_data, path_input_data_image):
    # Steel
    steel = ResourceModel(resource_group = 'metals', resource = 'steel', 
                        image_mat_available = True, start_year = 1971,
                        scenario=scenario,
                        convert_image=True, end_year = 2024, convert_to_tons = 1/1000_000, 
                        trade_data=True, 
                        path_input_data=path_input_data,
                        path_input_data_image=path_input_data_image)
    
    group_2 = ['class_ 1'] 
    group_3 = ['class_ 3', 'class_ 10', 'class_ 17']
    group_4 = ['class_ 24']
    group_5 = ['class_ 19', 'class_ 23']
    group_6 = ['class_ 20']
    group_7 = ['class_ 2', 'class_ 11' , 'class_ 13', 'class_ 12', "class_ 21"]
    group_8 = ["class_ 5", "class_ 6", "class_ 7"]

    # trajectory not to forseen, will be fitted with global regression
    spreaded_and_global = ['class_ 14', 'class_ 15', 'class_ 16']
    low = ['class_ 4', "class_ 25", "class_ 26", 'class_ 18', 'class_ 22']  # regions with very low consumption and no increase over time --> assign lowest fit
    exclude = ['class_ 8', "class_ 9"]  # regions excluded from the analysis no enough data to project

    # for these models a regression will be made
    steel_grouping = {'all_regions' : all_regions_list_class[:-1],
                    'group_2': group_2,
                    'group_3': group_3,
                    'group_4': group_4,
                    'group_5': group_5,
                    'group_6': group_6,
                    'group_7': group_7,
                    'group_8': group_8
                }
    
    steel.data_grouped_regions(regions_grouping = steel_grouping)

    # get drivers for fitting (regions dont need to be summed, regions dict is none)
    steel.sum_IMAGE_drivers_regions(regions_dict=None)

    steel.match_MAT_data_to_regions_year(match_external_regions=False)
    steel.calculate_historic_other_fraction()

    # deal with single negative numbers by removing them from dataset
    steel.historic_other_fraction_consumption[steel.historic_other_fraction_consumption < 0] = np.nan
    
    # Fit models 
    steel.calculate_regressors(steel.historic_other_fraction_consumption)

    bounds = {
    'all_regions': ([0, 0, 0], [0.5, 20, 100]),
    'group_2': ([0, 0, 0], [0.5, 20, 100]),
    'group_3': ([0, 0, 0], [0.5, 20, 100]),
    'group_4': ([0, 0, 0], [0.5, 20, 100]),
    'group_5': ([0, 0, 0], [0.5, 20, 100]),
    'group_6': ([0, 0, 0], [0.5, 20, 100]),
    'group_7': ([0, 0, 0], [0.5, 20, 100]),
    'group_8': ([0, 0, 0], [0.5, 20, 100])}

    # enforce that for all groups gompertz model is selected as best fit
    steel.fit_models(best_rmse_models={'all_regions' : 'gompertz model',
                                    'group_2': 'gompertz model',
                                    'group_3': 'gompertz model',
                                    'group_4': 'gompertz model',
                                    'group_5': 'gompertz model',
                                    'group_6': 'gompertz model',
                                    'group_7': 'gompertz model',
                                    'group_8': 'gompertz model'},
                                    bounds=bounds)  
    
    steel.get_X_max_scaling_factor()
    steel.assign_fit_to_groups_not_fitted(spreaded_and_global, 
                                        assign_model='all_regions', 
                                        model_nr=1, 
                                        overwrite_existing=True)
    steel.assign_fit_to_groups_not_fitted(low,
                                        assign_model='group_8',
                                        model_nr=1,
                                        overwrite_existing=True)
    

    
    steel.remove_regions_with_no_good_fit_from_region_model_match(exclude)
    
    return steel, steel_grouping


# Aluminium
def aluminium_projection(scenario: str, path_input_data, path_input_data_image):

    # Aluminium
    aluminium = ResourceModel(resource_group = 'metals', resource = 'aluminium', 
                        image_mat_available = True, start_year = 1998, 
                        scenario=scenario, end_year = 2014,
                        path_input_data=path_input_data,
                        path_input_data_image=path_input_data_image
                        )

    all_regions = ['Africa',
            'Estimated Unreported to IAI', 
            'Gulf Cooperation Council',
            'North America', 
            'Russia & Eastern Europe', 
            'South America',
            'Western & Central Europe',
            'Japan',
            'Asia (ex China)',
            'Oceania',
            'China (Estimated)']
    russia = ['Russia & Eastern Europe']
    north_america = ['North America']
    europe = ['Western & Central Europe']
    japan = ['Japan']
    oceania = ['Oceania']
    south_america = ['South America']

    low = ['Africa', 'Asia (ex China)',]

    # own category because fitting otherwise somehow not possible
    china = ['China (Estimated)', 'Gulf Cooperation Council']

    # will be fitted to global curve or according to IMAGE Mat with some additions
    # rest = ['Estimated Unreported to IAI'] # not necessary, are in low
    exclude = ['class_ 6', 'class_ 8', 'class_ 10', 'class_ 26'] 

    aluminium_regions = {
    'all_regions' : all_regions,
    'russia' : russia,
    'north_america' : north_america,
    'china' : china,
    'europe' : europe,
    'low' : low,
    'japan' : japan,
    'oceania' : oceania,
    'south_america' : south_america,
    }
    
    aluminium.data_grouped_regions(regions_grouping = aluminium_regions) 

    aluminium.sum_IMAGE_drivers_regions(IAI_TO_IMAGE_CLASSES)
    aluminium.match_MAT_data_to_regions_year(match_external_regions=True)
    aluminium.calculate_historic_other_fraction()
    aluminium.historic_other_fraction_consumption[aluminium.historic_other_fraction_consumption < 0] = np.nan

    # Fit models
    aluminium.calculate_regressors(aluminium.historic_other_fraction_consumption)

    best_rmse_models= {
        'all_regions' : 'gompertz model',
        'russia' : 'gompertz model',
        'north_america' : 'gompertz model',
        'china' : 'gompertz model',
        'europe' : 'gompertz model', 
        'low' : 'gompertz model',
        'japan' : 'gompertz model',
        'oceania' : 'gompertz model',
        'south_america' : 'gompertz model',
        'rest' : 'gompertz model'
    }

    bounds = {
        'all_regions' : ([0, 0, 0], [0.02, 20, 100]),
        'russia' : ([0, 0, 0], [0.02, 20, 100]),
        'north_america' : ([0, 0, 0], [0.02, 20, 100]),
        'china' : ([0, 0, 0], [0.02, 20, 100]),
        'europe' : ([0, 0, 0], [0.02, 20, 100]),
        'low' : ([0, 0, 0], [0.02, 20, 100]),
        'japan' : ([0, 0, 0], [0.02, 20, 100]),
        'oceania' : ([0, 0, 0], [0.015, 20, 100]),
        'south_america' : ([0, 0, 0], [0.003, 20, 100]),
        'rest' : ([0, 0, 0], [0.02, 20, 100])
    }

    aluminium.fit_models(best_rmse_models, bounds)

    # add regions to regions model match that are not in there yet becaused they are fitted to the global average
    for key in IAI_TO_IMAGE_CLASSES.keys():
        if key not in aluminium.region_model_match:
            aluminium.region_model_match[key] = None

    aluminium.create_region_model_match_per_image(IAI_TO_IMAGE_CLASSES)
    aluminium.get_X_max_scaling_factor(regions_dict=IAI_TO_IMAGE_CLASSES, 
                                       alu_regions=aluminium_regions)

    aluminium.remove_regions_with_no_good_fit_from_region_model_match(exclude)

    return aluminium, aluminium_regions


