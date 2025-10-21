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
    group_3 = ["class_ 3", "class_ 12", "class_ 13", "class_ 15", 
            "class_ 16", "class_ 18", "class_ 21", "class_ 22"]
    group_4 = ["class_ 10", "class_ 19", "class_ 20"]
    group_5 = ["class_ 2"]


    # trajectory not to forseen, will be fitted with global regression
    no_data = [ "class_ 4", "class_ 5", "class_ 6", "class_ 7", 
            "class_ 8", "class_ 9", "class_ 14", "class_ 17", 
            "class_ 25", "class_ 26"]
    exclude = no_data 

    # for these models a regression will be made
    # all reginos that are not in the high, medium, low will be fitted with the global regression
    copper_regions = {'all_regions' : all_regions_list_class[:-1],
                'group_1': group_1,
                'group_2': group_2,
                'group_3': group_3,
                'group_4': group_4,
                'group_5': group_5
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
        'group_5': 'gompertz model'
    }

    bounds = {
        'all_regions' : ([0, 0, 0], [10, 10, 10]),
        'group_1': ([0, 0, 0], [0.005, 10, 10]),
        'group_2': ([0, 2, 2], [10, 10, 10]),
        'group_3': ([0, 0, 0], [10, 10, 10]),
        'group_4': ([0, 0, 0], [10, 10, 10]),
        'group_5': ([0, 2, 2], [10, 10, 10])
    }

    copper.calculate_regressors(copper.historic_other_fraction_consumption)
    copper.fit_models(best_rmse_models=best_rmse_models, bounds=bounds)
    copper.assign_fit_to_groups_not_fitted(list_regions=all_regions_list_class[:-1], 
                                           assign_model='group_3', model_nr=6)


    # Projections 
    # copper.project_on_total(all_regions_list_class[:-1])

    copper.remove_regions_with_no_good_fit_from_region_model_match(exclude)

    return copper


# Steel projections
def steel_projection(scenario: str, path_input_data, path_input_data_image):
    # Steel
    steel = ResourceModel(resource_group = 'metals', resource = 'steel', 
                        image_mat_available = True, start_year = 1971,
                        scenario=scenario,
                        convert_image=True, end_year = 2012, convert_to_tons = 1/1000_000, 
                        trade_data=True, 
                        path_input_data=path_input_data,
                        path_input_data_image=path_input_data_image)
    
    class_1 = ['class_ 1'] 

    high = ['class_ 19', 'class_ 23']

    china = ['class_ 20']

    low = ['class_ 2', 'class_ 11' , 'class_ 12', 'class_ 13', 'class_ 24']

    very_low = ["class_ 3", "class_ 5", "class_ 6", "class_ 7", 
                  "class_ 17", "class_ 18", "class_ 21"]

    # trajectory not to forseen, will be fitted with global regression
    spreaded = ['class_ 10', 'class_ 14', 'class_ 15', 'class_ 16']

    # will be excluded and assigned average diff
    fit_not_good = ['class_ 5', 'class_ 6', 'class_ 10', 'class_ 14', 
                    'class_ 15', "class_ 18", "class_ 24"]
    
    too_low = ['class_ 4', 'class_ 8', "class_ 9", 
               'class_ 22', 'class_ 25', "class_ 26"]

    exclude = too_low + fit_not_good

    # what is in rest will not be fitted because of outliers - will follow global projections       
    rest = all_regions_list_class[:-1]
    rest = [r for r in rest if r not in (low+class_1+high+very_low+too_low+fit_not_good+china)]

    # for these models a regression will be made
    # all reginos that are not in the high, medium, low will be fitted with the global regression
    steel_grouping = {'all_regions' : all_regions_list_class[:-1],
                      'class_ 1': class_1,
                      'high': high,
                      'china': china,
                      'low': low,
                      'very_low': very_low,
                      'too_low': too_low,
                    }

    #steel_grouping = {'all' : all_regions_list_class[:-1]}
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
    'all_regions': ([0, 0, 0], [10, 10, 10]),
    'class_ 1': ([0, 0, 0], [10, 10, 10]),
    'high': ([0, 0, 0], [0.7, 10, 10]),
    'china': ([0, 0, 0], [0.5, 10, 10]),
    'low': ([0, 0, 0], [10, 10, 10]),
    'very_low': ([0, 0, 0], [10, 10, 10]),
    'too_low': ([0, 0, 0], [10, 10, 10])}

    # enforce that for all groups gompertz model is selected as best fit
    steel.fit_models(best_rmse_models={'all_regions' : 'gompertz model',
                                    'class_ 1': 'gompertz model',
                                    'high': 'gompertz model',
                                    'china': 'gompertz model',
                                    'low': 'gompertz model',
                                    'very_low': 'gompertz model',
                                    'too_low': 'gompertz model'},
                                    bounds=bounds)  

    steel.remove_regions_with_no_good_fit_from_region_model_match(exclude)
    
    return steel


# Aluminium
def aluminium_projection(scenario: str, path_input_data, path_input_data_image):

    # Aluminium
    aluminium = ResourceModel(resource_group = 'metals', resource = 'aluminium', 
                        image_mat_available = True, start_year = 1998, 
                        scenario=scenario, end_year = 2014,
                        path_input_data=path_input_data,
                        path_input_data_image=path_input_data_image
                        )

    # removed outlier reginos so they are not represented in the global fit
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

    low = ['Africa', 'Asia (ex China)',]

    # own category because fitting otherwise somehow not possible
    china = ['China (Estimated)', 'Gulf Cooperation Council']

    # will be fitted to global curve or according to IMAGE Mat with some additions
    rest = ['Estimated Unreported to IAI', 
            'South America',
            'Oceania', 
            'Japan'
        ]

    aluminium_regions = {
        'all_regions' : all_regions,
        'russia' : russia,
        'north_america' : north_america,
        'china' : china,
        'europe' : europe,
        'low' : low
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
        'low' : 'gompertz model'
    }

    bounds = {
        'all_regions' : ([0, 0, 0], [10, 10, 10]),
        'russia' : ([0, 1, 1], [10, 10, 10]),
        'north_america' : ([0, 1, 1], [10, 10, 10]),
        'china' : ([0, 0, 0], [0.03, 10, 10]),
        'europe' : ([0, 0, 0], [0.01, 10, 10]),
        'low' : ([0, 1, 1], [0.01, 10, 10])
    }

    aluminium.fit_models(best_rmse_models, bounds)




    # add regions to regions model match that are not in there yet becaused they are fitted to the global average
    for key in IAI_TO_IMAGE_CLASSES.keys():
        if key not in aluminium.region_model_match:
            aluminium.region_model_match[key] = None

    aluminium.create_region_model_match_per_image(IAI_TO_IMAGE_CLASSES)

    aluminium.assign_fit_to_groups_not_fitted(IAI_TO_IMAGE_CLASSES.get("Estimated Unreported to IAI"), 
                                        assign_model='low', 
                                        model_nr=6)
    aluminium.assign_fit_to_groups_not_fitted(IAI_TO_IMAGE_CLASSES.get("South America"), 
                                    assign_model='low', 
                                    model_nr=6)
    aluminium.assign_fit_to_groups_not_fitted(IAI_TO_IMAGE_CLASSES.get("Oceania") +  
                                              IAI_TO_IMAGE_CLASSES.get("Japan"),
                                           assign_model='all_regions', 
                                           model_nr=6)
    

    
    aluminium.remove_regions_with_no_good_fit_from_region_model_match(['class_ 6', 'class_ 8', 
                                                                       'class_ 10', 'class_ 26'])

    return aluminium


