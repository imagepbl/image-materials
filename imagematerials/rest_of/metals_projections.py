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
def copper_projection(scenario: str):
    # COPPER
    copper = ResourceModel(resource_group = 'metals', resource = 'copper', 
                        image_mat_available = True, start_year = 2012, 
                        scenario=scenario)

    all_regions = list(COPPER_AVERAGE_REGIONS_TO_IMAGE.keys())
    oceania_eu_usa = ['Europe', 'USA', 'Oceania']
    asia = ['China', 'Japan', 'Korea']
    rest = [region for region in all_regions if region not in oceania_eu_usa + asia]

    copper_regions = {
        'all_regions': all_regions,
        'oceania_eu_usa' : oceania_eu_usa,
        'asia':asia}

    copper.data_grouped_regions(regions_grouping = copper_regions) #list(COPPER_AVERAGE_REGIONS_TO_IMAGE.keys()
    copper.sum_IMAGE_drivers_regions(COPPER_AVERAGE_REGIONS_TO_IMAGE)
    copper.match_MAT_data_to_regions_year(match_external_regions=True)
    copper.calculate_historic_other_fraction()

    # deal with single negative numbers in historic other fraction by replacing with np.nan
    copper.historic_other_fraction_consumption[copper.historic_other_fraction_consumption < 0] = np.nan
    
    # Fit models 
    best_rmse_models= {
        'all_regions' : 'gompertz model',
        'oceania_eu_usa' : 'gompertz model',
        'asia' : 'gompertz model'}


    bounds = {
        'all_regions' : ([0, 0, 0], [10, 10, 10]),
        'oceania_eu_usa' : ([0, 2, 2], [10, 10, 10]),
        'asia' : ([0, 1, 1], [10, 10, 10])}


    copper.calculate_regressors(copper.historic_other_fraction_consumption)
    copper.fit_models(best_rmse_models=best_rmse_models, bounds=bounds)

    # add regions to regions model match that are not in there yet becaused they are fitted to the global average
    for key in COPPER_AVERAGE_REGIONS_TO_IMAGE.keys():
        if key not in copper.region_model_match:
            copper.region_model_match[key] = copper.model_groups.get("all_regions")[6]

    copper.create_region_model_match_per_image(COPPER_AVERAGE_REGIONS_TO_IMAGE)

    # Projections 
    copper.project_on_total(list(COPPER_AVERAGE_REGIONS_TO_IMAGE.keys()))


        
    copper.smooth_out_interpolation_all(10, 2017)
    copper.adjust_alpha_and_project(list(COPPER_AVERAGE_REGIONS_TO_IMAGE.keys()), 
                            start_year_adjust=2025, 
                            end_year_adjust=2100, 
                            min_alpha=None)

    return copper


# Steel projections
def steel_projection(scenario: str):
    # Steel
    steel = ResourceModel(resource_group = 'metals', resource = 'steel', 
                        image_mat_available = True, start_year = 1971,
                        scenario=scenario,
                        convert_image=True, end_year = 2012, convert_to_tons = 1/1000_000, 
                        trade_data=True)

    high = ['class_ 19', 'class_ 20']

    medium = ['class_ 1','class_ 13', 'class_ 16', 'class_ 23'] 

    low = ['class_ 2',  'class_ 3', 'class_ 11' , 'class_ 24']

    # trajectory not to forseen, will be fitted with global regression
    low_gdp = ['class_ 4', 'class_ 5', 'class_ 6', 'class_ 7',
            'class_ 8', 'class_ 9',  'class_ 10', 'class_ 12',
            'class_ 15', 'class_ 17', 'class_ 18', 'class_ 21', 
            'class_ 22', 'class_ 25', 'class_ 26']

    # what is in rset will not be fitted because of outliers - will follow global projections       
    rest = all_regions_list_class[:-1]
    rest = [r for r in rest if r not in (high+low+medium+low_gdp)]

    # for these models a regression will be made
    # all reginos that are not in the high, medium, low will be fitted with the global regression
    steel_grouping = {'all' : all_regions_list_class[:-1],
                    'high': high,
                    'medium': medium,
                    'low': low,
                    }

    #steel_grouping = {'all' : all_regions_list_class[:-1]}
    steel.data_grouped_regions(regions_grouping = steel_grouping)

    # get drivers for fitting (regions dont need to be summed, regions dict is none)
    steel.sum_IMAGE_drivers_regions(regions_dict=None)

    steel.match_MAT_data_to_regions_year(match_external_regions=False)
    steel.calculate_historic_other_fraction()

    neg_classes = ['class_ 4', 'class_ 8', 'class_ 9', 'class_ 22', 'class_ 25', 'class_ 26']
    steel.historic_other_fraction_consumption[neg_classes] = steel.historic_consumption_data[neg_classes]

    # deal with single negative numbers by removing them from dataset
    steel.historic_other_fraction_consumption[steel.historic_other_fraction_consumption < 0] = np.nan
    
    # Fit models 
    steel.calculate_regressors(steel.historic_other_fraction_consumption)

    # enforce that for all groups gompertz model is selected as best fit
    steel.fit_models(best_rmse_models={'all' : 'gompertz model',
                                    'high': 'gompertz model',
                                    'medium': 'gompertz model',
                                    'low': 'gompertz model'})


    # project based on best model
    steel.project_on_total(all_regions_list_class[:-1])
    steel.smooth_out_interpolation_all(10, 2012)
    steel.adjust_alpha_and_project(all_regions_list_class[:-1], 
                               start_year_adjust=2025, 
                               end_year_adjust=2100, 
                               min_alpha=None)

    return steel

# Aluminium
def aluminium_projection(scenario: str):

    # Aluminium
    aluminium = ResourceModel(resource_group = 'metals', resource = 'aluminium', 
                        image_mat_available = True, start_year = 1998, 
                        scenario=scenario, end_year = 2024
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

    europe = ['Western & Central Europe', 'Japan', 'Oceania']

    # own category because fitting otherwise somehow not possible
    china = ['China (Estimated)', 'Gulf Cooperation Council']

    # will be fitted to global curve or according to IMAGE Mat with some additions
    rest = ['Africa',
            'Estimated Unreported to IAI', 
            'South America',
            'Asia (ex China)']

    aluminium_regions = {
        'all_regions' : all_regions,
        'russia' : russia,
        'north_america' : north_america,
        'china' : china,
        'europe' : europe
    }

    aluminium.data_grouped_regions(regions_grouping = aluminium_regions) 

    aluminium.sum_IMAGE_drivers_regions(IAI_TO_IMAGE_CLASSES)
    aluminium.match_MAT_data_to_regions_year(match_external_regions=True)
    aluminium.calculate_historic_other_fraction()


    # Deal with negative values in other fraction BEFORE recalculation of other fraction consumption
    # 1) Africa: for all negative values: assumption that apparent MAT is 60% of total consumption (mean until 2017)
    # Share of IAMGE MAT data to total consumption
    share = (aluminium.image_mat_material_regions.loc[:2017]/aluminium.historic_consumption_data.loc[:2017]*100).mean()
    # starting at 2017 (when negative values start):
    aluminium.image_mat_material_regions.loc[2017:, 'Africa'] = aluminium.historic_consumption_data.loc[2017:, 'Africa'] * share['Africa'] / 100

    # 4) North America
    # problem starts at 2015, so take share mean of IMAGE Mat data before that and apply it to 2015-2024
    share_na = (aluminium.image_mat_material_regions.loc[:2015]/aluminium.historic_consumption_data.loc[:2015]*100).mean()
    aluminium.image_mat_material_regions.loc[:, 'North America'] = aluminium.historic_consumption_data.loc[:, 'North America'] * share_na['North America'] / 100

    # 6) South America: assume that share of Mat until 2010 is correct and that apparent consumption needs to be adjusted accordingly 
    share_sa = (aluminium.image_mat_material_regions.loc[:2010, 'South America'] / aluminium.historic_consumption_data.loc[:2010, 'South America']).mean()
    aluminium.historic_consumption_data.loc[:, 'South America'] = aluminium.image_mat_material_regions.loc[:, 'South America'] / share_sa

    # Deal with negative values in other fraction AFTER recalculation of other fraction consumption
    # redo calculation of historic other fraction consumption 
    aluminium.calculate_historic_other_fraction()

    # 2) rest of Asia: 
    # per capita consmption of total data is too low compared to other regions, so we assume that IMAGE Mat is correct
    # --> will be fitted to global curve 
    # 3) Estimated Unreported to IAI: assume that IMAGE Mat is correct (these are mostly IMAGE region 6 regions)
    # Most are small island territories, overseas territories or constituent countries mainly linked to France
    # --> will be fitted to global curve 

    # 5) Gulf Cooperation Council: remove negative values to nan, no problems in more recent years
    # & Finally check the whole dataframe for single negative values and make them NaN
    aluminium.historic_other_fraction_consumption[aluminium.historic_other_fraction_consumption < 0] = np.nan


    # Fit models
    aluminium.calculate_regressors(aluminium.historic_other_fraction_consumption)

    best_rmse_models= {
        'all_regions' : 'gompertz model',
        'russia' : 'gompertz model',
        'north_america' : 'gompertz model',
        'china' : 'gompertz model',
        'europe' : 'gompertz model'}

    bounds = {
        'all_regions' : ([0, 0, 0], [10, 10, 10]),
        'russia' : ([0, 1, 1], [10, 10, 10]),
        'north_america' : ([0, 1, 1], [10, 10, 10]),
        'china' : ([0, 0, 0], [0.03, 10, 10]),
        'europe' : ([0, 0, 0], [10, 10, 10])
    }

    aluminium.fit_models(best_rmse_models, bounds)



    # add regions to regions model match that are not in there yet becaused they are fitted to the global average
    for key in IAI_TO_IMAGE_CLASSES.keys():
        if key not in aluminium.region_model_match:
            aluminium.region_model_match[key] = aluminium.model_groups.get("all_regions")[6]

    aluminium.create_region_model_match_per_image(IAI_TO_IMAGE_CLASSES)

    # Projections
    aluminium.project_on_total(list(IAI_TO_IMAGE_CLASSES.keys()), start_year_projection=2012)
    aluminium.smooth_out_interpolation_all(10, 2014)
    aluminium.adjust_alpha_and_project(list(IAI_TO_IMAGE_CLASSES.keys()), 
                               start_year_adjust=2025, 
                               end_year_adjust=2100, 
                               min_alpha=None, start_year_projection=2014)


    return aluminium


