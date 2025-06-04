import numpy as np

import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter('ignore', OptimizeWarning)
warnings.simplefilter('ignore', RuntimeWarning)


from imagematerials.rest_of.resource_model import ResourceModel
from imagematerials.rest_of.const import (REGION_TO_CLASS_DICT,
                                          COPPER_AVERAGE_REGIONS_TO_IMAGE, 
                                          copper_regions,
                                          IAI_TO_IMAGE_CLASSES,
                                          aluminium_regions,
                                          all_regions_list_class)


# COPPER
def copper_projection():
    copper = ResourceModel(resource_group = 'metals', resource = 'copper', 
                        image_mat_available = True, start_year = 2012)

    copper.data_grouped_regions(regions_grouping = copper_regions) #list(COPPER_AVERAGE_REGIONS_TO_IMAGE.keys()
    copper.sum_IMAGE_drivers_regions(COPPER_AVERAGE_REGIONS_TO_IMAGE)
    copper.match_MAT_data_to_regions_year(match_external_regions=True)
    copper.calculate_historic_other_fraction()

    # deal with single negative numbers in historic other fraction by replacing with np.nan
    copper.historic_other_fraction_consumption[copper.historic_other_fraction_consumption < 0] = np.nan

    # deal with negative numbers for Africa by assuming IMAGE Mat values are right + adding a fraction on top that is similar to the other countries

    share_mat_data = copper.image_mat_material_regions/copper.historic_consumption_data
    mean = share_mat_data.loc[:, "China" : "USA"].mean().mean() # MAT data is roughly 50% of total consumption in other regions --> assume same for African regions


    africa_copper_mat = copper.image_mat_material_regions["Africa"]
    tota_africa_new = africa_copper_mat/(mean*100)*100

    # 1- mean is part of total copper consumption that is not covered by MAT
    share_to_project_africa = tota_africa_new*(1-mean)

    # replace nan values for Africa
    copper.historic_other_fraction_consumption["Africa"] = share_to_project_africa

    # Fit models 
    copper.calculate_regressors(copper.historic_other_fraction_consumption)
    copper.fit_models(best_rmse_models=None)

    # Project
    copper.project_on_total(list(COPPER_AVERAGE_REGIONS_TO_IMAGE.keys()))
    copper.project_on_total_IMAGE_regions(REGION_TO_CLASS_DICT, COPPER_AVERAGE_REGIONS_TO_IMAGE)
    
    return copper


# Steel projections
def steel_projection():
    # Steel
    steel = ResourceModel(resource_group = 'metals', resource = 'steel', 
                        image_mat_available = True, start_year = 1971, 
                        convert_image=True, end_year = 2012, convert_to_tons = 1/1000_000, 
                        trade_data=True)
    
    high = ['class_ 14', 'class_ 19', 'class_ 20']
    low = ['class_ 3', 'class_ 4', 'class_ 5', 'class_ 6', 
        'class_ 7', 'class_ 8', 'class_ 9', 'class_ 10', 
        'class_ 11', 'class_ 17', 'class_ 18', 'class_ 21', 
        'class_ 22', 'class_ 24', 'class_ 25', 'class_ 26']
    
    rest = all_regions_list_class[:-1]
    rest = [r for r in rest if r not in (high+low)]

    steel_grouping = {'high' : high,
                    'low' : low,
                    'medium' : rest, 
                    'all_regions' : all_regions_list_class[:-1]}

    steel.data_grouped_regions(regions_grouping = steel_grouping)

    # get drivers for fitting (regions dont need to be summed, regions dict is none)
    steel.sum_IMAGE_drivers_regions(regions_dict=None)

    steel.match_MAT_data_to_regions_year(match_external_regions=False)
    steel.calculate_historic_other_fraction()

    # deal with regions where there are negative values because consumption was lower than mat projections
    steel.historic_other_fraction_consumption[['class_ 4', 'class_ 8', 'class_ 9', 'class_ 22', 'class_ 25', 'class_ 26']]

    # too many negative: class 4, 8, 9, 22, 25, 26
    # assumption: IMAGE MATERIALS is overestimating, 
    # real consumption numbers are true &  
    neg_classes = ['class_ 4', 'class_ 8', 'class_ 9', 'class_ 22', 'class_ 25', 'class_ 26']
    steel.historic_other_fraction_consumption[neg_classes] = steel.historic_consumption_data[neg_classes]

    # deal with single negative numbers by removing them from dataset
    steel.historic_other_fraction_consumption[steel.historic_other_fraction_consumption < 0] = np.nan

    # Fit models 
    steel.calculate_regressors(steel.historic_other_fraction_consumption)
    steel.fit_models(best_rmse_models=None)

    # project based on best model
    steel.project_on_total(all_regions_list_class[:-1])
    
    return steel

# Aluminium
def aluminium_projection():
    # Aluminium
    aluminium = ResourceModel(resource_group = 'metals', resource = 'aluminium', 
                        image_mat_available = True, start_year = 1998, end_year = 2024)

    aluminium.data_grouped_regions(regions_grouping = aluminium_regions)

    aluminium.sum_IMAGE_drivers_regions(IAI_TO_IMAGE_CLASSES)
    aluminium.match_MAT_data_to_regions_year(match_external_regions=True)
    aluminium.calculate_historic_other_fraction()

    # Deal with negative values in other fraction BEFORE recalculation of other fraction consumption
    # 1) Africa: all negative: assumption that apparent MAT is 60% of total consumption (mean until 2017)
    # Share of IAMGE MAT data to total consumption
    share = (aluminium.image_mat_material_regions.loc[:2017]/aluminium.historic_consumption_data.loc[:2017]*100).mean()
    share['Africa']

    # starting at 2013 (when negative values start):
    aluminium.image_mat_material_regions.loc[:, 'Africa'] = aluminium.historic_consumption_data.loc[:, 'Africa'] * share['Africa'] / 100

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

    # 2) rest of Asia: assume that IMAGE Mat is correct and is total per cap consumption 
    # so will be projected on total not on diff
    aluminium.historic_other_fraction_consumption.loc[:, 'Asia (ex China)'] = aluminium.historic_consumption_data.loc[:, 'Asia (ex China)']


    # 3) Estimated Unreported to IAI: assume that IMAGE Mat is correct (these are mostlly IMAGE region 6 regions)
    # Most are small island territories, overseas territories or constituent countries mainly linked to France
    aluminium.historic_other_fraction_consumption.loc[:, 'Estimated Unreported to IAI'] = aluminium.historic_consumption_data.loc[:, 'Estimated Unreported to IAI']

    # 5) Gulf Cooperation Council: remove negative values to nan, no problems in more recent years
    aluminium.historic_other_fraction_consumption.loc[aluminium.historic_other_fraction_consumption.loc[:, 'Gulf Cooperation Council'] < 0, 'Gulf Cooperation Council'] = np.nan

    # Finally check the whole dataframe for single negative values and make them NaN
    aluminium.historic_other_fraction_consumption[aluminium.historic_other_fraction_consumption < 0] = np.nan

    # Fit models
    aluminium.calculate_regressors(aluminium.historic_other_fraction_consumption)
    aluminium.fit_models(best_rmse_models={'all_together': 'logistic growth model',
                                        'china': 'gompertz model',
                                        'oceania': 'gompertz model'}) 
    aluminium.project_on_total(list(IAI_TO_IMAGE_CLASSES.keys()))
    return aluminium


