import warnings
from scipy.optimize import OptimizeWarning
warnings.simplefilter('ignore', OptimizeWarning)

from imagematerials.rest_of.resource_model import ResourceModel
from imagematerials.rest_of.const import (all_regions_list_class)


def cement_projection():
    # cement
    cement = ResourceModel(resource_group = 'nmm', resource = 'cement', 
                        image_mat_available = True, start_year = 1971, 
                        convert_image=True, end_year = 2012, convert_to_tons = 1/1000_000, 
                        trade_data=True)
    # cement net trade
    # Historical export per region for Cement (Mtonne), 1970-2000 + 2100 (constant from 2000 on) 
    # (because export and import did not add up to 0, import has been increased by 25%, see Roorda, page 13)

    # production unit: Mt
        
    all = all_regions_list_class[:-1]
    cement_grouping = {'all' : all}

    cement.data_grouped_regions(regions_grouping = cement_grouping)

    # get drivers for fitting (regions dont need to be summed, regions dict is none)
    cement.sum_IMAGE_drivers_regions(regions_dict=None)

    cement.match_MAT_data_to_regions_year(match_external_regions=False)
    cement.calculate_historic_other_fraction()

    # Fit models 
    cement.calculate_regressors(cement.historic_consumption_data)
    cement.fit_models(best_rmse_models=None)


    # project based on best model
    cement.project_on_total(all_regions_list_class[:-1])

    return cement

