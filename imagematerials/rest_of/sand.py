import numpy as np
import matplotlib.pyplot as plt

from imagematerials.rest_of.resource_model import ResourceModel
from imagematerials.rest_of.const import (all_regions_list_class, 
                                          SAND_GROUPING_REGIONS)

def sand_projections():
    # sand
    sand = ResourceModel(resource_group = 'nmm', resource = 'sand_gravel_crushed_rock', 
                        image_mat_available = True, start_year = 2012)

    sand.data_grouped_regions(regions_grouping = SAND_GROUPING_REGIONS) #list(sand_AVERAGE_REGIONS_TO_IMAGE.keys()
    sand.sum_IMAGE_drivers_regions(regions_dict=None)
    sand.match_MAT_data_to_regions_year(match_external_regions=False)
    sand.calculate_historic_other_fraction()
    sand.calculate_regressors(sand.historic_other_fraction_consumption)
    sand.fit_models(best_rmse_models=None)
    sand.project_on_total(all_regions_list_class[:-1])

    return sand