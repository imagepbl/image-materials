# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:31:40 2024

@author: Arp00003
"""

import pandas as pd
import numpy as np

from const import (COPPER_AVERAGE_REGIONS_TO_IMAGE, CLASS_TO_REGION_DICT, 
                   REGION_TO_CLASS_DICT, SAND_GROUPING_REGIONS, path_input_data)

from correlation_materials import (calculate_gdp, summarize_IMAGE_regions, 
                                   calculate_material_consumption_pc_and_gdp_pc_groups, 
                                   sum_total_over_grouped_regions)
from projections_materials import estimate_models_per_region_group, match_regions_to_best_model

class ResourceModel():
    '''
    initialize basic resource model with resource data and gdp, pop data
    '''
    def __init__(self, resource_group: str, resource: str, start_year: int, image_mat_available: bool, end_year = 2017):
        
        # Name resource group
        self.resource_group = resource_group
        # Name resource
        self.resource = resource
        
        # historic conumpstion data
        self.historic_consumption_data = pd.read_csv(f'{path_input_data}/{resource_group}/{self.resource}.csv', 
                                                     index_col=0).loc[:end_year]
        
        # data if IMAGE Mat calculations available
        if image_mat_available == True:
            self.image_mat_data = pd.read_csv(f'{path_input_data}/{resource_group}/image_mat_{self.resource}.csv', 
                                              index_col=0)
        
        # start year & end year of data
        self.start_year = start_year
        self.end_year = end_year
        
        # regions that are available for analysis
        self.regions = self.historic_consumption_data.columns.to_list()
        
        # IMAGE data regarding gdp, population, ...
        (self.gdp_original, self.gdp_global_original, 
         self.gdp_pc_original, self.pop_original, 
         self.gdp_pc_100_original, self.pop_100_original,
         self.gdp_100) = calculate_gdp()
        
        
    def data_grouped_regions(self, regions_grouping):
        # IMAGE regions that are analysed together
        self.region_groups = regions_grouping 
        
    
    def sum_IMAGE_drivers_regions(self, regions_dict):
        # dict from total external data on consumption to IMAGE regions
        self.regions_dict = regions_dict 
        # gdp etc per region is overwritten if IMAGE regions are summarized
        (self.gdp, self.pop, 
         self.pop_100, self.gdp_pc_100) = summarize_IMAGE_regions(regions_dict, 
                                                                  self.gdp_original, 
                                                                  self.pop_original, 
                                                                  self.pop_100_original, 
                                                                  self.gdp_pc_100_original)
        
        
    def match_MAT_data_to_regions_year(self, match_external_regions: bool):  
        """
        Use TRUE this if IMAGE Mat data needs to be matched to grouped regions other than IMAGE 26.
        jump if no IMAGE Mat data available

        Use FASE if IMAGE Mat data does not need to be matched to grouped regions other than IMAGE 26.
        jump if no IMAGE Mat data available

        """
        # EITHER THIS
        # match IMAGE Mat data to regions of data source
        if match_external_regions == True:
            image_mat_material_regions = sum_total_over_grouped_regions(self.regions_dict, 
                                                                        self.image_mat_data)
            
            # select years used for analysis
            self.image_mat_material_regions = image_mat_material_regions.loc[str(self.start_year)
                                                                             : str(self.end_year)]
            # substraction to get difference of total material consumption and IMAGE Mat projections 
            # only works if index of both df is equal
            self.image_mat_material_regions.index = self.historic_consumption_data.loc[self.start_year
                                                                                     :self.end_year].index
        # OR THIS
        if match_external_regions == False:
            
            self.image_mat_data = self.image_mat_data.T
            self.image_mat_data = self.image_mat_data.loc[self.start_year:str(self.end_year)]
            self.historic_consumption_data = self.historic_consumption_data.loc[self.start_year:self.end_year]
            
            self.image_mat_data.index = self.historic_consumption_data.index
            

    def calculate_historic_other_fraction(self):
        
        self.historic_other_fraction_consumption = self.historic_consumption_data - self.image_mat_material_regions
        
        
    def calculate_regressors(self, historic_consumption: pd.DataFrame):
        
        """
        historic_consumption: either total or other fraction (diff) depending what is available
        """
        
        #calculate gdp per capita
        self.gdp_pc = self.gdp_pc_100.loc[self.start_year:self.end_year]
        
        # from total consumption per region and population derive consumption per cap
        self.cons_capita = historic_consumption/self.pop
        self.cons_capita = self.cons_capita.loc[self.start_year:self.end_year]
        
        # # get per capita copper consumption of difference
        # self.cons_diff_pc = self.diff/self.pop.loc[self.start_year:self.end_year]
        
        # make cons_data to same length as diff_data
        self.historic_consumption_data_adapted_years = self.historic_consumption_data.loc[self.start_year:self.end_year]
        
        # get dict of regions that are fitted together (list of names, gdp per cap and cons per cap)
        (self.cons_pc_groups, 
         self.gdp_pc_groups) = calculate_material_consumption_pc_and_gdp_pc_groups(self.region_groups, 
                                                                                   self.gdp_pc, 
                                                                                   self.cons_capita)
               
    
    def fit_models(self):
        # fit all groups of regions to mathematical models and do statistical analysis (RMSE and R2)
        (self.model_groups, 
         self.rmse_r2_groups, 
         self.merged_rmse_r2) = estimate_models_per_region_group(self.region_groups, 
                                                                 self.cons_pc_groups, 
                                                                 self.gdp_pc_groups)
    
        # select best fitting model per group of region and match them back to every 26 IMAGE regions
        (self.best_rmse_models, 
         self.region_model_match) = match_regions_to_best_model(self.rmse_r2_groups, 
                                                                self.model_groups, 
                                                                self.region_groups)           
            
              
    def project_on_total(self, regions_list, REGION_TO_CLASS_DICT, gdp_pc):
        self.projection_per_region = []

        # loop over every region
        for region in regions_list:
            # get gdp_pc as predictor
            gdp_pc_region = self.gdp_pc_100.loc[self.end_year:, region]
            
            # reshape gdp_pc
            gdp_pc_region = gdp_pc_region.to_numpy().reshape(len(gdp_pc_region), 1)
           
            # use predict function and model for projection
            if self.region_model_match.get(region) is None:
                # print(region, 'could not be projected')
                continue
            region_projected_data = self.region_model_match.get(region).predict(gdp_pc_region)
            # numpy array from nd array to 1d array
            region_projected_data = region_projected_data.ravel()
            self.projection_per_region.append(region_projected_data)
            
            # list of projections of all regions to DataFrame
        self.projection_per_region = pd.DataFrame(self.projection_per_region).transpose()
        self.projection_per_region.columns = self.pop.columns
    
        self.projection_per_region.index = np.arange(2017, 2101)

    
    def project_on_total_IMAGE_regions(self, REGION_TO_CLASS_DICT, GROUPS_TO_IMAGE_DICT):
        self.projection_per_region_IMAGE = []

        # loop over every region
        for region in REGION_TO_CLASS_DICT.values():
            if region == "class_ 27":
                continue
            # get gdp_pc as predictor
            gdp_pc_region = self.gdp_pc_100_original.loc[self.end_year:, region]
            
            # reshape gdp_pc
            gdp_pc_region = gdp_pc_region.to_numpy().reshape(len(gdp_pc_region), 1)   

            for key, value in GROUPS_TO_IMAGE_DICT.items():
                if region in value:
                    # use predict function and model for projection
                    if self.region_model_match.get(key) is None:
                        # print(region, 'could not be projected')
                        continue
                    region_projected_data = self.region_model_match.get(key).predict(gdp_pc_region)
                    # numpy array from nd array to 1d array
                    region_projected_data = region_projected_data.ravel()
                    self.projection_per_region_IMAGE.append(region_projected_data)
    
            # list of projections of all regions to DataFrame
        self.projection_per_region_IMAGE = pd.DataFrame(self.projection_per_region_IMAGE).transpose()
        self.projection_per_region_IMAGE.columns = self.pop_original.columns
        
        self.projection_per_region_IMAGE.index = np.arange(2017, 2101)
     
        





# sand.calculate_regressors(sand.diff_historic)
# sand.fit_models()
# sand.replace_sand_best_rmse_models()
# sand.project_on_total(list(CLASS_TO_REGION_DICT.keys())[:-1], REGION_TO_CLASS_DICT)


# LIMESTONE
# limestone = ResourceModel(resource_group = 'non_metallic', resource = 'limestone', image_mat_data = None, start_year = 1971)
# limestone.data_grouped_regions(regions_grouping = SAND_GROUPING_REGIONS)
# limestone.calculate_regressors()
# limestone.fit_models()
# limestone.project_on_total(list(CLASS_TO_REGION_DICT.keys())[:-1], REGION_TO_CLASS_DICT)


# %% CONTINUE HERE
# copper.projection_per_region.plot()

# TODO:
# why are projections all the same, should be based on diff gdp_pc_groups, therefore different?? eventhough they were grouped and same regression, they should be different
# which projection makes actually sense? 
# china out of grouping, all analyzed together?
# go back to how you defined this for sand with the scatter plot! similar gdp_pc consumption is fitted together


# # SAND
# sand = ResourceModel(resource_group = 'non_metallic', resource = 'sand_gravel_crushed_rock', image_mat_available = True, start_year = str(1971))
# sand.data_grouped_regions(regions_grouping = SAND_GROUPING_REGIONS)
# sand.match_MAT_data_to_IMAGE()
# sand.calculate_regressors(sand.historic_other_fraction_consumption)
# sand.fit_models()
# sand.project_on_total(list(REGION_TO_CLASS_DICT.values())[:-1], REGION_TO_CLASS_DICT)
