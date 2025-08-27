# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:25:57 2024

@author: Arp00003
"""
# This script prepares the data to calculate correlations 

import pandas as pd
from imagematerials.read_mym import read_mym_df

from imagematerials.rest_of.const import path_input_data

#%% Calculate consumption, GDP, and population data both real data points and projections

def calculate_gdp(scenario: str, end_year_gdp_pc: int = 47, end_year_pop: int = 48, 
                  keep_global = False, path_input_data_image = path_input_data):
    """
    Read in gdp per capita data and population data (real & projections) from IMAGE EDITS project to calculate total and global gdp per IMAGE region.

    Parameters
    ----------
    end_year_gdp_pc : int, optional
        end year of gdp per capita data that is returned. 
        The default is 47 (2017).
    end_year_pop : int, optional
        the end year of population data that is returned. The default is 48 (2017)
    image_pop_gd_data: str, optional
        location of image population data
    keep_global : bool, optional
        if Global data should also be provided. The default is True.

    Returns
    -------
    gdp : pd.DataFrame
        gross domestic output per IMAGE region.
    gdp_global : pd.DataFrame
        Global gdp.
    gdp_pc_2017 : pd.DataFrame
        gdp per capita until 2017.
    pop : pd.DataFrame
        population until 2017.
    gdp_pc_2100 : pd.DataFrame
        gdp per capita until 2017.
    pop_100 : pd.DataFrame
        interpolated population until 2100.
    """
    # GDP per capita of IMAGE regions 
    gdp_pc = read_mym_df(path_input_data_image.joinpath(scenario, "Socioeconomic", "gdp_pc.scn"))
    gdp_pc = gdp_pc.drop(columns=27) #drop empty global column
    gdp_pc = gdp_pc.rename(columns={28: 27})
    # rename index to class_ 1, ...
    gdp_pc.columns = [f'class_ {i+1}' for i in range(len(gdp_pc.columns))]

    # Population of IMAGE regions (from EDITS project)
    pop_100: pd.DataFrame = read_mym_df(path_input_data_image.joinpath(scenario, "Socioeconomic", "pop.scn"))
    pop_100 = pop_100.loc[:, :26]
    pop_100.columns = [f'class_ {i+1}' for i in range(len(pop_100.columns))]
    pop_100 = pop_100*1000_000 # convert to millions
    
    # Get exact population data (no projections)
    pop = pop_100.iloc[1:end_year_pop, :] # 1971 - 2017
    
    if keep_global == False:
        gdp_pc_2017 = gdp_pc.iloc[0:end_year_gdp_pc, 0:-1] # 1971 - 2017 & removed sum of ''region 28''
        # gdp until 2100 for projections
        gdp_pc_2100 = gdp_pc.iloc[:, 0:-1] # 1971 - 2100 & removed global 
        # pop = pop.drop(columns=['class_ 27']) #drop empty global column
        # pop_100 = pop_100.drop(columns=['class_ 27']) #drop empty global column
    
    if keep_global == True:
        gdp_pc_2017 = gdp_pc.iloc[0:end_year_gdp_pc, :] # 1971 - 2017 & removed sum of ''region 28''
        gdp_pc_2017 = gdp_pc_2017
        
        gdp_pc_2100 = gdp_pc.iloc[:, :] # 1971 - 2100 & removed global

        pop = pop.drop(columns=['class_ 27']) #drop empty global column
        pop['class_ 27'] = pop.sum(axis = 1)
        
        pop_100 = pop_100.drop(columns=['class_ 27']) #drop empty global column
        pop_100['class_ 27'] = pop_100.sum(axis = 1)
    
    gdp = gdp_pc_2017*pop
    
    # depending wether global values are kept
    if keep_global == False:
        # calculate global GDP
        gdp_global = gdp.sum(axis=1)
        gdp_global = gdp_global.rename('gdp')
        
    if keep_global == True:
        gdp_global = gdp.loc[:, 'class_ 27']
        gdp_global = gdp_global.rename('gdp')
    
    # fill population data with missing years by interpolation to use as x axis
    pop_100 = pop_100.reindex(list(range(1971, 2101))).interpolate(method = 'linear')
    
    # from gdp per capita and population calculate total gdp per IMAGE region and sum to get global GDP
    gdp_100 = gdp_pc_2100*pop_100

    return gdp, gdp_global, gdp_pc_2017, pop, gdp_pc_2100, pop_100, gdp_100


def summarize_IMAGE_regions(material_to_region_dict: dict,
                            gdp: pd.DataFrame,
                            pop: pd.DataFrame,
                            pop_100: pd.DataFrame,
                            gdp_pc_100: pd.DataFrame):
    """
    In case provided data on material consumption is in more aggregared form than IMAGE regions this functions brings them together to new Top-regions.

    Parameters
    ----------
    material_to_region_dict : dict
        Assigns the aggregated regions to IMAGE classes e.g. {"region1": ["class_ 1", "class_ 3"]}.
    gdp : pd.DataFrame
        gdp data per IMAGE region.
    pop : pd.DataFrame
        population data per IMAGE region.
    pop_100 : pd.DataFrame
        population projections per IMAGE regions.
    gdp_pc_100 : pd.DataFrame
        gdp per capita until 2100 per IMAGE regions.

    Returns
    -------
    gdp_adapted : pd.DataFrame
        gdp per new aggregated regions.
    pop_adapted : pd.DataFrame
        pop per new aggregated regions.
    pop_100_adapted : pd.DataFrame
        pop until 2100 per new aggregated regions.
    gdp_pc_100_adapted : pd.DataFrame
        gdp pc until 2100 per new aggregated regions.

    """
    # create empty pd.DataFrames to store new aggregated values in
    gdp_adapted = pd.DataFrame()
    pop_adapted = pd.DataFrame()
    pop_100_adapted = pd.DataFrame()
    gdp_pc_100_adapted = pd.DataFrame()
    gdp_100_adapted = pd.DataFrame()
    
    # calculate gdp until 100 
    gdp_100 = gdp_pc_100*pop_100

    # bring gdp and population data to same resultion as  available source for specific material 
    # Sum IMAGE classes (regions) according to material_to_region_dict
    for value, key in material_to_region_dict.items():
        gdp_adapted[value] = gdp[key].sum(axis = 1)
        pop_adapted[value] = pop[key].sum(axis = 1)
        pop_100_adapted[value] = pop_100[key].sum(axis=1)
        gdp_100_adapted[value] = gdp_100[key].sum(axis=1)

    # gdp pc can not be summed but needs to be rescaled to per capita level
    
    gdp_pc_100_adapted = gdp_100_adapted/pop_100_adapted
        
    return gdp_adapted, pop_adapted, pop_100_adapted, gdp_pc_100_adapted


def sum_total_over_grouped_regions(regions_dict: dict, image_mat_data: pd.DataFrame):
    image_mat_material_regions = pd.DataFrame()
    # sum consumption over image regions
    for name, region_list in regions_dict.items():
        image_mat_material_regions[name] =  image_mat_data.loc[:, region_list].sum(axis = 1)
        
    return image_mat_material_regions 
    

def calculate_material_consumption_pc_and_gdp_pc_groups(regions_groups_dict: dict, 
                                                        gdp_pc: pd.DataFrame,
                                                        cons_capita: pd.DataFrame):
    cons_pc_groups = {}
    gdp_pc_groups = {}
    
    # for every clustered group of regions calculate consumption and gdp per capita
    for key, value in regions_groups_dict.items():
        cons_pc_groups[key] = cons_capita[value]
        gdp_pc_groups[key] = gdp_pc[value]
    
    return cons_pc_groups, gdp_pc_groups