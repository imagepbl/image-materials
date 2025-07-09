# TODO: collect the global values
# TODO: Sankey production --> consumption make a first example and draw out what could be possible
# TODO: Kyra had: Burnedresidues = read_mym_df(path + 'AGWBUR.SCN') # Fraction of above ground residues burned

# Difference woody biofuels and fuelwod & charcoal from wood
# yes, quite different. woody biofuels are only used for 2nd generation bioenergy and produced on dedicated timber plantations with short rotation cycles (10yrs). 
# fuelwood and charcoal are sourced from naturally (managed) forests and far less efficient

#%% import packages
# import packages

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path

from imagematerials.rest_of.const import (parse_dim,
                                          path_input_data)

from imagematerials.read_mym import read_mym_df
#%% Conversion

# cleared: we will calculate everything with dry matter to keep it comparable
# grassconv = 0.85 # Kyra: conversion to freshweight for grass (moisture content of 15% in Material flows report, 86% from Jonathan); IRP annex: 15% moisture content 
gg_to_bt = 0.000001 #convert Gg to bt--> 1Gg is 1000 tonnes, is therefore 0.001bt
kt_to_bt = 1e-06 # kilotons to billion tons (because the initial unit for wood is already 1000m^3/year)

# TODO: here we are introducing quite some insecurity!
wooddensity = 0.7  # kilotons/(1000 m^3) (Kyra: from Jonathan: Faostat forestry) 
# https://www.fao.org/3/a1106e/a1106e05.pdf / https://unece.org/fileadmin/DAM/timber/meetings/forest-products-conversion-factors.pdf
wood_dm = 0.8 # kg-dm/kg fresh (from Jonathan: from FAO year book of forest products)


#%% CROPS (food, feed, total)
# extract different types of crops, etc for total use case --> sum in IRP categories
# information on allocation of crop types --> X:\IMAGE\data\processed\FAO\FAO_21march2018_IMAGE_38crops\ini\cropprod.ini
# according to IRP: all domestic extraction

def split_up_crops(crops_cons, unit_conversion = gg_to_bt, drop_dim = ['DIM_1', 'DIM_2'], selected_DIM1 = 'total', biomass_type = 'crops'):
    # Dimensions 1 and 2 are not important in the following and therefore dropped
    # are directly dropped to remove multiindex using .droplevel
    # all mutiplied to bring from from giga grams to billion tons (IRP unit) for benchmarking

    # sugar crops
    sugar_crops_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "sugar crops")}').droplevel(drop_dim)*gg_to_bt
    
    # cereals
    cereals_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "tropical cereals", "other temperate cereals")}').droplevel(drop_dim)*gg_to_bt
    cereals_cons = cereals_cons.groupby('time').sum() #sum the two different DIM_2 dimensions

    # vegetables and fruits
    veg_fruit_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "maize", "vegetables & fruits")}').droplevel(drop_dim)*gg_to_bt
    veg_fruit_cons = veg_fruit_cons.groupby('time').sum()

    # oil bearing crops
    oil_crops_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "temperate oil crops", "tropical oil crops", "oil, palm fruit")}').droplevel(drop_dim)*gg_to_bt
    oil_crops_cons = oil_crops_cons.groupby('time').sum()

    # roots and tubers
    roots_tubers_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "temperate roots & tubers", "tropical roots & tubers")}').droplevel(drop_dim)*gg_to_bt
    roots_tubers_cons = roots_tubers_cons.groupby('time').sum()

    # rice 
    rice_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "rice")}').droplevel(drop_dim)*gg_to_bt

    # wheat 
    wheat_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "wheat")}').droplevel(drop_dim)*gg_to_bt

    # pulses (& soybeans)
    pulses_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "pulses", "soybeans")}').droplevel(drop_dim)*gg_to_bt
    pulses_cons = pulses_cons.groupby('time').sum()

    # Spice - beverage - pharmaceutical crops & Tobacco
    spices_etc_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "other non-food luxury, spices")}').droplevel(drop_dim)*gg_to_bt

    # fibres
    fibres_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "plant based fibers")}').droplevel(drop_dim)*gg_to_bt
    
    # total
    total_cons = crops_cons.query(f'{parse_dim(biomass_type, 1, selected_DIM1)} and {parse_dim(biomass_type, 2, "total")}').droplevel(drop_dim)*gg_to_bt
    
    return {
         'sugar crops': sugar_crops_cons,
         'cereals': cereals_cons,
         'vegetables and fruits': veg_fruit_cons,
         'oil bearing crops': oil_crops_cons,
         'roots and tubers': roots_tubers_cons, 
         'rice': rice_cons, 
         'wheat': wheat_cons,
         'pulses': pulses_cons,
         'spices etc.': spices_etc_cons,
         'fibres': fibres_cons,
         'total' : total_cons
         }

#%% FEED consumption except crops (is in crops above)
# important: not all feed output is also domestic extraction (DE) --> useful for sankey charts (metabolism)

def split_up_feed(feed_cons, unit_conversion = gg_to_bt, drop_dim = ['DIM_1', 'DIM_2', 'DIM_3']):
    # Food crops for feed is not calculated, as this is already in crops for feed (avoid double counting!)
    # Feed consumption grazing and fodder crops / is DE
    grazed_biomass_feed_cons = feed_cons.query(f'{parse_dim("tfeed", "1", "total")} and {parse_dim ("tfeed", "2", "grass & fodder")} and {parse_dim("tfeed", "3", "total")}').droplevel(drop_dim)*gg_to_bt
    # Feed consumption animal products / not domestic extraction (IRP)
    animal_products_feed_cons = feed_cons.query(f'{parse_dim("tfeed", "1", "total")} and {parse_dim ("tfeed", "2", "animal products")} and {parse_dim("tfeed", "3", "total")}').droplevel(drop_dim)*gg_to_bt
    # Feed consumption residues / not domestic extraction (IRP)
    residues_feed_cons = feed_cons.query(f'{parse_dim("tfeed", "1", "total")} and {parse_dim ("tfeed", "2", "residues")} and {parse_dim("tfeed", "3", "total")}').droplevel(drop_dim)*gg_to_bt
    # Feed consumption scavenging / is DE
    scavenging_feed_cons = feed_cons.query(f'{parse_dim("tfeed", "1", "total")} and {parse_dim ("tfeed", "2", "scavenging")} and {parse_dim("tfeed", "3", "total")}').droplevel(drop_dim)*gg_to_bt
    # Total feed
    total_feed = feed_cons.query(f'{parse_dim("tfeed", "1", "total")} and {parse_dim ("tfeed", "2", "total")} and {parse_dim("tfeed", "3", "total")}').droplevel(drop_dim)*gg_to_bt
    
    return {
        'animal products' : animal_products_feed_cons,
        'residues' : residues_feed_cons,
        'scavenging' : scavenging_feed_cons,
        'grass & fodder': grazed_biomass_feed_cons, # grazed biomass IRP name
        'total' : total_feed
            }
#%% WOOD BUMA 

def mask_index(index, filters):
    """
    index is a pandas.MultiIndex object, filters a dictionary keyed by the index 
    level names and with values the restrictions to impose.
    """
    # Determine mask for each index level to restrict
    level_masks = [index.get_level_values(k) == v for (k, v) in filters.items()]
    # Use bitwise and (.all) to build the desired mask with all restrictions
    return np.all(level_masks, axis=0)


def wood_buildings(buildings_mat, filters: dict[str, str]):
    # Filters should be a dictionary with keys the index names and values the desired restrictions, e.g., {'material': 'wood'}
    
    buildings_mat.index.names = ['region', 'flow', 'type', 'area', 'material']
    buildings_mat = buildings_mat.loc[:, "1970" : "2060"]
    mask = mask_index(buildings_mat.index, filters)
    buildings_mat_wood = buildings_mat[mask] #create a boolean mask to filter for wood
    buildings_mat_wood = buildings_mat_wood.groupby('region').sum() # sum values for the same regions
    #print(buildings_mat_wood)
    buildings_mat_wood = buildings_mat_wood.transpose()*kt_to_bt
    #print(buildings_mat_wood)
    buildings_mat_wood['27'] = buildings_mat_wood.sum(axis = 1) #add column for total value (region: 27 as in IMAGE for consitency)
    
    return buildings_mat_wood

#%% WOOD demand split up

def split_up_wood(wood_demand, buildings_mat: pd.DataFrame,
                  wooddensity = wooddensity, 
                  wood_dm = wood_dm, drop_dim = 'DIM_1'):
    
    # create list of all modelled years to interpolate missing years, as IMAGE output is in 5 year steps
    all_years = list(range(1970, 2101))
    
    wood_demand_pulp = wood_demand.query(f'{parse_dim("wood", "1", "pulpw. & particles")}').droplevel(drop_dim)*wooddensity*wood_dm*kt_to_bt
    wood_demand_pulp = wood_demand_pulp.reindex(all_years).interpolate(method='linear')
    wood_demand_sawlogs = wood_demand.query(f'{parse_dim("wood", "1", "sawlogs, veneer, other")}').droplevel(drop_dim)*wooddensity*wood_dm*kt_to_bt
    wood_demand_sawlogs = wood_demand_sawlogs.reindex(all_years).interpolate(method='linear')
    wood_demand_fuelwood = wood_demand.query(f'{parse_dim("wood", "1", "fuelw. & charcoal")}').droplevel(drop_dim)*wooddensity*wood_dm*kt_to_bt
    wood_demand_fuelwood = wood_demand_fuelwood.reindex(all_years).interpolate(method='linear')
    wood_demand_total = wood_demand.query(f'{parse_dim("wood", "1", "total")}').droplevel(drop_dim)*wooddensity*wood_dm*kt_to_bt
    wood_demand_total = wood_demand_total.reindex(all_years).interpolate(method='linear')
    
    buma_wood = wood_buildings(buildings_mat, filters={'material': 'wood', 'flow' : 'inflow'})
    
    return {
        'pulp & particles' : wood_demand_pulp,
        'sawlogs, veneer, other' : wood_demand_sawlogs,
        'fuelwood & charcoal' : wood_demand_fuelwood,
        'total wood IMAGE' : wood_demand_total,
        'wood buildings (BUMA)' : buma_wood
        }

#TODO: is BUMA in dry matter??

#%% ANIMAL PRODUCTS split up


def split_up_animal_products(animal_products_cons, unit_conversion = gg_to_bt, drop_dim = ['DIM_1', 'DIM_2']):
    # DIM1 as crops, DIM2 always total, as not important from which animal the product is
    # animal products used for food
    animal_products_food = animal_products_cons.query(f'{parse_dim("crops", "1", "food")} and {parse_dim("animalproducts", "2", "total")}').droplevel(drop_dim)*gg_to_bt
    # animal products used for feed
    animal_products_feed = animal_products_cons.query(f'{parse_dim("crops", "1", "feed")} and {parse_dim("animalproducts", "2", "total")}').droplevel(drop_dim)*gg_to_bt
    # animal products used for other use
    animal_products_ohter_use = animal_products_cons.query(f'{parse_dim("crops", "1", "other use")} and {parse_dim("animalproducts", "2", "total")}').droplevel(drop_dim)*gg_to_bt
    # animal products total
    animal_products_total = animal_products_cons.query(f'{parse_dim("crops", "1", "total")} and {parse_dim("animalproducts", "2", "total")}').droplevel(drop_dim)*gg_to_bt
    
    return {
        'animal products for food' : animal_products_food,
        'animal products for feed' : animal_products_feed,
        'animal products other use' : animal_products_ohter_use,
        'animal products total' : animal_products_total
        }

#%% BIOFUEL CROPS split up


def split_up_biofuelcrops(biofuel_crops, unit_conversion = gg_to_bt, drop_dim = ['DIM_1']):
    sugarcane_biofuel = biofuel_crops.query(f'{parse_dim("biofuelcrops", "1", "sugar cane")}').droplevel(drop_dim)*gg_to_bt
    maize_biofuel = biofuel_crops.query(f'{parse_dim("biofuelcrops", "1", "maize")}').droplevel(drop_dim)*gg_to_bt
    woody_biofuel = biofuel_crops.query(f'{parse_dim("biofuelcrops", "1", "woody biofuels")}').droplevel(drop_dim)*gg_to_bt
    grassy_biofuel = biofuel_crops.query(f'{parse_dim("biofuelcrops", "1", "non woody biofuels")}').droplevel(drop_dim)*gg_to_bt
    total_biofuel = biofuel_crops.query(f'{parse_dim("biofuelcrops", "1", "total")}').droplevel(drop_dim)*gg_to_bt
    
    return {
        'sugarcane biofuel' : sugarcane_biofuel,
        'maize biofuel' : maize_biofuel,
        'woody biofuel' : woody_biofuel,
        'grassy biofuel' : grassy_biofuel,
        'total biofuel' : total_biofuel
        }
 
#%% Get consumption (demand) per defined regions or globally 
# this function is adaptable to sum not only globally but also for different regions (currently: global sum only [-1] --> 27) 
# TODO: create dict with IMAGE regions to make this function more adaptable
    

def sum_by_region(splitted_up_biomass: dict[str, pd.DataFrame], columns: [int] = None) -> pd.DataFrame:
    """
    Sums the crops by region.

    Parameters
    ----------
    splitted_up_crops : dict[str, pd.DataFrame]
        The dictionary keyed by the crop name and with values having the corresponding DataFrame time series
    columns : [int], optional
        A list of integers for the columns over which to sum.
        Column index starts at 1. The last column (currently column 27) is already the total sum.
        The default is None, in which case the sum is over all columns, i.e. the last column is of each DataFrame is used.

    Returns
    -------
    splitted_up_crops_df : pd.DataFrame
        The dataframe with the time series of the total crops over the provided regions.

    """
    if columns is None:
        columns = [-1]  #default is global sum
    
    series = []
    for biomass, biomass_df in splitted_up_biomass.items():  # loop over the key-value pairs of the dictionary using items()
        biomass_df = biomass_df.iloc[:, columns]  # retain the columns provided as a DataFrame (even if only one column is retained)
        biomass_series = biomass_df.sum(axis=1)  # sum row-wise across years, makes the DataFrame to a Series
        biomass_series.name = biomass
        series.append(biomass_series)
    
    splitted_up_biomass_df = pd.concat(series, axis=1)
        
    return splitted_up_biomass_df

#%%


def sankey_total_biomass(splitted_up_crops_food, 
                        splitted_up_crops_feed, 
                        splitted_up_biofuel_crops, 
                        splitted_up_animal_products, 
                        splitted_up_wood, 
                        splitted_up_feed, 
                        path_figures,  # path to save the figures
                        year, region):
    
    crops_food = splitted_up_crops_food.get('total').loc[year, region]
    crops_feed_fodder_grass = splitted_up_crops_feed.get('total').loc[year, region] + splitted_up_feed.get('grass & fodder').loc[year, region]
    crops_biofuel_crops = splitted_up_biofuel_crops.get('total biofuel').loc[year, region] - splitted_up_biofuel_crops.get('woody biofuel').loc[2020, 27] #exclude woody from biofuel
    animal_products = splitted_up_animal_products.get('animal products total').loc[year, region]
    animal_products_feed = splitted_up_animal_products.get('animal products for feed').loc[year, region]
    animal_products_food = splitted_up_animal_products.get('animal products for food').loc[year, region]
    crops_biofuel_wood = splitted_up_biofuel_crops.get('woody biofuel').loc[year, region]
    wood_energy = splitted_up_wood.get('fuelwood & charcoal').loc[year, region]
    wood_buildings_ = splitted_up_wood.get('wood buildings (BUMA)').loc[str(year), str(region)] + splitted_up_wood.get('sawlogs, veneer, other').loc[year, region]
    wood_pulp = splitted_up_wood.get('pulp & particles').loc[year, region]
    
    # Define the nodes and links for the Sankey diagram
    
    nodes = {
        'Animal Products' : 0,
        'Crops' : 1,
        'Feed' : 2,
        'Food' : 3,
        'Energy' : 4,
        'Timber Plantations' : 5,
        '(Managed) Forests' : 6,
        'Infrastructure' : 7,
        'Pulp and Particles' : 8,
        'Sawlogs, Veneers, etc' : 9
        }
    
    link_source = [nodes['Crops'], nodes['Crops'], nodes['Crops'], nodes['Feed'], nodes['Animal Products'], 
                   nodes['Animal Products'], nodes['Timber Plantations'], nodes['(Managed) Forests'], 
                   nodes['(Managed) Forests'], nodes['(Managed) Forests']]
    
    link_target = [nodes['Feed'], nodes['Food'], nodes['Energy'], nodes['Animal Products'], nodes['Feed'], 
                   nodes['Food'], nodes['Energy'], nodes['Energy'], 
                   nodes['Infrastructure'], nodes['Pulp and Particles']]
    
    link_value = [crops_feed_fodder_grass, crops_food, crops_biofuel_crops, animal_products, animal_products_feed, 
                  animal_products_food, crops_biofuel_wood, wood_energy, 
                  wood_buildings_, wood_pulp]
    
    fig = go.Figure(data=[go.Sankey( 
        node = dict( 
          thickness = 50, 
          pad = 25,
          line = dict(color = "black", width = 0.3), 
          label = list(nodes.keys()), 
          color = ['#FF6F61', '#6B8E23',
                    '#FFD700',
                    '#FF4500',
                    '#1E90FF',
                    '#8B4513',
                    '#228B22',
                    '#A9A9A9',
                    '#D2691E',
                    '#8A2BE2']
          
        ), 
        link = dict(
            source = link_source,  
            target = link_target, 
            value = link_value,
            color = [
                'rgba(255, 215, 0, 0.5)',   # Crops to Feed (Gold, representing grain)
                'rgba(50, 205, 50, 0.5)',   # Crops to Food (Lime Green, representing fresh produce)
                'rgba(255, 140, 0, 0.5)',   # Crops to Energy (Dark Orange, representing bioenergy)
                'rgba(139, 69, 19, 0.5)',   # Feed to Animal Products (Saddle Brown, representing animal feed)
                'rgba(165, 42, 42, 0.5)',   # Animal Products to Animal Products (Brown, representing livestock)
                'rgba(255, 99, 71, 0.5)',   # Animal Products to Food (Tomato, representing meat products)
                'rgba(139, 0, 0, 0.5)',     # Timber Plantations to Energy (Dark Red, representing biomass)
                'rgba(0, 100, 0, 0.5)',     # Forests to Energy (Dark Green, representing natural biomass)
                'rgba(169, 169, 169, 0.5)', # Forest to Infrastructure (Dark Gray, representing construction materials)
                'rgba(210, 105, 30, 0.5)'   # Forest to Pulp and Particles (Chocolate, representing processed wood products)
            ]
            
      ))]) 
    
    fig.update_layout(
    font_size=30,
    width=1400,  # Increase width
    height=800,  # Increase height
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
    paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
     )
    
    fig.write_image(f"{path_figures}/biomass_sankey.svg")
    fig.write_html(f"{path_figures}/biomass_sankey.html")
    
    return fig, link_source, link_target, link_value 
    

def biomass_data(scenario: str):
    # Crop consumption in Gg dm/yr, per type of use and crop type including other crops. Dimensions:  [5,17,27] (t) , [NUFPT, NFCT, NRT](time)
    crops_cons = read_mym_df(path_input_data.joinpath(scenario, 'Biomass/AGRCONSCTF_DM.OUT')).set_index(["time", "DIM_1", "DIM_2"])  

    # Wood demand per woodtype in 1000m3/yr. Dimensions: [4,27](t), [NWCT,NRT] (time)
    wood_demand = read_mym_df(path_input_data.joinpath(scenario, 'Biomass/WDEMAND.OUT')).set_index(["time", "DIM_1"])

    # Feed consumption per grazing system type, feed product type and animal type in Gg dm/yr Dimensions:  [3,6,6,27] (t), [NGST,NFPT,NAT,NRT] (t)
    feed_cons = read_mym_df(path_input_data.joinpath(scenario, 'Biomass/TFEED.OUT')).set_index(["time", "DIM_1", "DIM_2", "DIM_3"])

    # Animal products  Unit=Gg dm/yr; Label=Consumption of animal products in dry matter, per type of use and animal type [5,6,27] (t) [NUFPT,NAPT,NRT] 
    animal_products_cons = read_mym_df(path_input_data.joinpath(scenario, 'Biomass/AGRCONSA_DM.OUT')).set_index(["time", "DIM_1", "DIM_2"])

    # Biofuel crops production (same as consumption) difference is only made in energy trade, not for actual crops calculation
    # Unit= Gg dm/yr; Label= Production of biofuels (dry matter) [5,27] [NBCT,NRT] (t)
    biofuel_crops = read_mym_df(path_input_data.joinpath(scenario, 'Biomass/AGRPRODBF_dm.OUT')).set_index(["time", "DIM_1"])

    # Materials buildings (BUMA output) in kt
    # TODO: adapt scenario path
    buildings_mat = pd.read_csv('../../../data/raw/rest-of/IMAGE_MAT_out/SSP2_CP/material_output_buma_RASMI.csv', header = 0).set_index(['Unnamed: 0', 'flow', 'type', 'area', 'material'])

    # Split up different biomass types 
    # Crops: Split up crops

    splitted_up_crops_total = split_up_crops(crops_cons, gg_to_bt, drop_dim = ['DIM_1', 'DIM_2'], 
                                                selected_DIM1= 'total', biomass_type = 'crops')
    splitted_up_crops_food = split_up_crops(crops_cons, gg_to_bt, drop_dim = ['DIM_1', 'DIM_2'], 
                                            selected_DIM1= 'food', biomass_type = 'crops')
    splitted_up_crops_feed = split_up_crops(crops_cons, gg_to_bt, drop_dim = ['DIM_1', 'DIM_2'], 
                                            selected_DIM1= 'feed', biomass_type = 'crops')
    splitted_up_crops_other_use = split_up_crops(crops_cons, gg_to_bt, drop_dim = ['DIM_1', 'DIM_2'], 
                                            selected_DIM1= 'other use', biomass_type = 'crops')
    # Feed
    splitted_up_feed = split_up_feed(feed_cons)

    # Wood & wood from buildings
    splitted_up_wood = split_up_wood(wood_demand, buildings_mat)

    # Animal Products
    splitted_up_animal_products = split_up_animal_products(animal_products_cons)

    # Biofuel crops 
    splitted_up_biofuel_crops = split_up_biofuelcrops(biofuel_crops)

    #%% get global values

    crops_total_consumption_global = sum_by_region(splitted_up_crops_total)
    feed_total_consumption_global = sum_by_region(splitted_up_feed)

    return_dict = {
        'splitted_up_crops_total': splitted_up_crops_total,
        'splitted_up_crops_food': splitted_up_crops_food,
        'splitted_up_crops_feed': splitted_up_crops_feed,
        'splitted_up_crops_other_use': splitted_up_crops_other_use,
        'splitted_up_feed': splitted_up_feed,
        'splitted_up_wood': splitted_up_wood,
        'splitted_up_animal_products': splitted_up_animal_products,
        'splitted_up_biofuel_crops': splitted_up_biofuel_crops
    }

    return return_dict