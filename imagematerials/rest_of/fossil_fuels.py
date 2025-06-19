# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:15:30 2024

@author: Arp00003
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plotly.graph_objects as go

from imagematerials.read_mym import read_mym_df
from imagematerials.rest_of.const import (parse_dim, get_key, DIM2_primary_dict, 
                                          path_figures, path_scenario_data_fossil)

from imagematerials.rest_of.sankey_function import create_node_dict, index_mapper, convert_index_to_node_id, prepare_Sankey_lists

coal_conversion = 28.9 # coal: 28.9 MJ/kg Coal (low bituminous)
oil_conversion = 45.5 # oil: 45.5 MJ/kg Crude oil
gas_conversion = 52.2 # gas: 52.2 MJ/kg Natural gas (95% methane)

mega_to_peta = 1e9
giga_to_peta = 1e6

#%% Primary energy from Joule to kg
def converte_primary_energy_to_mass(fossils_primary):
    coal_converted = fossils_primary.query(parse_dim('primary', '2', 'coal')) * mega_to_peta / coal_conversion
    oil_converted = fossils_primary.query(parse_dim('primary', '2', 'oil')) * mega_to_peta / oil_conversion
    gas_converted = fossils_primary.query(parse_dim('primary', '2', 'natural gas')) * mega_to_peta / gas_conversion
    
    fossils_primary_converteted = pd.concat([coal_converted, oil_converted, gas_converted])
    
    return fossils_primary_converteted




def convert_primary_to_secondary_to_mass(fossils_prim_per_sec):
    coal_converted = fossils_prim_per_sec.query(parse_dim('primsec_reversed', '1', 'coal')) * mega_to_peta / coal_conversion
    conv_oil_converted = fossils_prim_per_sec.query(parse_dim('primsec_reversed', '1', 'conventional oil')) * mega_to_peta / oil_conversion
    unconv_oil_converted = fossils_prim_per_sec.query(parse_dim('primsec_reversed', '1', 'unconventional oil')) * mega_to_peta / oil_conversion
    gas_converted = fossils_prim_per_sec.query(parse_dim('primsec_reversed', '1', 'natural gasses')) * mega_to_peta / gas_conversion
    
    fossils_primsecond_converted = pd.concat([coal_converted, conv_oil_converted, unconv_oil_converted, gas_converted])
    
    return fossils_primsecond_converted
    


def convert_secondary_to_final_mass(fossils_final):
    coal_converted = fossils_final.query(parse_dim('seconden_reversed', '3', 'coal')) * mega_to_peta / coal_conversion
    heavy_oil_converted = fossils_final.query(parse_dim('seconden_reversed', '3', 'heavy oil')) * mega_to_peta / oil_conversion
    light_oil_converted = fossils_final.query(parse_dim('seconden_reversed', '3', 'light oil')) * mega_to_peta / oil_conversion
    gas_converted = fossils_final.query(parse_dim('seconden_reversed', '3', 'natural gas')) * mega_to_peta / gas_conversion
    
    fossils_final_converteted = pd.concat([coal_converted, heavy_oil_converted, light_oil_converted, gas_converted])
    
    return fossils_final_converteted



#%% Plot primary energy of selected region (in joule or kg)

def plot_stacked_fossils(country_id: int, fossils_primary: pd.DataFrame, unit: str):
    fossils_primary_region = fossils_primary.loc[:, country_id].unstack() # unstacks second index (primary energy types)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    # fossils_primary_region.plot(kind='area', ax=ax)
    
    # define labels via dict
    labels = [get_key(i, DIM2_primary_dict) for i in list(fossils_primary_region.columns)]
    
    # Add labels and title
    ax.stackplot(fossils_primary_region.index, 
                 fossils_primary_region.loc[:, 1], 
                 fossils_primary_region.loc[:, 2], 
                 fossils_primary_region.loc[:, 4],
                 labels = labels)
    
    # .rename(get_key(1, DIM2_primary_dict))
    ax.legend(loc='upper left')
    ax.set_xlabel('Years')
    ax.set_ylabel(f'Fossil energy [{unit}]')
    
    plt.xlim(left=1971, right=2100)
    fig.tight_layout()

#%% plot primary per secondary sankey 

def plot_fossils_sankey(year: int, country_id: int, df1: pd.DataFrame, 
                        df2: pd.DataFrame, DIM1_primsec_dict, DIM2_primsec_dict, 
                        DIM3_seconden_dict, DIM2_sectors_dict, unit: str):

    # Create nodes
    energy_sources = [*DIM1_primsec_dict.values(), *DIM2_primsec_dict.values(),
                      *DIM3_seconden_dict.values(), *DIM2_sectors_dict.values()
                      ]  # * flattens the dict_keys objects
    nodes = create_node_dict(energy_sources)
    
    # Prepare dataframes
    # Get "real" name of energy sources instead of their numbers
    fossils_prim_per_sec_sankey = df1.loc[year, country_id].copy()
    fossils_prim_per_sec_sankey.index = fossils_prim_per_sec_sankey.index.map(index_mapper(
        DIM1_primsec_dict.get,
        DIM2_primsec_dict.get
    ))
    
    fossils_final_energy_sankey = df2.loc[year, country_id].copy()
    fossils_final_energy_sankey.index = fossils_final_energy_sankey.index.map(index_mapper(
        DIM3_seconden_dict.get,
        DIM2_sectors_dict.get
    ))
    
    # Drop where either dimension is "total"
    is_total = fossils_prim_per_sec_sankey.index.get_level_values(0).isin(["total"]) | fossils_prim_per_sec_sankey.index.get_level_values(1).isin(["total"])
    fossils_prim_per_sec_sankey = fossils_prim_per_sec_sankey[~is_total]
    
    is_total = fossils_final_energy_sankey.index.get_level_values(0).isin(["total"]) | fossils_final_energy_sankey.index.get_level_values(1).isin(["total"])
    fossils_final_energy_sankey = fossils_final_energy_sankey[~is_total]
    
    # Convert "real" names to unique node identifiers
    convert_index_to_node_id(fossils_prim_per_sec_sankey, nodes)
    convert_index_to_node_id(fossils_final_energy_sankey, nodes)
    
    # Prepare lists for Sankey diagram
    link_source, link_target, link_value = prepare_Sankey_lists(fossils_prim_per_sec_sankey)
    prepare_Sankey_lists(fossils_final_energy_sankey, link_source, link_target, link_value)
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey( 
        node = dict( 
          thickness = 5, 
          line = dict(color = "green", width = 0.1), 
          label = list(nodes.keys()), 
          color = "blue"
        ), 
        link = dict(
            source = link_source,  
            target = link_target, 
            value = link_value
      ))]) 
    
    # fig.write_image("figures/energy_sankey.jpg")
    fig.write_html(f"{path_figures}/fossils_global_{unit}.html")


def fossil_fuel_data():
    # https://www.engineeringtoolbox.com/fossil-fuels-energy-content-d_1298.html
    # read in and format relevant IMAGE data

    # Total Primary Energy Supply (TPES) in PJ per region by energy carrier, [NRCT, PRIM + 4](t), [28,13](t), # unit: PJ
    primary_energy_supply = read_mym_df(f'{path_scenario_data_fossil}/tpes_ext.out').set_index(["time", "DIM_1"])
    # Primary to Secondary energy flows DIM_1:Primary, DIM_2:Secondary GJ/yr
    prim_per_sec = read_mym_df(f'{path_scenario_data_fossil}/PrimPerSec.out').set_index(["time", "DIM_1", "DIM_2"])
    # Final Energy in PJ/yr by region, sector, and energy carrier [NRCT, S, NECS9T](t), [28,8,10](t) # PJ
    final_energy = read_mym_df(f'{path_scenario_data_fossil}/final_energy_rt.out').set_index(["time", "DIM_1", "DIM_2"])  

    # Total Primary Energy Supply (TPES) in PJ/yr
    # TODO: region 27 has some values? Why? it is not exactly the sum of sth...
    primary_energy_supply = primary_energy_supply.T  # transpose to make DIM_2 the index (DIM_1 is regions, make this columns)
    primary_energy_supply = primary_energy_supply.stack('time')  # add years as index
    primary_energy_supply = primary_energy_supply.swaplevel()  # make years the first index
    primary_energy_supply = primary_energy_supply.sort_index()  # sort dataframe based on the index
    primary_energy_supply.index.names = ['time','DIM_2']

    # Primary to Secondary energy flows in GJ/yr
    prim_per_sec = prim_per_sec/giga_to_peta # GJ to PJ
    prim_per_sec = prim_per_sec.rename(columns={27: 28}) # rename 27 to 28 to fit other dfs
    prim_per_sec[27] = np.nan # create empty dummy 27 region
    prim_per_sec = prim_per_sec[sorted(prim_per_sec.columns)] # sort so that 27 is before 28

    # Final Energy in PJ/yr 
    #TODO: region 27 is empty
    final_energy = final_energy.T # transpose to make DIM_1 the index
    final_energy = final_energy.stack(['time', 'DIM_2'])  # add years as and DIM_2 index
    final_energy.index.names = ['DIM_3', 'time', 'DIM_2']
    final_energy = final_energy.swaplevel(0, 1)  # make years the first index

    #%% Fossil energy only

    fossils_primary = primary_energy_supply.query(parse_dim('primary', '2', 'coal' ,'oil', 
                                                    'natural gas'))
    fossils_prim_per_sec = prim_per_sec.query(parse_dim('primsec_reversed', '1', 'coal', 
                                                'conventional oil', 'unconventional oil', 
                                                'natural gasses'))
    fossils_final = final_energy.query(parse_dim('seconden_reversed', '3', 'coal',
                                            'heavy oil', 'light oil', 'natural gas'))
    
    fossils_primary_converted = converte_primary_energy_to_mass(fossils_primary)
    fossils_primsecond_converted = convert_primary_to_secondary_to_mass(fossils_prim_per_sec)
    fossils_final_converteted = convert_secondary_to_final_mass(fossils_final)

    return (fossils_primary, fossils_prim_per_sec, fossils_final,
            fossils_primary_converted, fossils_primsecond_converted, fossils_final_converteted)



