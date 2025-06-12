import pandas as pd

from imagematerials.read_mym import read_mym_df
from imagematerials.rest_of.const import path_scenario_data_water

scen = 'SSP2'

def water_consumption():
    # water conusmption 
    # 1: electricity, 2: industrial, 3: municipality, 4: total
    water_consumption = read_mym_df(f'{path_scenario_data_water}/ALLWCS.out').set_index(["time", "DIM_1"])
    
    # Split up in dimensions
    electricity = water_consumption.xs(key=1, level='DIM_1')
    industrial = water_consumption.xs(key=2, level='DIM_1')
    municipality = water_consumption.xs(key=3, level='DIM_1')

    return electricity, industrial, municipality



def water_withdrawl():
    water_withdrawel = read_mym_df(f'{path_scenario_data_water}/ALLWWD.out').set_index(["time", "DIM_1"])
    return water_withdrawel
