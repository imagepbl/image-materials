import pandas as pd

from imagematerials.read_mym import read_mym_df
from imagematerials.rest_of.const import path_input_data, SCENARIO

scen = 'SSP2'

def water_consumption(scenario: str = SCENARIO):
    # water conusmption 
    # 1: electricity, 2: industrial, 3: municipality, 4: total
    water_consumption = read_mym_df(path_input_data.joinpath(scenario, 'EnergyFlows/ALLWCS.out')).set_index(["time", "DIM_1"])
    
    # Split up in dimensions
    electricity = water_consumption.xs(key=1, level='DIM_1')
    industrial = water_consumption.xs(key=2, level='DIM_1')
    municipality = water_consumption.xs(key=3, level='DIM_1')

    return electricity, industrial, municipality



def water_withdrawl(scenario: str = SCENARIO):
    water_withdrawel = read_mym_df(path_input_data.joinpath(scenario, 'EnergyFlows/ALLWWD.out')).set_index(["time", "DIM_1"])
    return water_withdrawel
