# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:11:03 2020
@author: Sebastiaan Deetman (deetman@cml.leidenuniv.nl) with contributions from Rombout Huisman 

The purpose of this module is to generate global material demand scenarios for vehicles based on IMAGE model. It can use any set of scenario output files, but initial setup is based on the SSP2 Baseline.

input:  
    1) IMAGE scenario files 
    2) Material & Weight related assumptions
    3) Additional assumptions (translating IMAGE data to the nr of vehicles)

output: 
    1) Graphs on the materials in vehicles
    2) csv database with material use (inflow, stock & outlfow) in kt

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def postprocess(vehicle_materials_stock_passenger, vehicle_materials_stock_freight, vehicle_materials_inflow_passenger,
                vehicle_materials_inflow_freight, vehicle_materials_outflow_passenger, vehicle_materials_outflow_freight, OUTPUT_FOLDER
                ):
    # combine dataframes for output

    # add flow descriptor to the multi-index & fill na values with 0 (the for-loop above didn't cover the battery materials in vhicles without batteries, so these are set to 0 now)
    vehicle_materials_stock_passenger  = pd.concat([vehicle_materials_stock_passenger.fillna(0)],    keys=['stock'],    names=['flow'])   
    vehicle_materials_stock_freight    = pd.concat([vehicle_materials_stock_freight.fillna(0)],      keys=['stock'],    names=['flow'])      
    vehicle_materials_inflow_passenger = pd.concat([vehicle_materials_inflow_passenger.fillna(0)],   keys=['inflow'],   names=['flow'])     
    vehicle_materials_inflow_freight   = pd.concat([vehicle_materials_inflow_freight.fillna(0)],     keys=['inflow'],   names=['flow'])     
    vehicle_materials_outflow_passenger = pd.concat([vehicle_materials_outflow_passenger.fillna(0)], keys=['outflow'],  names=['flow'])      
    vehicle_materials_outflow_freight   = pd.concat([vehicle_materials_outflow_freight.fillna(0)],   keys=['outflow'],  names=['flow']) 

    # concatenate stock, inflow & outflow into 1 dataframe (1 for passenger & 1 for freight)
    vehicle_materials_passenger = pd.concat([vehicle_materials_stock_passenger, vehicle_materials_inflow_passenger,  vehicle_materials_outflow_passenger]) 
    vehicle_materials_freight   = pd.concat([vehicle_materials_stock_freight,   vehicle_materials_inflow_freight,    vehicle_materials_outflow_freight])

    # add category descriptors to the multi-index (pass vs. freight)
    vehicle_materials_passenger = pd.concat([vehicle_materials_passenger], keys=['passenger'], names=['category']) 
    vehicle_materials_freight   = pd.concat([vehicle_materials_freight],    keys=['freight'], names=['category'])  

    vehicle_materials_passenger.columns.name = vehicle_materials_freight.columns.name = 'elements'

    # concatenate into 1 single dataframe & add the 'vehicle' descriptor
    vehicle_materials = pd.concat([vehicle_materials_passenger.stack().unstack(level=2), vehicle_materials_freight.stack().unstack(level=2)])
    vehicle_materials = pd.concat([vehicle_materials], keys=['vehicles'], names=['sector'])

    # re-order multi-index to the desired output & sum to global total
    vehicle_materials_out = vehicle_materials.reorder_levels([3, 2, 0, 1, 6, 5, 4]) / 1000000  # div by 1*10^6 to translate from kg to kt
    vehicle_materials_out.reset_index(inplace=True)                                         # return to columns

    vehicle_materials_out.to_csv(OUTPUT_FOLDER + '\\vehicle_materials_kt.csv', index=False) # in kt

    # ========================================== Visualization


