# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:31:59 2020
@author: S. Deetman - deetman@cml.leidenuniv.nl
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import os
import seaborn as sns     # color palettes
import json

#choices & settings
end_year    = 2060
year_select = list(range(1990,end_year+1))
year_select = [str(i) for i in year_select]
idx = pd.IndexSlice                     # needed for slicing multi-index
region_dict = {'Canada':1, 'US':2, 'Mexico':3, 'Rest C.Am.':4, 'Brazil':5, 'Rest S.Am.':6, 'N.Africa':7, 'W.Africa':8, 'E.Africa':9, 'South Africa':10, 'W.Europe':11, 'C.Europe':12, 'Turkey':13, 'Ukraine':14, 'Stan':15, 'Russia':16, 'M.East':17, 'India':18, 'Korea':19, 'China':20, 'SE.Asia':21, 'Indonesia':22, 'Japan':23, 'Oceania':24, 'Rest S.Asia':25, 'Rest S.Africa':26 }
region_colorset = ['#59250c','#d45b26','#f0c22b']
regions_steady   = [1,2,3,11,12,13,14,15,16,17,24]
regions_fastdv   = [4,5,6,7,8,9,10,18,19,21,22,25,26]
regions_chinjp   = [20,23]

os.chdir("C:\\Users\\Admin\\surfdrive\\Projects\\IRP\\GRO23\\Modelling\\2060\\")

#scenario selection
ssp = "SSP2"
variant = "15D"
scenario_variant = ssp + "_" + variant + "\\"

folder_buildings       = "BUMA\\output\\SSP2_CP\\"
folder_electricity_BL  = "ELMA\\output\\" + scenario_variant + "\\default\\"
folder_vehicles_BL     = "VEMA\\output\\" + scenario_variant

# Read output files on drivers (nr of vehicles, MW of generation & m2 of buildings), global results
vehicle_nr_BL      = pd.read_csv(folder_vehicles_BL +    'global_vehicle_nr.csv', index_col=0)
electricity_km_BL  = pd.read_csv(folder_electricity_BL + 'grid_length_HV_km.csv', index_col=[0,1])  # since the HV grid length growth is based on the growth of the generation capacity, we take it as the growth indicator here 
buildings_m2       = pd.read_csv(folder_buildings +      'sqmeters_output.csv',   index_col=[0,1,2,3])

# Read output files on materials
vehicle_mat_BL            = pd.read_csv(folder_vehicles_BL +    'vehicle_materials_kt.csv'    , index_col=[0,1,2,3,4,5,6])
electricity_stor_mat_BL   = pd.read_csv(folder_electricity_BL + 'stor_materials_output_kt.csv', index_col=[0,1,2,3,4,5])
electricity_gcap_mat_BL   = pd.read_csv(folder_electricity_BL + 'gcap_materials_output_kt.csv', index_col=[0,1,2,3,4,5])
electricity_grid_mat_BL   = pd.read_csv(folder_electricity_BL + 'grid_materials_output_kt.csv', index_col=[0,1,2,3,4,5])
buildings_mat_BL          = pd.read_csv(folder_buildings +      'material_output.csv'         , index_col=[0,1,2,3,4])

# Read external plot data (SSP2 population & current production in kt)
current_production_steel     = pd.read_csv('data\\current_production_steel.csv', index_col=0)     # kt
current_production_copper    = pd.read_csv('data\\current_production_copper.csv', index_col=0)    # kt
current_production_aluminium = pd.read_csv('data\\current_production_aluminium.csv', index_col=0) # kt
current_production_concrete  = pd.read_csv('data\\current_production_concrete.csv', index_col=0) # kt
population                   = pd.read_csv('data\\pop.csv', index_col=0) # kt

population_steady = population.loc[:, map(str, regions_steady)].sum(axis=1)
population_fastdv = population.loc[:, map(str, regions_fastdv)].sum(axis=1)
population_chinjp = population.loc[:, map(str, regions_chinjp)].sum(axis=1)

#%% Pre processing
# re-arrange & sum to required aggregation levels (total global)

# DRIVERS
freight_nr_BL     = vehicle_nr_BL[['Trucks','Cargo Trains','Ships','Cargo Planes']].sum(axis=1)
passenger_nr_BL   = vehicle_nr_BL[['Buses','Trains','Cars','Planes','Bikes']].sum(axis=1)
electricity_km_BL = electricity_km_BL.sum(axis=1).sum(axis=0, level=1)
buildings_m2_comm = buildings_m2.sum(level=[1,2]).loc[idx['stock',['office','retail','hotels','govern']],[str(i) for i in range(1971,end_year+1)]].sum(level=0).transpose()
buildings_m2_urb  = buildings_m2.sum(level=[1,2,3]).loc[idx['stock',['detached','semi-detached','appartments','high-rise'],'urban'],[str(i) for i in range(1971,end_year+1)]].sum(level=0).transpose()
buildings_m2_rur  = buildings_m2.sum(level=[1,2,3]).loc[idx['stock',['detached','semi-detached','appartments','high-rise'],'rural'],[str(i) for i in range(1971,end_year+1)]].sum(level=0).transpose()

index_year = 2010
freight_index_BL     = freight_nr_BL.div(freight_nr_BL.loc[index_year])
passenger_index_BL   = passenger_nr_BL.div(passenger_nr_BL.loc[index_year])
electricity_index_BL = electricity_km_BL.div(electricity_km_BL.loc[index_year])
buildings_index_comm = buildings_m2_comm.div(buildings_m2_comm.loc[str(index_year)])
buildings_index_urb  = buildings_m2_urb.div(buildings_m2_urb.loc[str(index_year)])
buildings_index_rur  = buildings_m2_rur.div(buildings_m2_rur.loc[str(index_year)])
pop_steady_index     = population_steady.div(population_steady[index_year])
pop_fastdv_index     = population_fastdv.div(population_fastdv[index_year])
pop_chinjp_index     = population_chinjp.div(population_chinjp[index_year])

#MATERIALS (selection from 1990)
vehicle_mat_BL_tot           = vehicle_mat_BL.sum(level=[1,3,5])
electricity_stor_mat_BL_tot  = electricity_stor_mat_BL.sum(level=[1,5])
electricity_gcap_mat_BL_tot  = electricity_gcap_mat_BL.sum(level=[1,5])
electricity_grid_mat_BL_tot  = electricity_grid_mat_BL.sum(level=[1,5])
electricity_mat_BL_tot       = electricity_stor_mat_BL_tot[[str(i) for i in range(1990,end_year+1)]] + electricity_gcap_mat_BL_tot[[str(i) for i in range(1990,end_year+1)]] + electricity_grid_mat_BL_tot[[str(i) for i in range(1990,end_year+1)]]
electricity_mat_BL_tot       = electricity_mat_BL_tot.dropna(axis=0, how='all')       # only the materials covered in all 3 sub-sectors are selected (otherwise they would have been a NaN value)
buildings_mat_BL_tot         = buildings_mat_BL.sum(level=[1,3,4])[[str(i) for i in range(1971,end_year+1)]]

#MATERIALS REGIONAL (selection from 1990)
vehicle_mat_BL_reg           = vehicle_mat_BL.sum(level=[0,1,3,5])
electricity_stor_mat_BL_reg  = electricity_stor_mat_BL.sum(level=[0,1,5])
electricity_gcap_mat_BL_reg  = electricity_gcap_mat_BL.sum(level=[0,1,5])
electricity_grid_mat_BL_reg  = electricity_grid_mat_BL.sum(level=[0,1,5])
electricity_mat_BL_reg       = electricity_stor_mat_BL_reg[[str(i) for i in range(1990,end_year+1)]] + electricity_gcap_mat_BL_reg[[str(i) for i in range(1990,end_year+1)]] + electricity_grid_mat_BL_reg[[str(i) for i in range(1990,end_year+1)]]
electricity_mat_BL_reg       = electricity_mat_BL_reg.dropna(axis=0, how='all')       # only the materials covered in all 3 sub-sectors are selected (otherwise they would have been a NaN value)
buildings_mat_BL_reg         = buildings_mat_BL.sum(level=[0,1,3,4])[[str(i) for i in range(1971,end_year+1)]]

buildings_mat_BL_reg_in      = buildings_mat_BL_reg.loc[idx[:,'inflow',:,:], year_select].sum(axis=0, level=[0,3])    # all inflow by region
buildings_mat_BL_reg_out     = buildings_mat_BL_reg.loc[idx[:,'outflow',:,:], year_select].sum(axis=0, level=[0,3])   # all outflow by region

vehicle_mat_BL_reg_in        = vehicle_mat_BL_reg.loc[idx[:,'inflow',:,:], year_select].sum(axis=0, level=[0,3])      #
vehicle_mat_BL_reg_out       = vehicle_mat_BL_reg.loc[idx[:,'outflow',:,:], year_select].sum(axis=0, level=[0,3])     #

electricity_mat_BL_reg_in    = electricity_mat_BL_reg.loc[idx[:,'inflow',:], year_select].sum(axis=0, level=[0,2])    # all inflow by region
electricity_mat_BL_reg_out   = electricity_mat_BL_reg.loc[idx[:,'outflow',:], year_select].sum(axis=0, level=[0,2])   # all inflow by region
electricity_mat_BL_reg_in    = electricity_mat_BL_reg_in.rename(region_dict, axis='index', level=0)
electricity_mat_BL_reg_out   = electricity_mat_BL_reg_out.rename(region_dict, axis='index', level=0)

#%%  Panel on DRIVERS (indexed) & POPULATION (regional)

years_select_str = list([str(i) for i in range(2000,end_year+1)])
years_select     = list(range(2000,end_year+1))

years_plot2      = sorted(list(set(years_select) & set(list(population.index)))) # match between available years and the selected period

color_set = ["#d5573b","#b05f63","#45873d","#9bc7e8","#407c9c","#1e6185",] 
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(20,8), frameon=True)

ax2.margins(0)
ax2.set_ylim(0,400)
ax2.plot(years_select, freight_index_BL.loc[years_select] * 100, label='Freight Vehicles (number)',linewidth=3, color=color_set[0])
ax2.plot(years_select, passenger_index_BL.loc[years_select] * 100, label='Passenger Vehicles (number)',linewidth=3, color=color_set[1])
ax2.plot(years_select, electricity_index_BL.loc[years_select] * 100, label='Electricity Infrastructure (km HV grid)',linewidth=3, color=color_set[2])
ax2.plot(years_select, buildings_index_comm.loc[years_select_str] * 100, label='Commercial Buildings (square meters)',linewidth=3, color=color_set[3])
ax2.plot(years_select, buildings_index_urb.loc[years_select_str] * 100, label='Urban Housing (square meters)',linewidth=3, color=color_set[4])
ax2.plot(years_select, buildings_index_rur.loc[years_select_str] * 100, label='Rural Housing (square meters)',linewidth=3, color=color_set[5])
handles, labels = ax2.get_legend_handles_labels()
ax2.set_ylabel('Indexed growth of stock (2010=100)', fontsize=14).set_rotation(90)
ax2.yaxis.set_label_coords(-0.07,0.55)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_title('Indexed Growth of Global Stocks, SSP2 Baseline (2010=100)', fontsize = 15)
ax2.legend(reversed(handles), reversed(labels), loc='upper left', ncol=1, frameon=False, fontsize=14)

ax1.set_title('Indexed Growth of Population (grouped), SSP2 Baseline (2010=100)', fontsize = 15)
ax1.set_ylabel('Indexed growth of population (2010=100)', fontsize=14).set_rotation(90)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.plot(years_plot2, pop_steady_index[years_plot2] * 100, label='Steady Developed', linewidth=3, color=region_colorset[0])
ax1.plot(years_plot2, pop_fastdv_index[years_plot2] * 100, label='Fast Developing',  linewidth=3, color=region_colorset[1])
ax1.plot(years_plot2, pop_chinjp_index[years_plot2] * 100, label='China & Japan',    linewidth=3, color=region_colorset[2])
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(reversed(handles), reversed(labels), loc='upper left', ncol=1, frameon=False, fontsize=14)

plt.savefig('output\\' + scenario_variant + 'indexed_drivers_population_panel.png', dpi=600)
plt.show()

#%%  Line plot on Population (Absolute)
years_select_str = list([str(i) for i in range(2000,end_year+1)])
years_select     = list(range(2000,end_year+1))

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1, 1, 1)
ax.margins(0)
ax.set_ylim(0,7000)
plt.plot(years_plot2, population_steady[years_plot2], label='Steady Developed',linewidth=3, color=region_colorset[0])
plt.plot(years_plot2, population_fastdv[years_plot2], label='Fast Developing', linewidth=3, color=region_colorset[1])
plt.plot(years_plot2, population_chinjp[years_plot2], label='China & Japan',   linewidth=3, color=region_colorset[2])
handles, labels = ax.get_legend_handles_labels()
plt.ylabel('Global population in 3 regional groups', fontsize=14).set_rotation(90)
ax.yaxis.set_label_coords(-0.07,0.55)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.title('Population (Millions), SSP2 Baseline', fontsize = 15)
ax.legend(reversed(handles), reversed(labels), loc='upper left', ncol=1, frameon=False, fontsize=16)
plt.savefig('output\\' + scenario_variant + 'population_growth_SSP2_3reg_abs.png', dpi=600)
plt.show()
   
#%% 6 graphs pannel stock, inflow + current production

years_select_str1 = list([str(i) for i in range(1996,end_year+1)])
years_select_str2 = list([str(i) for i in range(2000,end_year+1)])

mat_select1 = 'Steel'
mat_select2 = 'Aluminium'
mat_select3 = 'copper'
mat_select3b = 'Cu'

graph1_data       = [[],[],[]]
graph1_data[0]    = buildings_mat_BL_tot.loc[idx['stock',:,str.lower(mat_select1)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph1_data[1]    = vehicle_mat_BL_tot.loc[idx['stock',:,mat_select1], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph1_data[2]    = electricity_mat_BL_tot.loc[idx['stock',mat_select1], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000

graph2_data       = [[],[],[]]
graph2_data[0]    = buildings_mat_BL_tot.loc[idx['stock',:,str.lower(mat_select2)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph2_data[1]    = vehicle_mat_BL_tot.loc[idx['stock',:,mat_select2], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph2_data[2]    = electricity_mat_BL_tot.loc[idx['stock',mat_select2], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000

graph3_data       = [[],[],[]]
graph3_data[0]    = buildings_mat_BL_tot.loc[idx['stock',:,str.lower(mat_select3)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph3_data[1]    = vehicle_mat_BL_tot.loc[idx['stock',:,mat_select3b], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph3_data[2]    = electricity_mat_BL_tot.loc[idx['stock',mat_select3b], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000

graph4_data       = [[],[],[]]
graph4_data[0]    = buildings_mat_BL_tot.loc[idx['inflow',:,str.lower(mat_select1)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph4_data[1]    = vehicle_mat_BL_tot.loc[idx['inflow',:,mat_select1], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph4_data[2]    = electricity_mat_BL_tot.loc[idx['inflow',mat_select1], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph4_sum        = graph4_data[0]+graph4_data[1]+graph4_data[2]

graph5_data       = [[],[],[]]
graph5_data[0]    = buildings_mat_BL_tot.loc[idx['inflow',:,str.lower(mat_select2)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph5_data[1]    = vehicle_mat_BL_tot.loc[idx['inflow',:,mat_select2], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph5_data[2]    = electricity_mat_BL_tot.loc[idx['inflow',mat_select2], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph5_sum        = graph5_data[0]+graph5_data[1]+graph5_data[2]

graph6_data       = [[],[],[]]
graph6_data[0]    = buildings_mat_BL_tot.loc[idx['inflow',:,str.lower(mat_select3)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph6_data[1]    = vehicle_mat_BL_tot.loc[idx['inflow',:,mat_select3b], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph6_data[2]    = electricity_mat_BL_tot.loc[idx['inflow',mat_select3b], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph6_sum        = graph6_data[0]+graph6_data[1]+graph6_data[2]

y_fixed = [0, 35000, 2250, 1200, 1200, 120, 50]
color_set = ["#3d4a40", "#5a8a64", "#89c496"]

plt.figure()
fig, ((ax1, ax4),(ax2, ax5),(ax3, ax6)) = plt.subplots(3, 2, figsize=(17,20), frameon=True)
fig.suptitle('Global Material Stocks & Flows in 3 categories', y=0.92, fontsize=16)
plt.subplots_adjust(wspace = 0.25, bottom = 0.05)

ax1.set_title('Total Global Steel Stock', fontsize=14)
ax1.set_ylabel('Total Steel Stock (Mt)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.margins(x=0)
ax1.set_ylim(0,y_fixed[1])
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.stackplot(years_select, graph1_data, labels=['Buildings','Vehicles','Electricity Infrastructure'], colors=color_set)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

ax2.set_title('Total Global Aluminium Stock', fontsize=14)
ax2.set_ylabel('Total Aluminium Stock (Mt)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
ax2.margins(x=0)
ax2.set_ylim(0,y_fixed[2])
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.stackplot(years_select, graph2_data, labels=['Buildings','Vehicles','Electricity Infrastructure'], colors=color_set)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

ax3.set_title('Total Global Copper Stock', fontsize=14)
ax3.set_ylabel('Total Copper Stock (Mt)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')
ax3.margins(x=0)
ax3.set_ylim(0,y_fixed[3])
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.stackplot(years_select, graph3_data, labels=['Buildings','Vehicles','Electricity Infrastructure'], colors=color_set)
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

ax4.set_title('Global Annual Steel Demand', fontsize=14)
ax4.set_ylabel('Annual Steel Demand (Mt/yr)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax4.yaxis.set_ticks_position('left')
ax4.xaxis.set_ticks_position('bottom')
ax4.margins(x=0)
ax4.set_ylim(0,y_fixed[4])
#ax4.set_ylim(0,max(graph4_sum[-1],max(list(current_production_steel.values[0]/1000)))) #-1 presumes that the last year is the max, which is not always the case
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.stackplot(years_select, graph4_data, labels=['_nolegend_','_nolegend_','_nolegend_'], colors=color_set)
#ax4.plot(list(map(int, current_production_steel.columns)), list(current_production_steel.values[0]/1000), '.', markersize=14, color='black', label='Total production')
handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles, labels, bbox_to_anchor=(1, 1), loc=1, ncol=1, frameon=False, fontsize=13)

ax5.set_title('Global Annual Aluminium Demand', fontsize=14)
ax5.set_ylabel('Annual Aluminium Demand (Mt/yr)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax5.yaxis.set_ticks_position('left')
ax5.xaxis.set_ticks_position('bottom')
ax5.margins(x=0)
ax5.set_ylim(0,y_fixed[5])
#ax5.set_ylim(0,max(graph5_sum[-1],max(list(current_production_aluminium.values[0]/1000)))) #-1 presumes that the last year is the max, which is not always the case
ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.stackplot(years_select, graph5_data, labels=['_nolegend_','_nolegend_','_nolegend_'], colors=color_set)
ax5.plot(list(map(int, current_production_aluminium.columns)), list(current_production_aluminium.values[0]/1000), '.',  markersize=14, color='black', label='Total production')
handles, labels = ax5.get_legend_handles_labels()
ax5.legend(handles, labels, bbox_to_anchor=(1, 1), loc=1, ncol=1, frameon=False, fontsize=13)

ax6.set_title('Global Annual Copper Demand', fontsize=14)
ax6.set_ylabel('Annual Copper Demand (Mt/yr)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax6.yaxis.set_ticks_position('left')
ax6.xaxis.set_ticks_position('bottom')
ax6.margins(x=0)
ax6.set_ylim(0,y_fixed[6])
#ax6.set_ylim(0,max(graph6_sum[-1],max(list(current_production_copper.values[0]/1000)))) #-1 presumes that the last year is the max, which is not always the case
ax6.tick_params(axis='both', which='major', labelsize=14)
ax6.stackplot(years_select, graph6_data, labels=['_nolegend_','_nolegend_','_nolegend_'], colors=color_set)
ax6.plot(list(map(int, current_production_copper.columns)), list(current_production_copper.values[0]/1000), '.', markersize=14, color='black', label='Total production')
handles, labels = ax6.get_legend_handles_labels()
ax6.legend(handles, labels, bbox_to_anchor=(1, 1), loc=1, ncol=1, frameon=False, fontsize=13)

plt.savefig('output\\' + scenario_variant + 'overview_panel_new_IRP3.png', dpi=600)
plt.show()

#%% intermediate output for comparisson
year_compare = ['2017','2018','2019']
outvar_compare = pd.DataFrame(index=['Steel','Aluminium','Copper'], columns=['Buildings','Electricity','Vehicles'])
outvar_compare.loc['Steel','Buildings']   = buildings_mat_BL_tot.loc[idx['inflow',:,str.lower('Steel')], year_compare].sum(axis=0).mean()
outvar_compare.loc['Steel','Electricity'] = electricity_mat_BL_tot.loc[idx['inflow','Steel'], year_compare].mean()
outvar_compare.loc['Steel','Vehicles']    = vehicle_mat_BL_tot.loc[idx['inflow',:,'Steel'], year_compare].sum(axis=0).mean()
outvar_compare.loc['Aluminium','Buildings']   = buildings_mat_BL_tot.loc[idx['inflow',:,str.lower('Aluminium')], year_compare].sum(axis=0).mean()
outvar_compare.loc['Aluminium','Electricity'] = electricity_mat_BL_tot.loc[idx['inflow','Aluminium'], year_compare].mean()
outvar_compare.loc['Aluminium','Vehicles']    = vehicle_mat_BL_tot.loc[idx['inflow',:,'Aluminium'], year_compare].sum(axis=0).mean()
outvar_compare.loc['Copper','Buildings']   = buildings_mat_BL_tot.loc[idx['inflow',:,str.lower('Copper')], year_compare].sum(axis=0).mean()
outvar_compare.loc['Copper','Electricity'] = electricity_mat_BL_tot.loc[idx['inflow','Cu'], year_compare].mean()
outvar_compare.loc['Copper','Vehicles']    = vehicle_mat_BL_tot.loc[idx['inflow',:,'Cu'], year_compare].sum(axis=0).mean()


#%% 6 graphs pannel outflow, inflow + regional detail

graph1x_inflow        = graph4_sum   # material 1, Steel       Total inflow
graph2x_inflow        = graph5_sum   # material 2, Aluminium   Total inflow
graph3x_inflow        = graph6_sum   # material 3, Copper      Total inflow

graph1_outflow        = [[],[],[]]
graph1_outflow[0]     = buildings_mat_BL_tot.loc[idx['outflow',:,str.lower(mat_select1)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph1_outflow[1]     = vehicle_mat_BL_tot.loc[idx['outflow',:,mat_select1], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph1_outflow[2]     = electricity_mat_BL_tot.loc[idx['outflow',mat_select1], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph1_outflow_sum    = graph1_outflow[0] + graph1_outflow[1] + graph1_outflow[2]

graph2_outflow        = [[],[],[]]
graph2_outflow[0]     = buildings_mat_BL_tot.loc[idx['outflow',:,str.lower(mat_select2)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph2_outflow[1]     = vehicle_mat_BL_tot.loc[idx['outflow',:,mat_select2], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph2_outflow[2]     = electricity_mat_BL_tot.loc[idx['outflow',mat_select2], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph2_outflow_sum    = graph2_outflow[0] + graph2_outflow[1] + graph2_outflow[2]

graph3_outflow        = [[],[],[]]
graph3_outflow[0]     = buildings_mat_BL_tot.loc[idx['outflow',:,str.lower(mat_select3)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph3_outflow[1]     = vehicle_mat_BL_tot.loc[idx['outflow',:,mat_select3b], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph3_outflow[2]     = electricity_mat_BL_tot.loc[idx['outflow',mat_select3b], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
graph3_outflow_sum    = graph3_outflow[0] + graph3_outflow[1] + graph3_outflow[2]

# intermediate output (synthesis text)
print(1 - (graph1_outflow_sum[-5:].sum()/5) / (graph1x_inflow[-5:].sum()/5))
print(1 - (graph2_outflow_sum[-5:].sum()/5) / (graph2x_inflow[-5:].sum()/5))
print(1 - (graph3_outflow_sum[-5:].sum()/5) / (graph3x_inflow[-5:].sum()/5))

graph4_inflow_steady  = buildings_mat_BL_reg_in.loc[idx[regions_steady, str.lower(mat_select1)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + vehicle_mat_BL_reg_in.loc[idx[regions_steady, mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + electricity_mat_BL_reg_in.loc[idx[regions_steady, mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph4_inflow_fastdv  = buildings_mat_BL_reg_in.loc[idx[regions_fastdv, str.lower(mat_select1)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + vehicle_mat_BL_reg_in.loc[idx[regions_fastdv ,mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + electricity_mat_BL_reg_in.loc[idx[regions_fastdv, mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph4_inflow_chinjp  = buildings_mat_BL_reg_in.loc[idx[regions_chinjp, str.lower(mat_select1)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + vehicle_mat_BL_reg_in.loc[idx[regions_chinjp ,mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + electricity_mat_BL_reg_in.loc[idx[regions_chinjp, mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]

graph4_outflow_steady = buildings_mat_BL_reg_out.loc[idx[regions_steady, str.lower(mat_select1)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + vehicle_mat_BL_reg_out.loc[idx[regions_steady, mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + electricity_mat_BL_reg_out.loc[idx[regions_steady, mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph4_outflow_fastdv = buildings_mat_BL_reg_out.loc[idx[regions_fastdv, str.lower(mat_select1)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + vehicle_mat_BL_reg_out.loc[idx[regions_fastdv ,mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + electricity_mat_BL_reg_out.loc[idx[regions_fastdv, mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph4_outflow_chinjp = buildings_mat_BL_reg_out.loc[idx[regions_chinjp, str.lower(mat_select1)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + vehicle_mat_BL_reg_out.loc[idx[regions_chinjp ,mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + electricity_mat_BL_reg_out.loc[idx[regions_chinjp, mat_select1],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]

graph5_inflow_steady  = buildings_mat_BL_reg_in.loc[idx[regions_steady, str.lower(mat_select2)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + vehicle_mat_BL_reg_in.loc[idx[regions_steady, mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + electricity_mat_BL_reg_in.loc[idx[regions_steady, mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph5_inflow_fastdv  = buildings_mat_BL_reg_in.loc[idx[regions_fastdv, str.lower(mat_select2)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + vehicle_mat_BL_reg_in.loc[idx[regions_fastdv ,mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + electricity_mat_BL_reg_in.loc[idx[regions_fastdv, mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph5_inflow_chinjp  = buildings_mat_BL_reg_in.loc[idx[regions_chinjp, str.lower(mat_select2)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + vehicle_mat_BL_reg_in.loc[idx[regions_chinjp ,mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + electricity_mat_BL_reg_in.loc[idx[regions_chinjp, mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]

graph5_outflow_steady = buildings_mat_BL_reg_out.loc[idx[regions_steady, str.lower(mat_select2)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + vehicle_mat_BL_reg_out.loc[idx[regions_steady, mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + electricity_mat_BL_reg_out.loc[idx[regions_steady, mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph5_outflow_fastdv = buildings_mat_BL_reg_out.loc[idx[regions_fastdv, str.lower(mat_select2)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + vehicle_mat_BL_reg_out.loc[idx[regions_fastdv ,mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + electricity_mat_BL_reg_out.loc[idx[regions_fastdv, mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph5_outflow_chinjp = buildings_mat_BL_reg_out.loc[idx[regions_chinjp, str.lower(mat_select2)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + vehicle_mat_BL_reg_out.loc[idx[regions_chinjp ,mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + electricity_mat_BL_reg_out.loc[idx[regions_chinjp, mat_select2],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]

graph6_inflow_steady  = buildings_mat_BL_reg_in.loc[idx[regions_steady, str.lower(mat_select3)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + vehicle_mat_BL_reg_in.loc[idx[regions_steady, mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + electricity_mat_BL_reg_in.loc[idx[regions_steady, mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph6_inflow_fastdv  = buildings_mat_BL_reg_in.loc[idx[regions_fastdv, str.lower(mat_select3)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + vehicle_mat_BL_reg_in.loc[idx[regions_fastdv ,mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + electricity_mat_BL_reg_in.loc[idx[regions_fastdv, mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph6_inflow_chinjp  = buildings_mat_BL_reg_in.loc[idx[regions_chinjp, str.lower(mat_select3)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + vehicle_mat_BL_reg_in.loc[idx[regions_chinjp ,mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]  + electricity_mat_BL_reg_in.loc[idx[regions_chinjp, mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]

graph6_outflow_steady = buildings_mat_BL_reg_out.loc[idx[regions_steady, str.lower(mat_select3)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + vehicle_mat_BL_reg_out.loc[idx[regions_steady, mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + electricity_mat_BL_reg_out.loc[idx[regions_steady, mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph6_outflow_fastdv = buildings_mat_BL_reg_out.loc[idx[regions_fastdv, str.lower(mat_select3)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + vehicle_mat_BL_reg_out.loc[idx[regions_fastdv ,mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + electricity_mat_BL_reg_out.loc[idx[regions_fastdv, mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]
graph6_outflow_chinjp = buildings_mat_BL_reg_out.loc[idx[regions_chinjp, str.lower(mat_select3)],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + vehicle_mat_BL_reg_out.loc[idx[regions_chinjp ,mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2] + electricity_mat_BL_reg_out.loc[idx[regions_chinjp, mat_select3b],:].sum(axis=0).rolling(window=5).mean()[years_select_str2]

plt.figure()
fig, ((ax1, ax4),(ax2, ax5),(ax3, ax6)) = plt.subplots(3, 2, figsize=(17,20), frameon=True)
fig.suptitle('Global & Regional Material Inflow & Outlow', y=0.92, fontsize=16)
plt.subplots_adjust(wspace = 0.25, bottom = 0.05)

ax1.set_title('Global Steel Outflows (stacked) and Inflows (line)', fontsize=14)
ax1.set_ylabel('Total Steel Flows (Mt/yr)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')
ax1.margins(x=0)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.stackplot(years_select, graph1_outflow, labels=['Buildings (outflow)','Vehicles (outflow)','Electricity Infrastructure (outflow)'], colors=color_set)
#ax1.plot(years_select, graph1_outflow_sum, '--', color='red')
ax1.plot(years_select, graph1x_inflow,     '-', color='green', label='total inflow')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

ax2.set_title('Global Aluminium Outflows (stacked) and Inflows (line)', fontsize=14)
ax2.set_ylabel('Total Aluminium Stock (Mt/yr)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')
ax2.margins(x=0)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.stackplot(years_select, graph2_outflow, labels=['Buildings (outflow)','Vehicles (outflow)','Electricity Infrastructure (outflow)'], colors=color_set)
#ax2.plot(years_select, graph2_outflow_sum, '--', color='red')
ax2.plot(years_select, graph2x_inflow,     '-', color='green', label='total inflow')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

ax3.set_title('Global Copper Outflows (stacked) and Inflows (line)', fontsize=14)
ax3.set_ylabel('Total Copper Stock (Mt/yr)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax3.yaxis.set_ticks_position('left')
ax3.xaxis.set_ticks_position('bottom')
ax3.margins(x=0)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.stackplot(years_select, graph3_outflow, labels=['Buildings (outflow)','Vehicles (outflow)','Electricity Infrastructure (outflow)'], colors=color_set)
#ax3.plot(years_select, graph3_outflow_sum, '--', color='red')
ax3.plot(years_select, graph3x_inflow,     '-', color='green', label='total inflow')
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

ax4.set_title('Regional Steel Inflows and Outflows', fontsize=14)
ax4.set_ylabel('Annual Steel Flow (Mt/yr)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax4.yaxis.set_ticks_position('left')
ax4.xaxis.set_ticks_position('bottom')
ax4.margins(x=0)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.plot(years_select, graph4_inflow_steady/1000,  '-',  linewidth=2, color=region_colorset[0], label='Developed (inflow)')
ax4.plot(years_select, graph4_outflow_steady/1000, '--', linewidth=2, color=region_colorset[0], label='Developed (outflow)')
ax4.plot(years_select, graph4_inflow_fastdv/1000,  '-',  linewidth=2, color=region_colorset[1], label='Developing (inflow)')
ax4.plot(years_select, graph4_outflow_fastdv/1000, '--', linewidth=2, color=region_colorset[1], label='Developing (outflow)')
ax4.plot(years_select, graph4_inflow_chinjp/1000,  '-',  linewidth=2, color=region_colorset[2], label='China+Japan (inflow)')
ax4.plot(years_select, graph4_outflow_chinjp/1000, '--', linewidth=2, color=region_colorset[2], label='China+Japan (outflow)')
handles, labels = ax4.get_legend_handles_labels()
#ax4.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

ax5.set_title('Regional Aluminium Inflows and Outflows', fontsize=14)
ax5.set_ylabel('Annual Aluminium Flow (Mt/yr)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax5.yaxis.set_ticks_position('left')
ax5.xaxis.set_ticks_position('bottom')
ax5.margins(x=0)
ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.plot(years_select, graph5_inflow_steady/1000,  '-',  linewidth=2, color=region_colorset[0],  label='Developed (inflow)')
ax5.plot(years_select, graph5_outflow_steady/1000, '--', linewidth=2, color=region_colorset[0],  label='Developed (outflow)')
ax5.plot(years_select, graph5_inflow_fastdv/1000,  '-',  linewidth=2, color=region_colorset[1],  label='Developing (inflow)')
ax5.plot(years_select, graph5_outflow_fastdv/1000, '--', linewidth=2, color=region_colorset[1],  label='Developing (outflow)')
ax5.plot(years_select, graph5_inflow_chinjp/1000,  '-',  linewidth=2, color=region_colorset[2],  label='China+Japan (inflow)')
ax5.plot(years_select, graph5_outflow_chinjp/1000, '--', linewidth=2, color=region_colorset[2],  label='China+Japan (outflow)')
handles, labels = ax5.get_legend_handles_labels()
#ax5.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

ax6.set_title('Regional Copper Inflows and Outflows', fontsize=14)
ax6.set_ylabel('Annual Copper Flow (Mt/yr)', rotation='vertical', x=-0.1, y=0.5, fontsize=13)
ax6.yaxis.set_ticks_position('left')
ax6.xaxis.set_ticks_position('bottom')
ax6.margins(x=0)
ax6.tick_params(axis='both', which='major', labelsize=14)
ax6.plot(years_select, graph6_inflow_steady/1000,  '-',  linewidth=2, color=region_colorset[0],  label='Developed (inflow)')
ax6.plot(years_select, graph6_outflow_steady/1000, '--', linewidth=2, color=region_colorset[0],  label='Developed (outflow)')
ax6.plot(years_select, graph6_inflow_fastdv/1000,  '-',  linewidth=2, color=region_colorset[1],  label='Developing (inflow)')
ax6.plot(years_select, graph6_outflow_fastdv/1000, '--', linewidth=2, color=region_colorset[1],  label='Developing (outflow)')
ax6.plot(years_select, graph6_inflow_chinjp/1000,  '-',  linewidth=2, color=region_colorset[2],  label='China+Japan (inflow)')
ax6.plot(years_select, graph6_outflow_chinjp/1000, '--', linewidth=2, color=region_colorset[2],  label='China+Japan (outflow)')
handles, labels = ax6.get_legend_handles_labels()
ax6.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

plt.show()
plt.savefig('output\\' + scenario_variant + 'overview_panel_in_out.png', dpi=600)

#%% Indictaors for PBL presentation
steel_demand      = pd.DataFrame(index=list(map(int, years_select_str2)), columns = ['Buildings','Vehicles','Electricity','Steady','Developing','China&Japan'])
steel_demand['Buildings']    = buildings_mat_BL_tot.loc[idx['inflow',:,str.lower(mat_select1)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
steel_demand['Vehicles']     = vehicle_mat_BL_tot.loc[idx['inflow',:,mat_select1], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
steel_demand['Electricity']  = electricity_mat_BL_tot.loc[idx['inflow',mat_select1], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
steel_demand['Steady']       = graph4_inflow_steady.to_numpy() / 1000
steel_demand['Developing']   = graph4_inflow_fastdv.to_numpy() / 1000
steel_demand['China&Japan']  = graph4_inflow_chinjp.to_numpy() / 1000

concr_demand      = pd.DataFrame(index=list(map(int, years_select_str2)), columns = ['Buildings','Electricity','Steady','Developing','China&Japan'])
concr_demand['Buildings']   = buildings_mat_BL_tot.loc[idx['inflow',:,str.lower(mat_select2)], years_select_str1].sum(axis=0).rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
concr_demand['Electricity'] = electricity_mat_BL_tot.loc[idx['inflow',mat_select2], years_select_str1].rolling(window=5).mean()[years_select_str2].to_numpy() / 1000
concr_demand['Steady']       = graph5_inflow_steady.to_numpy() / 1000
concr_demand['Developing']   = graph5_inflow_fastdv.to_numpy() / 1000
concr_demand['China&Japan']  = graph5_inflow_chinjp.to_numpy() / 1000

steel_demand.to_csv('output\\' + scenario_variant + 'steel_global_demand_Mtperyear.csv') 
concr_demand.to_csv('output\\' + scenario_variant + 'concr_global_demand_Mtperyear.csv') 

#%% Separate indicators for main text

# Growth indicators for the main text
steel_growth = graph4_sum[-5:].sum()/graph4_sum[15:20].sum()
alumi_growth = graph5_sum[-5:].sum()/graph5_sum[15:20].sum()
coppr_growth = graph6_sum[-5:].sum()/graph6_sum[15:20].sum()

current_share_steel_build = graph4_data[0][14:18].sum()/(current_production_steel.values[0][14:18].sum()/1000)
current_share_steel_vehic = graph4_data[1][14:18].sum()/(current_production_steel.values[0][14:18].sum()/1000)
current_share_steel_elect = graph4_data[2][14:18].sum()/(current_production_steel.values[0][14:18].sum()/1000)
current_share_steel_total = graph4_sum[14:18].sum()/(current_production_steel.values[0][14:18].sum()/1000)

current_share_alumi_build = graph5_data[0][14:18].sum()/(current_production_aluminium.values[0][14:18].sum()/1000)
current_share_alumi_vehic = graph5_data[1][14:18].sum()/(current_production_aluminium.values[0][14:18].sum()/1000)
current_share_alumi_elect = graph5_data[2][14:18].sum()/(current_production_aluminium.values[0][14:18].sum()/1000)
current_share_alumi_total = graph5_sum[14:18].sum()/(current_production_aluminium.values[0][14:18].sum()/1000)

current_share_coppr_build = graph6_data[0][17].sum()/(current_production_copper.values[0][17].sum()/1000)
current_share_coppr_vehic = graph6_data[1][17].sum()/(current_production_copper.values[0][17].sum()/1000)
current_share_coppr_elect = graph6_data[2][17].sum()/(current_production_copper.values[0][17].sum()/1000)
current_share_coppr_total = graph6_sum[17].sum()/(current_production_copper.values[0][17].sum()/1000)


