# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:53:21 2020
@author: deetman@cml.leidenuniv.nl

This module is used to calculate the materials involved in the electricity grid
Input:  1) scenario files from the IMAGE Integrated Assessment Model
        2) data files on material intensities, costs, lifetimes and weights
Output: Total global material use (stocks, inflow & outflow) in the electricity grid for the period 2000-2050 based on:
        1) the second Shared socio-economic pathway (SSP2) Baseline (2022 update)
        2) the second Shared socio-economic pathway (SSP2) Baseline with Current CLimate Policies (2022 update)
        3) the second Shared socio-economic pathway (SSP2) 1.5-degree Climate Policy scenario (2022 update)

4 senitivity settings are defined:
1) 'default'    is the default model setting, used for the outcomes as described in the main text
2) 'high_stor'  defines a pessimistic setting with regard to storage demand (high) and availability (low) (not relevant here, see Electricity_sector.py)
3) 'high_grid'  defines alternative assumptions with respect to the growth of the grid 
4) 'dynamic_MI' uses dynamic Material Intensity assumptions with regard to solar and wind, batteries and underground HV transmission cables as described in the main text

"""
# define imports, counters & settings
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from past.builtins import execfile

YOUR_DIR = 'C:\\Users\\Admin\\surfdrive\\Projects\\IRP\\GRO23\\Modelling\\2060\\ELMA'
os.chdir(YOUR_DIR)

datapath = YOUR_DIR + '\\data\\'

execfile('read_mym.py')        # module to read in mym (IMAGE output) files
idx = pd.IndexSlice            # needed for slicing multi-index dataframes (pandas)
scenario = "SSP2"
variant  = "2D_RE"             # switch to 'CP' for Current Policies or '2D' for a 2-Degrees Climate Policy target (Add '_RE' to apply Resource Efficiency)
sa_settings = "default"        # settings for the sensitivity analysis (default, high_grid, dynamic_MI)
scen_folder = scenario + "_" + variant + "\\"
path = scenario + "\\" +  scen_folder

startyear = 1971
endyear   = 2060               # end year of the calculations
fileyear  = 2100               # last year available in the IMAGE data-files (which are input to ELMA)
outyear   = 2060               # last year of reporting (in the output files)
years = endyear - startyear  + 1
regions   = 26
epg_techs = 34
material_select = 'Steel'

#%% Read in files --------------------------------------------------------------------------

# IMAGE file: Generation capacity (stock) in MW peak capacity (Used as proxy for grid growth)
gcap_BL_data = read_mym_df('SSP2\\SSP2_BL\\Gcap.out')
gcap_BL_data = gcap_BL_data.loc[~gcap_BL_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_BL = pd.pivot_table(gcap_BL_data[gcap_BL_data['time'].isin(list(range(startyear,endyear+1)))], index=['time','DIM_1'], values=list(range(1,epg_techs+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

gcap_data = read_mym_df(path + 'Gcap.out')
gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(startyear,endyear+1)))], index=['time','DIM_1'], values=list(range(1,epg_techs+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

# read in additional (non-IMAGE) grid data files (grid length, grid additions & material composition)
grid_length_Hv = pd.read_csv(datapath + 'grid_length_Hv.csv', index_col=0, names=None).transpose()    # lenght of the High-voltage (Hv) lines in the grid, based on Open Street Map (OSM) analysis (km)
ratio_Hv = pd.read_csv(datapath + 'Hv_ratio.csv', index_col=0)                                        # Ratio between the length of Medium-voltage (Mv) and Low-voltage (Lv) lines in relation to Hv lines (km Lv /km Hv) & (km Mv/ km Hv)
underground_ratio = pd.read_csv(datapath + 'underground_ratio.csv', index_col=[0,1])                  # these contain the definition of the constants in the linear function used to determine the relation between income & the percentage of underground power-lines (% underground = mult * gdp/cap + add)
grid_additions = pd.read_csv(datapath + 'grid_additions.csv', index_col=0)                            # Transformers & substations per km of grid (Hv, Mv & Lv, in units/km)

# dynamic or scenario-dependent data (lifetimes & material intensity)
lifetime_grid_elements = pd.read_csv(path + 'operational_lifetime_grid.csv', index_col=0)         # Average lifetime in years of grid elements

# dynamic material intensity files (kg/km or kg/unit)
materials_grid = pd.read_csv(datapath + scen_folder + 'Materials_grid_dynamic.csv', index_col=[0,1])                # Material intensity of grid lines specific material content for Hv, Mv & Lv lines, & specific for underground vs. aboveground lines. (kg/km)
materials_grid_additions = pd.read_csv(datapath + scen_folder + 'Materials_grid_additions.csv', index_col=[0,1])    # (not part of the SA yet) Additional infrastructure required for grid connections, such as transformers & substations (material compositin in kg/unit)

region_list = list(grid_length_Hv.columns.values)
material_list = list(materials_grid.columns.values)
   
# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap_techlist = ['Solar PV', 'Solar Decentral', 'CSP', 'Wind onshore', 'Wind offshore', 'Wave', 'Hydro', 'Other Renewables', 'Geothermal', 'Hydrogen', 'Nuclear', '<EMPTY>', 'Conv. Coal', 'Conv. Oil', 'Conv. Natural Gas', 'Waste', 'IGCC', 'OGCC', 'NG CC', 'Biomass CC', 'Coal + CCS', 'Oil/Coal + CCS', 'Natural Gas + CCS', 'Biomass + CCS', 'CHP Coal', 'CHP Oil', 'CHP Natural Gas', 'CHP Biomass', 'CHP Coal + CCS', 'CHP Oil + CCS', 'CHP Natural Gas + CCS', 'CHP Biomass + CCS', 'CHP Geothermal', 'CHP Hydrogen']
gcap.index = pd.MultiIndex.from_product([list(range(startyear,endyear+1)), region_list], names=['years', 'regions'])
gcap_BL.index = pd.MultiIndex.from_product([list(range(startyear,endyear+1)), region_list], names=['years', 'regions'])
gcap.columns = gcap_techlist
gcap_BL.columns = gcap_techlist

# IMAGE file: GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
gdp_pc = pd.read_csv(path + 'gdp_pc.csv', index_col=0)  
gdp_pc.columns = region_list
gdp_pc = gdp_pc.drop([1970]).drop(list(range(endyear+1,fileyear+1)))

#%% length calculations ----------------------------------------------------------------------------

# only the regional total (peak) generation capacity is used as a proxy for the grid growth (BL to 2016, then BL or 450)
gcap_BL_total = gcap_BL.sum(axis=1).unstack()
gcap_BL_total = gcap_BL_total[region_list]               # re-order columns to the original TIMER order
gcap_growth = gcap_BL_total / gcap_BL_total.loc[2016]    # define growth according to 2016 as base year
gcap_total = gcap.sum(axis=1).unstack()
gcap_total = gcap_total[region_list]                     # re-order columns to the original TIMER order
gcap_growth.loc[2016:endyear] = gcap_total.loc[2016:endyear] / gcap_total.loc[2016]        # define growth according to 2016 as base year

# in the sensitivity variant, additional growth is presumed after 2020 based on the fraction of variable renewable energy (vre) generation capacity (solar & wind)
vre_fraction = gcap[['Solar PV', 'CSP', 'Wind onshore', 'Wind offshore']].sum(axis=1).unstack().divide(gcap.sum(axis=1).unstack())
add_growth = vre_fraction * 1                  # 0.2 = 20% additional HV lines per doubling of vre gcap
red_growth = (1-vre_fraction) * 0.7            # 0.2 = 20% less HV lines per doubling of baseline gcap
add_growth.loc[list(range(1971,2020+1)),:] = 0  # pre 2020, additional HV grid growth is 0, afterwards the additional line length is gradually introduced (towards 2050)
red_growth.loc[list(range(1971,2020+1)),:] = 0  # pre 2020, reduction of HV grid growth is 0, afterwards the line length reduction is gradually introduced (towards 2050)
for year in range(2020,2050+1):
   add_growth.loc[year] = add_growth.loc[year] * (1/30*(year-2020)) 
   red_growth.loc[year] = red_growth.loc[year] * (1/30*(year-2020)) 

# Hv length (in kms) is region-specific. However, we use a single ratio between the length of Hv and Mv networks, the same applies to Lv networks 
grid_length_Mv = grid_length_Hv.mul(ratio_Hv['Hv to Mv'])
grid_length_Lv = grid_length_Hv.mul(ratio_Hv['Hv to Lv'])

# define grid length over time (fixed in 2016, growth according to gcap)
grid_length_Hv_time = pd.DataFrame().reindex_like(gcap_total)
grid_length_Mv_time = pd.DataFrame().reindex_like(gcap_total)
grid_length_Lv_time = pd.DataFrame().reindex_like(gcap_total)

#implement growth correction (sensitivity variant)
if sa_settings == 'high_grid':
   gcap_growth_HV = gcap_growth.add(add_growth.reindex_like(gcap_growth)).subtract(red_growth.reindex_like(gcap_growth))
else: 
   gcap_growth_HV = gcap_growth

for year in range(startyear, endyear+1):
   grid_length_Hv_time.loc[year] = gcap_growth_HV.loc[year].mul(grid_length_Hv.loc['2016'])
   grid_length_Mv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Mv.loc['2016'])
   grid_length_Lv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Lv.loc['2016'])

# define underground vs. aboveground fraction (%) based on static ratios (the Hv length is the aboveground fraction according to Open Street Maps, we add the underground fractions for 3 voltage networks)
# Based on new insights from Kalt et al. 2021, we adjust the underground ratios downwards for non-European regions
function_Hv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)
function_Mv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)
function_Lv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)

for region in region_list:
    if region in ['W.Europe','C.Europe']:
        select_proxy = 'Europe'
    else:
        select_proxy = 'Other'
    #print(str(region) + ': ' + select_proxy)
    function_Hv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'Hv'] + underground_ratio.loc[idx[select_proxy,'add'],'Hv']
    function_Mv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'Mv'] + underground_ratio.loc[idx[select_proxy,'add'],'Mv']
    function_Lv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'Lv'] + underground_ratio.loc[idx[select_proxy,'add'],'Lv']

# maximize linear function at 100 & minimize at 0 (%)
function_Hv_under = function_Hv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])                     
function_Hv_under = function_Hv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
function_Mv_under = function_Mv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
function_Mv_under = function_Mv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
function_Lv_under = function_Lv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
function_Lv_under = function_Lv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])

# MIND! the HV lines found in OSM (+national sources) are considered as the total of the aboveground line length + the underground line length
grid_length_Hv_total = grid_length_Hv_time                                      # assuming the length from OSM IS the abovegrond fraction
grid_length_Hv_above = grid_length_Hv_total * (1 - (function_Hv_under/100)) 
grid_length_Hv_under = grid_length_Hv_total * function_Hv_under/100

# out for main text figure 2
grid_length_HV_out_a = pd.concat([grid_length_Hv_above], keys=['aboveground'], names=['type']) 
grid_length_HV_out_u = pd.concat([grid_length_Hv_under], keys=['underground'], names=['type']) 
grid_length_HV_out = pd.concat([grid_length_HV_out_a, grid_length_HV_out_u])
grid_length_HV_out.to_csv('output\\' + scen_folder + '\\' + sa_settings + '\\grid_length_HV_km.csv') # in km

grid_length_Mv_above = grid_length_Mv_time * (1 - function_Mv_under/100)
grid_length_Mv_under = grid_length_Mv_time * function_Mv_under/100
grid_length_Mv_total = grid_length_Mv_above + grid_length_Mv_under

grid_length_Lv_above = grid_length_Lv_time * (1 - function_Lv_under/100)
grid_length_Lv_under = grid_length_Lv_time * function_Lv_under/100
grid_length_Lv_total = grid_length_Lv_above + grid_length_Lv_under

grid_subst_Hv = grid_length_Hv_total.mul(grid_additions.loc['Substations','Hv'])        # number of substations on Hv network
grid_subst_Mv = grid_length_Mv_total.mul(grid_additions.loc['Substations','Mv'])        # # of substations
grid_subst_Lv = grid_length_Lv_total.mul(grid_additions.loc['Substations','Lv'])        # # of substations
grid_trans_Hv = grid_length_Hv_total.mul(grid_additions.loc['Transformers','Hv'])       # number of transformers on the Hv network
grid_trans_Mv = grid_length_Mv_total.mul(grid_additions.loc['Transformers','Mv'])       # # of transformers
grid_trans_Lv = grid_length_Lv_total.mul(grid_additions.loc['Transformers','Lv'])       # # of transformers


#%% Inflow Outflow calculations 
from dynamic_stock_model import DynamicStockModel as DSM
first_year_grid = 1926          # UK Electricity supply act - https://www.bbc.com/news/uk-politics-11619751
stdev_mult = 0.214              # standard deviation as a fraction of the mean lifetime applicable to energy equipment (Asset Management for Infrastructure Systems: Energy and Water, Balzer & Schorn 2015)

# Interpolate material intensities (dynamic content from 1926 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(first_year_grid,endyear+1)), list(materials_grid.index.levels[1])])
materials_grid_interpol = pd.DataFrame(index=index, columns=materials_grid.columns)
materials_grid_additions_interpol = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(first_year_grid,endyear+1)), list(materials_grid_additions.index.levels[1])]), columns=materials_grid_additions.columns)

for cat in list(materials_grid.index.levels[1]):
   materials_grid_1st   = materials_grid.loc[idx[materials_grid.index[0][0], cat],:]
   materials_grid_interpol.loc[idx[first_year_grid ,cat],:] = materials_grid_1st                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   materials_grid_interpol.loc[idx[materials_grid.index.levels[0].min(),cat],:] = materials_grid.loc[idx[materials_grid.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   materials_grid_interpol.loc[idx[materials_grid.index.levels[0].max(),cat],:] = materials_grid.loc[idx[materials_grid.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   materials_grid_interpol.loc[idx[:,cat],:] = materials_grid_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(first_year_grid,endyear+1)), level=0).interpolate()

for cat in list(materials_grid_additions.index.levels[1]):
   materials_grid_additions_1st   = materials_grid_additions.loc[idx[materials_grid_additions.index[0][0], cat],:]
   materials_grid_additions_interpol.loc[idx[first_year_grid ,cat],:] = materials_grid_additions_1st          # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   materials_grid_additions_interpol.loc[idx[materials_grid_additions.index.levels[0].min(),cat],:] = materials_grid_additions.loc[idx[materials_grid_additions.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   materials_grid_additions_interpol.loc[idx[materials_grid_additions.index.levels[0].max(),cat],:] = materials_grid_additions.loc[idx[materials_grid_additions.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   materials_grid_additions_interpol.loc[idx[:,cat],:] = materials_grid_additions_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(first_year_grid,endyear+1)), level=0).interpolate()

# interpolate lifetimes
lifetime_grid_elements.loc[first_year_grid,:] = lifetime_grid_elements.loc[lifetime_grid_elements.first_valid_index(),:]
lifetime_grid_elements.loc[outyear,:]         = lifetime_grid_elements.loc[lifetime_grid_elements.last_valid_index(),:]
lifetime_grid_elements                        = lifetime_grid_elements.reindex(list(range(first_year_grid, outyear+1))).interpolate()
   
# In order to calculate inflow & outflow smoothly (without peaks for the initial years), we calculate a historic tail to the stock, by adding a 0 value for first year of operation (=1926), then interpolate values towards 1971
def stock_tail(grid_stock):
    zero_value = [0 for i in range(0,regions)]
    grid_stock.loc[first_year_grid] = zero_value  # set all regions to 0 in the year of initial operation
    grid_stock_new  = grid_stock.reindex(list(range(first_year_grid,endyear+1))).interpolate()
    return grid_stock_new

# call the stock_tail function on all lines, substations & transformers, to add historic stock tail between 1926 & 1971
grid_length_Hv_above_new = stock_tail(grid_length_Hv_above) # km
grid_length_Mv_above_new = stock_tail(grid_length_Mv_above) # km
grid_length_Lv_above_new = stock_tail(grid_length_Lv_above) # km
grid_length_Hv_under_new = stock_tail(grid_length_Hv_under) # km 
grid_length_Mv_under_new = stock_tail(grid_length_Mv_under) # km
grid_length_Lv_under_new = stock_tail(grid_length_Lv_under) # km
grid_subst_Hv_new = stock_tail(grid_subst_Hv)               # units
grid_subst_Mv_new = stock_tail(grid_subst_Mv)               # units
grid_subst_Lv_new = stock_tail(grid_subst_Lv)               # units
grid_trans_Hv_new = stock_tail(grid_trans_Hv)               # units
grid_trans_Mv_new = stock_tail(grid_trans_Mv)               # units
grid_trans_Lv_new = stock_tail(grid_trans_Lv)               # units

# Function in which the stock-driven DSM is applied to return (the moving average of the) materials in the inflow & outflow for all regions
# material calculations are done in the same function as the dynamic stock calculations to be lean on memory (stock- & outlfow-by-cohort are large dataframes, which do nod need to be stored in this way)
def inflow_outflow(stock, lifetime, material_intensity):

    initial_year = stock.first_valid_index()
    outflow_mat  = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,outyear+1), material_intensity.columns]), columns=stock.columns)
    inflow_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,outyear+1), material_intensity.columns]), columns=stock.columns)   
    stock_mat    = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,outyear+1), material_intensity.columns]), columns=stock.columns)
    out_oc_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,endyear+1), material_intensity.columns]), columns=stock.columns)
    out_sc_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,endyear+1), material_intensity.columns]), columns=stock.columns)
    out_in_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,endyear+1), material_intensity.columns]), columns=stock.columns)

    # define standard deviation (originally expressed as a fraction of the mean)
    stdev_list = lifetime * stdev_mult  

    for region in list(stock.columns):
        # define and run the DSM                                                                                            # list with the fixed (=mean) lifetime of grid elements, given for every timestep (1926-2100), needed for the DSM as it allows to change lifetime for different cohort (even though we keep it constant)
        DSMforward = DSM(t = np.arange(0,len(stock[region]),1), s=np.array(stock[region]), lt = {'Type': 'FoldedNormal', 'Mean': np.array(lifetime), 'StdDev': np.array(stdev_list)})  # definition of the DSM based on a folded normal distribution
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)                                                                 # run the DSM, to give 3 outputs: stock_by_cohort, outflow_by_cohort & inflow_per_year
    
        #convert to pandas df before multiplication with material intensity
        index=list(range(first_year_grid, endyear+1))
        out_sc_pd = pd.DataFrame(out_sc, index=index,  columns=index)
        out_oc_pd = pd.DataFrame(out_oc, index=index,  columns=index)
        out_in_pd = pd.DataFrame(out_i,  index=index)
        
        # sum the outflow & stock by cohort (using cohort specific material intensities)
        for material in list(material_intensity.columns):    
           out_oc_mat.loc[idx[:,material],region] = out_oc_pd.mul(material_intensity.loc[:,material], axis=1).sum(axis=1).to_numpy()
           out_sc_mat.loc[idx[:,material],region] = out_sc_pd.mul(material_intensity.loc[:,material], axis=1).sum(axis=1).to_numpy() 
           out_in_mat.loc[idx[:,material],region] = out_in_pd.mul(material_intensity.loc[:,material], axis=0).to_numpy()                
    
           # apply moving average & return only 1971-2050 values
           outflow_mat.loc[idx[:,material],region] = pd.Series(out_oc_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year,outyear+1))).rolling(window=5).mean().loc[list(range(1971,outyear + 1))].to_numpy()    # Apply moving average                                                                                                      # sum the outflow by cohort to get the total outflow per year
           inflow_mat.loc[idx[:,material],region]  = pd.Series(out_in_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year,outyear+1))).rolling(window=5).mean().loc[list(range(1971,outyear + 1))].to_numpy()
           stock_mat.loc[idx[:,material],region]   = pd.Series(out_sc_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year,outyear+1))).rolling(window=5).mean().loc[list(range(1971,outyear + 1))].to_numpy()                                                                                                       # sum the outflow by cohort to get the total outflow per year
     
    return inflow_mat, outflow_mat, stock_mat



# HV inflow/outflow (in kgs)
Hv_lines_above_in, Hv_lines_above_out, Hv_lines_above_stock = inflow_outflow(grid_length_Hv_above_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'HV overhead'],:].droplevel(1))    
Hv_lines_under_in, Hv_lines_under_out, Hv_lines_under_stock = inflow_outflow(grid_length_Hv_under_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'HV underground'],:].droplevel(1)) 
Hv_subst_in, Hv_subst_out, Hv_subst_stock                   = inflow_outflow(grid_subst_Hv_new,        lifetime_grid_elements['substations'],  materials_grid_additions_interpol.loc[idx[:,'Hv Substation'],:].droplevel(1))              
Hv_trans_in, Hv_trans_out, Hv_trans_stock                   = inflow_outflow(grid_trans_Hv_new,        lifetime_grid_elements['transformers'], materials_grid_additions_interpol.loc[idx[:,'Hv Transformer'],:].droplevel(1))            

#MV inflow/outflow (in kgs)
Mv_lines_above_in, Mv_lines_above_out, Mv_lines_above_stock = inflow_outflow(grid_length_Mv_above_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'MV overhead'],:].droplevel(1))  # For now, the same lifetime assumptions for HV elements are applied to MV & LV
Mv_lines_under_in, Mv_lines_under_out, Mv_lines_under_stock = inflow_outflow(grid_length_Mv_under_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'MV underground'],:].droplevel(1))
Mv_subst_in, Mv_subst_out, Mv_subst_stock                   = inflow_outflow(grid_subst_Mv_new,        lifetime_grid_elements['substations'],  materials_grid_additions_interpol.loc[idx[:,'Mv Substation'],:].droplevel(1))
Mv_trans_in, Mv_trans_out, Mv_trans_stock                   = inflow_outflow(grid_trans_Mv_new,        lifetime_grid_elements['transformers'], materials_grid_additions_interpol.loc[idx[:,'Mv Transformer'],:].droplevel(1))

#LV inflow/outflow (in kgs)
Lv_lines_above_in, Lv_lines_above_out, Lv_lines_above_stock = inflow_outflow(grid_length_Lv_above_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'LV overhead'],:].droplevel(1))
Lv_lines_under_in, Lv_lines_under_out, Lv_lines_under_stock = inflow_outflow(grid_length_Lv_under_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'LV underground'],:].droplevel(1))
Lv_subst_in, Lv_subst_out, Lv_subst_stock                   = inflow_outflow(grid_subst_Lv_new,        lifetime_grid_elements['substations'],  materials_grid_additions_interpol.loc[idx[:,'Lv Substation'],:].droplevel(1))
Lv_trans_in, Lv_trans_out, Lv_trans_stock                   = inflow_outflow(grid_trans_Lv_new,        lifetime_grid_elements['transformers'], materials_grid_additions_interpol.loc[idx[:,'Lv Transformer'],:].droplevel(1))

#%% prepare output variables

# overall container of stock, inflow & outflow of materials (in kgs)
items = ['lines', 'substations', 'transformers']
flow =  ['stock', 'inflow', 'outflow']
columns = pd.MultiIndex.from_product([region_list, items], names=['regions', 'elements'])
index = pd.MultiIndex.from_product([flow, list(range(startyear,outyear + 1)), material_list], names=['flow', 'years', 'materials'])
grid_materials = pd.DataFrame(index=index, columns=columns)

# list used to sum detailed (voltage level specific) stock to total of lines
lines_list = ['HV overhead', 'HV underground', 'MV overhead', 'MV underground', 'LV overhead', 'LV underground']    # lines
subst_list = ['HV substations','MV substations','LV substations']
trans_list = ['HV transformers','MV transformers','LV transformers']

grid_materials.loc[idx['stock', :,:],idx[:,'lines']]          =  Hv_lines_above_stock.to_numpy() + Hv_lines_under_stock.to_numpy() + Mv_lines_above_stock.to_numpy() + Mv_lines_under_stock.to_numpy() + Lv_lines_above_stock.to_numpy() + Lv_lines_under_stock.to_numpy() # kgs in stock for lines
grid_materials.loc[idx['stock', :,:],idx[:,'substations']]    =  Hv_subst_stock.to_numpy() + Mv_subst_stock.to_numpy() + Lv_subst_stock.to_numpy() # kgs in stock for substations
grid_materials.loc[idx['stock', :,:],idx[:,'transformers']]   =  Hv_trans_stock.to_numpy() + Mv_trans_stock.to_numpy() + Lv_trans_stock.to_numpy() # kgs in stock for transformers
grid_materials.loc[idx['inflow', :,:],idx[:,'lines']]         =  Hv_lines_above_in.to_numpy() + Hv_lines_under_in.to_numpy() + Mv_lines_above_in.to_numpy() + Mv_lines_under_in.to_numpy() + Lv_lines_above_in.to_numpy() + Lv_lines_under_in.to_numpy()
grid_materials.loc[idx['inflow', :,:],idx[:,'substations']]   =  Hv_subst_in.to_numpy() + Mv_subst_in.to_numpy() + Lv_subst_in.to_numpy()
grid_materials.loc[idx['inflow', :,:],idx[:,'transformers']]  =  Hv_trans_in.to_numpy() + Mv_trans_in.to_numpy() + Lv_trans_in.to_numpy() 
grid_materials.loc[idx['outflow', :,:],idx[:,'lines']]        =  Hv_lines_above_out.to_numpy() + Hv_lines_under_out.to_numpy() + Mv_lines_above_out.to_numpy() + Mv_lines_under_out.to_numpy() + Lv_lines_above_out.to_numpy() + Lv_lines_under_out.to_numpy()
grid_materials.loc[idx['outflow', :,:],idx[:,'substations']]  =  Hv_subst_out.to_numpy() + Mv_subst_out.to_numpy() + Lv_subst_out.to_numpy() 
grid_materials.loc[idx['outflow', :,:],idx[:,'transformers']] =  Hv_trans_out.to_numpy() + Mv_trans_out.to_numpy() + Lv_trans_out.to_numpy() 

#%% CSV output files (in kilo-tonnes, kt) - 3 flow types (stock, inflow, outlfow) - 26 Regions - 3 product categories (lines, substations, transformers) - 14 materials (but some are not releveant to the grid, so they are 0)
grid_materials_out = pd.concat([grid_materials], keys=['grid'], names=['category']).stack().stack() # add a descriptor column
grid_materials_out = pd.concat([grid_materials_out], keys=['electricity'], names=['sector'])
grid_materials_out = grid_materials_out.unstack(level=3).reorder_levels([5, 2, 0, 1, 4, 3]) / 1000000   # to kt
grid_materials_out.to_csv('output\\' + scen_folder + '\\' + sa_settings + '\\grid_materials_output_kt.csv') # in kt

