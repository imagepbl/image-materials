#%%
import pandas as pd
import numpy as np
import os
import scipy
import warnings
from pathlib import Path
import pint
import xarray as xr
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# path_current = Path(__file__).resolve().parent # absolute path of file
# path_base = path_current.parent.parent # base path of the project -> image-materials
# sys.path.append(str(path_base))

import prism
from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_life_time_vehicles
from imagematerials.model import GenericMainModel, GenericMaterials, GenericStocks, Maintenance, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.concepts import create_electricity_graph
from imagematerials.electricity.utils import MNLogit, stock_tail, materials_grid_additions_to_kgperkm, print_df_info


from imagematerials.electricity.constants import (
    YEAR_START,
    YEAR_FIRST,
    YEAR_FIRST_GRID,
    YEAR_END,
    YEAR_OUT,
    YEAR_SWITCH,
    YEAR_LAST,
    # COHORTS, # necessary?
    SCEN,
    VARIANT,
    SENS_ANALYSIS,
    REGIONS,
    TECH_GEN,
    STD_LIFETIMES_ELECTR,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    TONNES_TO_KGS,
    LOAD_FACTOR,
    BEV_CAPACITY_CURRENT,
    PHEV_CAPACITY_CURRENT,
    dict_gentech_styles,
    dict_materials_colors,
    dict_grid_colors
)


# Define paths ----------------------------------------------------------------------
#YOUR_DIR = "C:\\Users\\Admin\\surfdrive\\Projects\\IRP\\GRO23\\Modelling\\2060\\ELMA"   # Change the running directory here
# os.chdir(YOUR_DIR)
scen_folder = SCEN + "_" + VARIANT
# path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
path_image_output = Path(path_base, "data", "raw", SCEN, "EnergyServices")
path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)

assert path_image_output.is_dir()
assert path_external_data_standard.is_dir()
assert path_external_data_scenario.is_dir()

# create the folder out_test if it does not exist
if not (path_base / 'imagematerials' / 'electricity' / 'out_test').is_dir():
    (path_base / 'imagematerials' / 'electricity' / 'out_test').mkdir(parents=True)






years = YEAR_END - YEAR_START  + 1


# from past.builtins import execfile
# execfile('read_mym.py')
idx = pd.IndexSlice             # needed for slicing multi-index


# TODO: decide on variable naming convention
# V1: sector_variableinquestion_additionalinformation (gcap_lifetime_interpolated)
# V2: variableinquestion_sector_additionalinformation (lifetime_gcap_interpolated)

###########################################################################################################
###########################################################################################################
#%% 1) Generation 
###########################################################################################################
###########################################################################################################


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


# 1. External Data ======================================================================================== 

# lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
gcap_lifetime = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        
# gcap_lifetime = standardize_index_spelling(gcap_lifetime) # standardize the index name
# material compositions (generation capacity)
composition_generation = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()  # in gram/MW
# composition_generation = composition_generation.drop(columns=[col for col in composition_generation.columns if col[1] == '<EMPTY>'])
kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')  # to get region list without running storage - TODO: get regions from different source


# 2. IMAGE/TIMER files ====================================================================================

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data = read_mym_df(path_image_output / 'Gcap.out')


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.2) Prepare model specific variables
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

region_list = list(kilometrage.columns.values)   

gcap_tech_list = list(composition_generation.loc[:,idx[2020,:]].droplevel(axis=1, level=0).columns)    #list of names of the generation technologies (workaround to retain original order)
gcap_material_list = list(composition_generation.index.values)  #list of materials the generation technologies

gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (34));  the last column in gcap_data (= totals) is now removed

# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
gcap.columns = gcap_tech_list

# Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, YEAR_OUT+1)), list(composition_generation.index)])
gcap_materials_interpol = pd.DataFrame(index=index, columns=composition_generation.columns.levels[1])

# material intensities for gcap
for cat in list(composition_generation.columns.levels[1]):
   gcap_materials_1st   = composition_generation.loc[:,idx[composition_generation.columns[0][0],cat]]
   gcap_materials_interpol.loc[idx[YEAR_FIRST_GRID ,:],cat] = gcap_materials_1st.to_numpy()                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].min(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].min(),cat]].to_numpy()                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].max(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].max(),cat]].to_numpy()                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   gcap_materials_interpol.loc[idx[:,:],cat] = gcap_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

# interpolate Gcap (technical) lifetime data
gcap_lifetime.index = gcap_lifetime.index.set_levels(gcap_tech_list, level=1)
gcap_lifetime = gcap_lifetime.unstack().droplevel(axis=1, level=0)
gcap_lifetime = gcap_lifetime.reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1)), axis=0).interpolate(limit_direction='both')

# Calculate the historic tail to the Gcap (stock) 
gcap_new = pd.DataFrame(index=pd.MultiIndex.from_product([range(YEAR_FIRST_GRID,YEAR_OUT+1), region_list], names=['years', 'regions']), columns=gcap.columns)
for tech in gcap_tech_list:
    gcap_new.loc[idx[:,:],tech] = stock_tail(gcap.loc[idx[:,:],tech].unstack(level=1)).stack()


# Bring dataframes into correct shape for the results_dict
# I need:
#   A. with only Types: stocks Types (GW), lifetimes Types, material intensities Types
#   B. with also SubTypes: stocks Types (GW), lifetimes SubTypes, material intensities SubTypes, market shares SubTypes

# A.
# stocks: (years, regions) index and technologies as columns -> years as index and (technology, region) as columns
gcap_stock = gcap_new.unstack(level='regions')

# lifetimes
df_mean = gcap_lifetime.copy()
df_stdev = df_mean * STD_LIFETIMES_ELECTR
df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
gcap_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns

# MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
gcap_materials_interpol.index.names = ["Year", "Material"]
# gcap_materials_interpol = gcap_materials_interpol.loc[:, ~(gcap_materials_interpol == 0.0).all()] # delete empty columns
gcap_types_materials = gcap_materials_interpol.unstack(level='Material')


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.3) Prep_data File
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


ureg = pint.UnitRegistry(force_ndarray_like=True)
# Define the units for each dimension
unit_mapping = {
    'time': ureg.year,
    'year': ureg.year,
    'Year': ureg.year,
    'kg': ureg.kilogram,
    'yr': ureg.year,
    '%': ureg.percent,
    't': ureg.tonne,
    'MW': ureg.megawatt, #added
    'GW': ureg.gigawatt, #added
}

# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "gcap_stock": (["Time"], ["Type", "Region"],),
    "gcap_types_materials": (["Cohort"], ["Type", "material"],)
    # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
}


# results_dict = {
#         'total_nr_vehicles_simple': total_nr_vehicles_simple,
#         'material_fractions_simple': material_fractions_simple,
#         'material_fractions_typical': material_fractions_typical,
#         'vehicle_weights_simple': vehicle_weights_simple,
#         'vehicle_weights_typical': vehicle_weights_typical,
#         'lifetimes': lifetimes_vehicles,
#         'battery_weights_typical': battery_weights_typical,
#         'battery_materials': battery_materials,
#         'battery_shares': battery_shares,
#         'weight_boats': weight_boats,
#         'vehicle_shares_typical': vehicle_shares_typical
#     }
results_dict = {
        'gcap_stock': gcap_stock,
        'gcap_types_materials': gcap_types_materials,
        'gcap_lifetime_distr': gcap_lifetime_distr,
}

# df = gcap_materials_interpol.copy()


# Convert the DataFrames to xarray Datasets and apply units
preprocessing_results_xarray = {}


for df_name, df in results_dict.items():
    if df_name in conversion_table:
        data_xar_dataset = pandas_to_xarray(df, unit_mapping)
        data_xarray = dataset_to_array(data_xar_dataset, *conversion_table[df_name])
    else:
        # lifetimes_vehicles does not need to be converted in the same way.
        data_xarray = pandas_to_xarray(df, unit_mapping)
    preprocessing_results_xarray[df_name] = data_xarray



preprocessing_results_xarray["lifetimes"] = convert_life_time_vehicles(preprocessing_results_xarray["gcap_lifetime_distr"])
preprocessing_results_xarray["stocks"] = preprocessing_results_xarray.pop("gcap_stock")
preprocessing_results_xarray["material_intensities"] = preprocessing_results_xarray.pop("gcap_types_materials")
# preprocessing_results_xarray["shares"] = preprocessing_results_xarray.pop("vehicle_shares")

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.4) Run Stock Model New
###########################################################################################################
# TODO: move this to electricity.py
#----------------------------------------------------------------------------------------------------------

prep_data = preprocessing_results_xarray.copy()

# # Define the complete timeline, including historic tail
time_start = prep_data["stocks"].coords["Time"].min().values
# time_start = 1960
complete_timeline = prism.Timeline(time_start, 2060, 1)
simulation_timeline = prism.Timeline(1970, 2060, 1)

# Define the coordinates of all dimensions.
# Region = list(prep_data["stocks"].coords["Region"].values)
# Time = [t for t in complete_timeline]
# Cohort = Time
# Type = list(prep_data["stocks"].coords["Type"].values)
# material = list(prep_data["material_intensities"].coords["material"].values)

# Create
# main_model_normal = GenericMainModel(
#     complete_timeline, Region=Region, Time=Time, Cohort=Cohort, Type=Type, prep_data=prep_data,
#     compute_materials=True, compute_battery_materials=False, compute_maintenance_materials=False, 
#     material=material)

new_prep_data = prep_data.copy()
new_prep_data["knowledge_graph"] = create_electricity_graph()
new_prep_data["shares"] = None

sec_elctr_gen = Sector("elctr_gen", new_prep_data)



main_model_factory = ModelFactory(
    sec_elctr_gen, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory.simulate(simulation_timeline)



###########################################################################################################
#%%% 1.5) Tests: Check results

# list(main_model_factory.default)

main_model_factory.inflow

tv = main_model_factory.inflow

# Extract years
years = np.arange(tv.timeline.start, tv.timeline.end + 1, tv.timeline.stepsize)

# Extract dimension labels and coordinates
dim_labels = [d.label for d in tv.dims]
dim_coords = {d.label: d.coords for d in tv.dims}

# Example: extract regions and types
regions = [str(r) for r in dim_coords["Region"]]
types = [str(t) for t in dim_coords["Type"]]




###########################################################################################################
#%%% 1.6) Visualize Stocks

# Generation technologies
technologies = [
    'Solar PV', 'Solar PV residential', 'CSP', 'Wind onshore', 'Wind offshore', 'Wave', 'Hydro', 
    'Other Renewables', 'Geothermal', 'Hydrogen power', 'Nuclear', '<EMPTY>', 'Conv. Coal', 
    'Conv. Oil', 'Conv. Natural Gas', 'Waste', 'IGCC', 'OGCC', 'NG CC', 'Biomass CC', 
    'Coal + CCS', 'Oil/Coal + CCS', 'Natural Gas + CCS', 'Biomass + CCS', 'CHP Coal', 
    'CHP Oil', 'CHP Natural Gas', 'CHP Biomass', 'CHP Coal + CCS', 'CHP Oil + CCS', 
    'CHP Natural Gas + CCS', 'CHP Biomass + CCS', 'CHP Geothermal', 'CHP Hydrogen'
]
# Define color and linestyle pools
# colors = list(plt.get_cmap('tab10').colors)  # 20 distinct colors
# # Add two new distinct colors (example: magenta and teal)
# colors += [(1.0, 0.0, 1.0),  # magenta
#            (0.0, 0.5, 0.5)]  # teal
# linestyles = ['-', '--', ':'] #'-.'
# # Create a cycle of (color, linestyle) combinations
# style_combinations = list(itertools.product(colors, linestyles))
# assert len(technologies) <= len(style_combinations), "Not enough unique combinations for all technologies."
# # Map technologies to (color, linestyle)
# dict_gentech_styles = {tech: style_combinations[i] for i, tech in enumerate(technologies)}

path_test_plots = Path(path_base, "imagematerials", "electricity", "out_test", "Figures")



# stocks in main_model_factory.stocks -------------------------------------------------
da_stocks = main_model_factory.stocks.copy()

regions = da_stocks.Region.values[:2]  # First 2 regions
threshold = 100_000
# types_top = da_stocks.Type.values[1:15]   # Types 1–10
# types_bottom = da_stocks.Type.values[15:30]  # Types 11–20
# techs_upper = [col for col in da_stocks.columns if da_stocks[col].max() > threshold] #gcap.columns[15:30]
# techs_lower = [col for col in da_stocks.columns if da_stocks[col].max() <= threshold] #gcap.columns[:15]
techs_upper = [coord_name.item() for coord_name in da_stocks.coords['Type']  
               if da_stocks.sel(Type = coord_name).values.max() > threshold]
techs_lower = [coord_name.item() for coord_name in da_stocks.coords['Type']  
               if da_stocks.sel(Type = coord_name).values.max() <= threshold]



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=True)
axes[0, 1].sharey(axes[0, 0])
axes[1, 1].sharey(axes[1, 0])

for i, region in enumerate(regions):
    # Top row: Types 1–15
    for t in techs_upper:
        data = da_stocks.sel(Type=t, Region=region)
        color, ls = dict_gentech_styles.get(t, ('black', '-'))  # default style fallback
        axes[0, i].plot(data.Time, data.values, label=t, color=color, linestyle=ls)
    # axes[0, i].set_title(f"{region} (Types 1–15)")
    axes[0, i].grid(alpha=0.3, linestyle='--')

    # Bottom row: Types 16–30
    for t in techs_lower:
        data = da_stocks.sel(Type=t, Region=region)
        color, ls = dict_gentech_styles.get(t, ('black', '-'))
        axes[1, i].plot(data.Time, data.values, label=t, color=color, linestyle=ls)
    # axes[1, i].set_title(f"{region} (Types 16–30)")
    axes[1, i].set_xlabel("Time")
    axes[1, i].grid(alpha=0.3, linestyle='--')
    # axes[1, i].legend(fontsize="small", loc='upper left')

# Y-axis labels only on left
axes[0, 0].set_ylabel("Value")
axes[1, 0].set_ylabel("Value")
for ax in axes.flat: # change y axis ticks from 100000 to 100,000
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
axes[0, 0].legend(fontsize='small', ncol=2, loc='upper left')
axes[1, 0].legend(fontsize='small', ncol=2, loc='upper left')

plt.suptitle("Generation - Stocks in Brazil and C.Europe", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Gen_stocks_Brazil-CEurope.png", dpi=300)
plt.show()



# stocks from TIMER --------------------------------------------------

# 2 COUNTRIES
countries = ['Brazil','C.Europe']
threshold = 100_000
techs_upper = [col for col in gcap.columns if gcap[col].max() > threshold] #gcap.columns[15:30]
techs_lower = [col for col in gcap.columns if gcap[col].max() <= threshold] #gcap.columns[:15]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=True)
axes[0, 1].sharey(axes[0, 0])
axes[1, 1].sharey(axes[1, 0])

for i, country in enumerate(countries):
    df = gcap[gcap.index.get_level_values(1) == country].copy()
    df.index = df.index.droplevel(1)

    # Top row: techs 15–30
    for tech in techs_upper:
        color, ls = dict_gentech_styles[tech]
        axes[0, i].plot(df.index, df[tech], label=tech, color=color, linestyle=ls)
    # axes[0, i].set_title(f"{country} (Technologies 16–30)")
    axes[0, i].grid(alpha=0.3, linestyle='--')

    # Bottom row: techs 0–14
    for tech in techs_lower:
        color, ls = dict_gentech_styles[tech]
        axes[1, i].plot(df.index, df[tech], label=tech, color=color, linestyle=ls)
    axes[1, i].set_xlabel('Year')
    axes[1, i].grid(alpha=0.3, linestyle='--')
    

# Add Y-axis labels only on left side
axes[0, 0].set_ylabel('Value')
axes[1, 0].set_ylabel('Value')
for ax in axes.flat: # change y axis ticks from 100000 to 100,000
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
axes[0, 0].legend(fontsize='small', ncol=2, loc='upper left')
axes[1, 0].legend(fontsize='small', ncol=2, loc='upper left')

plt.suptitle(f"Generation - Stocks in {countries[0]}_{countries[1]} (TIMER)", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / f"TIMER_Gen_stocks_{countries[0]}-{countries[1]}.png", dpi=300)
plt.show()




# 1 COUNTRY
country = 'C.Europe'
df = gcap[gcap.index.get_level_values(1) == country]
df.index = df.index.droplevel(1)

# Split columns into two groups
techs_upper = df.columns[15:30]
techs_lower = df.columns[:15]

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for tech in techs_upper:
    color, ls = dict_gentech_styles[tech]
    df[tech].plot(ax=ax1, label=tech, color=color, linestyle=ls)
ax1.set_ylabel('Value')
ax1.legend(ncol=2, fontsize='small', loc='upper left')

# Lower panel: tech 16–30
for tech in techs_lower:
    color, ls = dict_gentech_styles[tech]
    df[tech].plot(ax=ax2, label=tech, color=color, linestyle=ls)
ax2.set_xlabel('Year')
ax2.set_ylabel('Value')
ax2.legend(ncol=2, fontsize='small', loc='upper left')

plt.tight_layout()
plt.suptitle(f"Generation - Stocks in {country} (TIMER)", fontsize=16)
# fig.savefig(path_test_plots / f"TIMER_Gen_stocks_{country}.png", dpi=300)
plt.show()

###########################################################################################################
#%% Visualize Inflow Materials

# da_x = main_model_factory.inflow.to_array()
da_x = main_model_factory.inflow_materials.to_array()
da_x = main_model_factory.inflow_materials.to_array().sum('Type')

regions = ['Brazil', 'C.Europe', 'China']  # Brazil and C.Europe
# regions = da_x.Region.values[:2]  # First 2 regions
# types_top = da_x.material.values[1:6]   # Types 1–10
# types_bottom = da_x.material.values[6:12]  # Types 11–20
types_level1 = [m for m in da_x.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in da_x.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in da_x.material.values if m not in (types_level1 + types_level2)]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), sharex=True)

# axes[0, 1].sharey(axes[0, 0])
# axes[0, 2].sharey(axes[0, 0])
# axes[1, 1].sharey(axes[1, 0])
# axes[1, 2].sharey(axes[1, 0])
# axes[2, 1].sharey(axes[2, 0])
# axes[2, 2].sharey(axes[2, 0])

data_plot = da_x.sel(time=slice(1971, None))

for i, region in enumerate(regions):
    # Top row: Level 1 materials
    for t in types_level1:
        if (region in data_plot.Region.values) and (t in data_plot.material.values):
            data_plot.sel(material=t, Region=region).plot(ax=axes[0, i], label=t, color=dict_materials_colors[t])
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel(" ")
    axes[0, i].legend()

    # Middle row: Level 2 materials
    for t in types_level2:
        if (region in data_plot.Region.values) and (t in data_plot.material.values):
            data_plot.sel(material=t, Region=region).plot(ax=axes[1, i], label=t, color=dict_materials_colors[t])
    axes[1, i].set_title(f" ")
    axes[1, i].set_xlabel(" ")
    axes[1, i].legend(loc='upper left')

    # Bottom row: Level 3 materials
    for t in types_level3:
        if (region in data_plot.Region.values) and (t in data_plot.material.values):
            data_plot.sel(material=t, Region=region).plot(ax=axes[2, i], label=t, color=dict_materials_colors[t])
    axes[2, i].set_title(f" ")
    axes[2, i].set_xlabel("Time")
    axes[2, i].legend(loc='upper left')

axes[0, 0].set_ylabel("Magnitude")
axes[1, 0].set_ylabel("Magnitude")
axes[2, 0].set_ylabel("Magnitude")

plt.suptitle("Generation - Inflow Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Gen_inflow-materials_Brazil-CEurope-China_1971.png", dpi=300)
# fig.savefig(path_test_plots / "Gen_inflow-materials_Brazil-CEurope-China_1971.svg", dpi=300)
plt.show()


#---------------

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=True)

axes[0, 1].sharey(axes[0, 0])
axes[1, 1].sharey(axes[1, 0])
axes[2, 1].sharey(axes[2, 0])

for i, region in enumerate(regions):
    # Top row: 
    for t in types_level1:
        da_x.sel(material=t, Region=region).plot(ax=axes[0, i], label=t, color=dict_materials_colors[t])
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel("Time")
    axes[0, i].legend()

    # Middle row: 
    for t in types_level2:
        da_x.sel(material=t, Region=region).plot(ax=axes[1, i], label=t, color=dict_materials_colors[t])
    axes[1, i].set_title(f"{region}")
    axes[1, i].set_xlabel("Time")
    axes[1, i].legend(loc ='upper left')

    # Bottom row: 
    for t in types_level3:
        da_x.sel(material=t, Region=region).plot(ax=axes[2, i], label=t, color=dict_materials_colors[t])
    axes[2, i].set_title(f"{region}")
    axes[2, i].set_xlabel("Time")
    axes[2, i].legend(loc ='upper left')

# Label only left side with y-axis label
axes[0, 0].set_ylabel("Value")
axes[1, 0].set_ylabel("Value")
axes[2, 0].set_ylabel("Value")

plt.suptitle("Generation - Inflow Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Gen_inflow-materials_Brazil-CEurope_1971.png", dpi=300)
# fig.savefig(path_test_plots / "Gen_inflow-materials_Brazil-CEurope_1971.svg", dpi=300)
plt.show()



###########################################################################################################
#%% Visualize Inflow Materials World

dict_materials_colors = {
    'Steel':     '#FF9B85',
    'Aluminium': '#B9FAF8',
    'Concrete':  '#AAF683',
    'Plastics':  '#60D394',
    'Glass':     '#EE6055',
    'Cu':        '#FB6376',
    'Nd':        '#B8D0EB',
    'Ta':        '#B298DC',
    'Co':        '#A663CC',
    'Pb':        '#6F2DBD',
    'Mn':        "#31E7E7",
    'Ni':        '#FCB1A6',
    'Other':     '#FFD97D'
}

# data IEA ---------------------------------------------------------------------
# values for APS  scenario (NZE scenario)

# Cu -------------
years = np.array([2024, 2030, 2035, 2040, 2045, 2050])
values_solar = np.array([1657, 2803, 2726, 2626, 2448, 2758])*1000 # values in kt -> to kg
values_wind = np.array([534, 1171, 1038, 919, 968, 1314])*1000 # values in kt -> to kg
values = values_solar + values_wind
df_iea_gen_cu = pd.DataFrame({ # lines & transformers, copper
    'Year': years,
    'Cu': values
})
df_iea_gen_cu.set_index('Year', inplace=True)

#------- ---------------------------------------------------------------------

# da_x = main_model_factory.inflow.to_array()
da_x = main_model_factory.inflow_materials.to_array()
da_x = main_model_factory.inflow_materials.to_array().sum('Type').sum('Region')

types_level1 = [m for m in da_x.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in da_x.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in da_x.material.values if m not in (types_level1 + types_level2)]




fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)

# only from 1971 onwards
# data_plot = da_x.sel(time=slice(1971, None))

# Top row: Level 1 materials
for t in types_level1:
    if t in data_plot.material.values:
        data_plot.sel(material=t).plot(ax=axes[0], label=t, color=dict_materials_colors[t])
axes[0].set_title("Level 1 Materials")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Magnitude")
axes[0].legend()

# Middle row: Level 2 materials
for t in types_level2:
    if t in data_plot.material.values:
        data_plot.sel(material=t).plot(ax=axes[1], label=t, color=dict_materials_colors[t])
    if t == "Cu":
        # Add IEA data for copper
        axes[1].scatter(df_iea_gen_cu.index, df_iea_gen_cu['Cu'], label="IEA Cu solar+wind", color='#00a5cf', s=4)

axes[1].set_title("Level 2 Materials")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Magnitude")
axes[1].legend(loc='upper left')

# Bottom row: Level 3 materials
for t in types_level3:
    if t in data_plot.material.values:
        data_plot.sel(material=t).plot(ax=axes[2], label=t, color=dict_materials_colors[t])
axes[2].set_title("Level 3 Materials")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Magnitude")
axes[2].legend(loc='upper left')

plt.suptitle("Generation - Inflow Materials - World", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Gen_inflow-materials_world.png", dpi=300)
# fig.savefig(path_test_plots / "Gen_inflow-materials_world_1971.png", dpi=300)
# fig.savefig(path_test_plots / "Gen_inflow-materials_world_1971.pdf", dpi=300)
# fig.savefig(path_test_plots / "Gen_inflow-materials_world_1971.svg", dpi=300)
plt.show()
















###########################################################################################################
###########################################################################################################
#%% 2) STORAGE
###########################################################################################################
###########################################################################################################



#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 2.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


# 1. External Data ======================================================================================== 


# read in the storage share in 2016 according to IEA (Technology perspectives 2017)
storage_IEA = pd.read_csv(path_external_data_standard / 'storage_IEA2016.csv', index_col=0)

# read in the storage costs according to IRENA storage report & other sources in the SI
storage_costs = pd.read_csv(path_external_data_standard / 'storage_cost.csv', index_col=0).transpose()

# read in the assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
storage_malus = pd.read_csv(path_external_data_standard / 'storage_malus.csv', index_col=0).transpose()

#read in the assumptions on the long-term price decline after 2050. Prices are in $ct / kWh electricity cycled (the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
storage_ltdecline = pd.Series(pd.read_csv(path_external_data_standard / 'storage_ltdecline.csv',index_col=0,  header=None).transpose().iloc[0])

#read in the energy density assumptions (kg/kWh storage capacity - mass required to store one unit of energy — more mass per energy = worse performance)
storage_density = pd.read_csv(path_external_data_standard / 'storage_density_kg_per_kwh.csv',index_col=0).transpose()

#read in the lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
storage_lifetime = pd.read_csv(path_external_data_standard / 'storage_lifetime.csv',index_col=0).transpose()

kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')   #annual car mileage in kms/yr, based  mostly  on  Pauliuk  et  al.  (2012a)

# material compositions (storage)
storage_materials = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv',index_col=[0,1]).transpose()  # wt% of total battery weight for various materials, total battery weight is given by the density file above

# Hydro-dam power capacity (also MW) within 5 regions reported by the IHS (international Hydropwer Association)
phs_projections = pd.read_csv(path_external_data_standard / 'PHS.csv', index_col='t')   # pumped hydro storage capacity (MW)


# 2. IMAGE/TIMER files ====================================================================================


# # The vehicle shares of trucks (heavy) of the SSP2
#     loadfactor_car_data: pd.DataFrame = read_mym_df(
#         image_folder.joinpath("trp_trvl_Load.out")). rename(
#         columns={
#             "DIM_1": "region"})

# read TIMER installed storage capacity (MWh, reservoir)
storage = read_mym_df(path_image_output.joinpath("StorResTot.out"))   #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
    
#storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)
storage_power = read_mym_df(path_image_output / 'StorCapTot.out')  

loadfactor_data = read_mym_df(path_image_output / 'trp_trvl_Load.out') 
  
passengerkms_data = read_mym_df(path_image_output / 'trp_trvl_pkm.out')   # passenger kilometers in Tera pkm

vehicleshare_data = read_mym_df(path_image_output / 'trp_trvl_Vshare_car.out')

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data = read_mym_df(path_image_output / 'Gcap.out') # needed to get hydro power for storage
gcap_data = gcap_data.iloc[:, :26]



#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%% 2.2) Prepare general variables
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

# Calculations used in both 'vehicle battery storage' and 'other storage'


##################
# Interpolations #
##################

storage = storage.iloc[:, :26]    # drop global total column and empty (27) column

if SENS_ANALYSIS == 'high_stor':
   storage_multiplier = storage
   for year in range(2021,2051):
        storage_multiplier.loc[year] = storage.loc[year] * (1 + (1/30*(year-2020)))
   for year in range(2051,YEAR_END+1):
        storage_multiplier.loc[year] = storage.loc[year] * 2


# turn index to integer for sorting during the next step
storage_costs.index = storage_costs.index.astype('int64')
storage_malus.index = storage_malus.index.astype('int64')
storage_density.index = storage_density.index.astype('int64')
storage_lifetime.index = storage_lifetime.index.astype('int64')

# to interpolate between 2018 and 2030, first create empty rows (NaN values) 
storage_start = storage_costs.first_valid_index()
storage_end =   storage_costs.last_valid_index()
for i in range(storage_start+1,storage_end):
    storage_costs = pd.concat([storage_costs, pd.DataFrame(index=[i])]) #, ignore_index=True
    storage_malus = pd.concat([storage_malus, pd.DataFrame(index=[i])])         # mind: the malus needs to be defined for the same years as the cost indications
    storage_density = pd.concat([storage_density, pd.DataFrame(index=[i])])     # mind: the density needs to be defined for the same years as the cost indications
    storage_lifetime = pd.concat([storage_lifetime, pd.DataFrame(index=[i])])   # mind: the lifetime needs to be defined for the same years as the cost indications
    
# then, do the actual interpolation on the sorted dataframes                                                    
storage_costs_interpol = storage_costs.sort_index(axis=0).interpolate(axis=0)#.index.astype('int64')
storage_malus_interpol = storage_malus.sort_index(axis=0).interpolate(axis=0)
storage_density_interpol = storage_density.sort_index(axis=0).interpolate(axis=0)  # density calculation continue with the material calculations
storage_lifetime_interpol = storage_lifetime.sort_index(axis=0).interpolate(axis=0)  # lifetime calculation continue with the material calculations

# energy density ---
# fix the energy density (kg/kwh) of storage technologies after 2030
for year in range(2030+1,YEAR_OUT+1):
    # storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.last_valid_index()], name=year))
    row = storage_density_interpol.loc[[storage_density_interpol.last_valid_index()]]
    row.index = [year]
    storage_density_interpol = pd.concat([storage_density_interpol, row])
# assumed fixed energy densities before 2018
for year in reversed(range(YEAR_SWITCH,storage_start)):
    # storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.first_valid_index()], name=year)).sort_index(axis=0)
    row = storage_density_interpol.loc[[storage_density_interpol.first_valid_index()]]
    row.index = [year]
    storage_density_interpol = pd.concat([storage_density_interpol, row]).sort_index(axis=0)

# storage material intensity ---
# Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, YEAR_OUT+1)), list(storage_materials.index)])
stor_materials_interpol = pd.DataFrame(index=index, columns=storage_materials.columns.levels[1])
# material intensities for storage
for cat in list(storage_materials.columns.levels[1]):
   stor_materials_1st   = storage_materials.loc[:,idx[storage_materials.columns[0][0],cat]]
   stor_materials_interpol.loc[idx[YEAR_FIRST_GRID ,:],cat] = stor_materials_1st.to_numpy()                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].min(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].min(),cat]].to_numpy()                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].max(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].max(),cat]].to_numpy()                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   stor_materials_interpol.loc[idx[:,:],cat] = stor_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

#############
# Lifetimes #
#############

# First the lifetime of storage technologies needs to be defined over time, before running the dynamic stock function
# before 2018
for year in reversed(range(YEAR_START,storage_start)):
    # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()], name=year)])
    row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()]])
    storage_lifetime_interpol.loc[year] = row.iloc[0]
# after 2030
for year in range(2030+1,YEAR_OUT+1):
    # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()], name=year)])
    row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()]])
    storage_lifetime_interpol.loc[year] = row.iloc[0]

storage_lifetime_interpol = storage_lifetime_interpol.sort_index(axis=0)
# drop the PHS from the interpolated lifetime frame, as the PHS is calculated separately
storage_lifetime_interpol = storage_lifetime_interpol.drop(columns=['PHS'])

#################
# Market Shares #
#################

# Determine MARKET SHARE of the storage capacity using a multi-nomial logit function

# storage costs ---
# determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
decline = ((storage_costs_interpol.loc[storage_start,:]-storage_costs_interpol.loc[storage_end,:])/(storage_end-storage_start))/storage_costs_interpol.loc[storage_start,:]
decline_used = decline*storage_ltdecline #TODO: what is happening here? Why?

storage_costs_new = storage_costs_interpol * storage_malus_interpol
# calculate the development from 2030 to 2050 (using annual price decline)
for year in range(storage_end+1,2050+1):
    # print(year)
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()]*(1-decline_used), name=year))
    # storage_costs_new = pd.concat([storage_costs_new, pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used), name=year)])
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used)])
    storage_costs_new.loc[year] = row.iloc[0]
# for historic price development, assume 2x AVERAGE annual price decline on all technologies, except lead-acid (so that lead-acid gets a relative price advantage from 1970-2018)
for year in reversed(range(YEAR_START,storage_start)):
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean())), name=year)).sort_index(axis=0)
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean()))])
    storage_costs_new.loc[year] = row.iloc[0]

storage_costs_new.sort_index(axis=0, inplace=True) 
storage_costs_new.loc[1971:2017,'Deep-cycle Lead-Acid'] = storage_costs_new.loc[2018,'Deep-cycle Lead-Acid']        # restore the exception (set to constant 2018 values)


# market shares ---
# use the storage price development in the logit model to get market shares
storage_market_share = MNLogit(storage_costs_new, -0.2) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)

# fix the market share of storage technologies after 2050
for year in range(2050+1,YEAR_OUT+1):
    # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
    row = pd.DataFrame([storage_market_share.loc[storage_market_share.last_valid_index()]])
    storage_market_share.loc[year] = row.iloc[0]


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%% 2.3) Vehicles
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

# Vehicle specific variables

# kilometrage is defined until 2008, fill 2008 values until 2100 
kilometrage = kilometrage.reindex(list(range(YEAR_START,YEAR_END))).interpolate(limit_direction='both')
region_list = list(kilometrage.columns.values)                          # get a list with region names

loadfactor = loadfactor_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1') # loadfactor for cars (in persons per vehicle)
loadfactor = loadfactor.loc[list(range(YEAR_START,YEAR_END+1))] * LOAD_FACTOR  # car loadfactor is expressed compared to an average global loadfactor from the IMAGE-TIMER-TRAVEL model (original: Girod, 2012; further elaborations by Edelenbosch et al.) 
  
indexNames = passengerkms_data[ passengerkms_data['DIM_1'] >= 27 ].index
passengerkms_data.drop(indexNames , inplace=True)
passengerkms = passengerkms_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1').loc[list(range(YEAR_START,YEAR_END+1))]
  
BEV_collist = [22, 23, 24, 25] # battery electric vehicles
PHEV_collist= [21, 20, 19, 18, 17, 16] # plug-in hybrid electric vehicles

vehicleshare_data['battery'] = vehicleshare_data[BEV_collist].sum(axis=1)
vehicleshare_data['PHEV'] = vehicleshare_data[PHEV_collist].sum(axis=1)

# get the regional sum of the BEV & PHEV fraction of the fleet, and replace the column names with the region names
PHEV_share = vehicleshare_data[['time','DIM_1', 'PHEV']].pivot_table(index='time',columns='DIM_1').loc[list(range(YEAR_START,YEAR_END+1))]
BEV_share = vehicleshare_data[['time','DIM_1', 'battery']].pivot_table(index='time',columns='DIM_1').loc[list(range(YEAR_START,YEAR_END+1))]

# storage.drop(storage.iloc[:, -2:], inplace = True, axis = 1)    # drop global total column and empty (27) column

# before we run any calculations, replace the column names to region names
PHEV_share.columns = region_list
BEV_share.columns = region_list
passengerkms.columns = region_list
loadfactor.columns = region_list
storage.columns = region_list



# BEV & PHEV vehicle stats

if SENS_ANALYSIS == 'high_stor':
   capacity_usable_PHEV = 0.025   # 2.5% of capacity of PHEV is usable as storage (in the pessimistic sensitivity variant)
   capacity_usable_BEV  = 0.05    # 5  % of capacity of BEVs is usable as storage (in the pessimistic sensitivity variant)
else: 
   capacity_usable_PHEV = 0.05    # 5% of capacity of PHEV is usable as storage
   capacity_usable_BEV  = 0.10    # 10% of capacity of BEVs is usable as storage

vehicle_kms = passengerkms.loc[:YEAR_OUT] * PKMS_TO_VKMS / loadfactor.loc[:YEAR_OUT]        # conversion from tera-Tkms  
vehicles_all = vehicle_kms / kilometrage.loc[:YEAR_OUT]

vehicles_PHEV = vehicles_all * PHEV_share.loc[:YEAR_OUT]
vehicles_BEV = vehicles_all * BEV_share.loc[:YEAR_OUT]
vehicles_EV = vehicles_PHEV + vehicles_BEV

# For the availability of vehicle store capacity we apply the assumption of fixed weight,
# So we first need to know the average density of the stock of EV batteries
# to get there we need to know the share of the technologies in the stock (based on lifetime & share in the inflow/purchases)
# to get there, we first need to know the market share of new stock additions (=inflow/purchases)

#First we calculate the share of the inflow using only a few of the technologies in the storage market share
#The selection represents only the batteries that are suitable for EV & mobile applications
EV_battery_list = ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic ', 'Lithium-air']
#normalize the selection of market shares, so that total market share is 1 again (taking the relative share in the selected battery techs)
market_share_EVs = pd.DataFrame().reindex_like(storage_market_share[EV_battery_list])

for year in list(range(YEAR_START,YEAR_OUT+1)):
    for tech in EV_battery_list:
        market_share_EVs.loc[year, tech] = storage_market_share[EV_battery_list].loc[year,tech] /storage_market_share[EV_battery_list].loc[year].sum()


test = market_share_EVs.sum(axis=1) # -> market share sums to 1 for each year

###########################################################################################################
#%%% 2.3.1) Prep_data File
###########################################################################################################


ureg = pint.UnitRegistry(force_ndarray_like=True)
# Define the units for each dimension
unit_mapping = {
    'time': ureg.year,
    'year': ureg.year,
    'Year': ureg.year,
    'kg': ureg.kilogram,
    'yr': ureg.year,
    '%': ureg.percent,
    't': ureg.tonne,
    'MW': ureg.megawatt, #added
    'GW': ureg.gigawatt, #added
}

# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "gcap_stock": (["Time"], ["Type", "Region"],),
    "gcap_types_materials": (["Cohort"], ["Type", "material"],)
    # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
}

# results_dict = {
#         'total_nr_vehicles_simple': total_nr_vehicles_simple,
#         'material_fractions_simple': material_fractions_simple,
#         'material_fractions_typical': material_fractions_typical,
#         'vehicle_weights_simple': vehicle_weights_simple,
#         'vehicle_weights_typical': vehicle_weights_typical,
#         'lifetimes': lifetimes_vehicles,
#         'battery_weights_typical': battery_weights_typical,
#         'battery_materials': battery_materials,
#         'battery_shares': battery_shares,
#         'weight_boats': weight_boats,
#         'vehicle_shares_typical': vehicle_shares_typical
#     }

results_dict = {
        'gcap_stock': gcap_stock,
        'gcap_types_materials': gcap_types_materials,
        'gcap_lifetime_distr': gcap_lifetime_distr,
}

# df = gcap_materials_interpol.copy()


# Convert the DataFrames to xarray Datasets and apply units
preprocessing_results_xarray = {}


for df_name, df in results_dict.items():
    if df_name in conversion_table:
        data_xar_dataset = pandas_to_xarray(df, unit_mapping)
        data_xarray = dataset_to_array(data_xar_dataset, *conversion_table[df_name])
    else:
        # lifetimes_vehicles does not need to be converted in the same way.
        data_xarray = pandas_to_xarray(df, unit_mapping)
    preprocessing_results_xarray[df_name] = data_xarray



preprocessing_results_xarray["lifetimes"] = convert_life_time_vehicles(preprocessing_results_xarray["gcap_lifetime_distr"])
preprocessing_results_xarray["stocks"] = preprocessing_results_xarray.pop("gcap_stock")
preprocessing_results_xarray["material_intensities"] = preprocessing_results_xarray.pop("gcap_types_materials")
# preprocessing_results_xarray["shares"] = preprocessing_results_xarray.pop("vehicle_shares")




###########################################################################################################
#%%% 2.3.2) Stock Modelling
###########################################################################################################
# TODO: move this to electricity.py

# then we use that market share in combination with the stock developments to derive the stock share 
# Here we use the vehcile stock (number of cars) as a proxy for the development of the battery stock (given that we're calculating the actual battery stock still, and just need to account for the dynamics of purchases to derive te stock share here) 
EV_inflow_by_tech, EV_stock_cohorts, EV_outflow_cohorts = stock_share_calc(vehicles_EV, market_share_EVs, 'NiMH', ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic ', 'Lithium-air'])







###########################################################################################################
#%% 2.4) Hydro Power & Other Storage
###########################################################################################################

# Take the TIMER Hydro-dam capacity (MW) & compare it to Pumped hydro capacity (MW) projections from the International Hydropower Association

Gcap_hydro = gcap_data[['time','DIM_1', 7]].pivot_table(index='time', columns='DIM_1')   # IMAGE-TIMER Hydro dam capacity (power, in MW)
Gcap_hydro = Gcap_hydro.iloc[:, :26]
Gcap_hydro.columns = region_list
Gcap_hydro = Gcap_hydro.loc[:YEAR_OUT]

#storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)              
storage_power.drop(storage_power.iloc[:, -2:], inplace = True, axis = 1)    
storage_power.columns = region_list
storage_power = storage_power.loc[:YEAR_OUT]

#Disaggregate the Pumped hydro-storgae projections to 26 IMAGE regions according to the relative Hydro-dam power capacity (also MW) within 5 regions reported by the IHS (international Hydropwer Association)
phs_regions = [[10,11],[19],[1],[22],[0,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,20,21,23,24,25]]   # subregions in IHS data for Europe, China, US, Japan, RoW, MIND: region refers to IMAGE region MINUS 1
phs_projections_IMAGE = pd.DataFrame(index=Gcap_hydro.index, columns=Gcap_hydro.columns)        # empty dataframe

for column in range(0,len(phs_regions)):
    sum_data = Gcap_hydro.iloc[:,phs_regions[column]].sum(axis=1)                               # first, get the sum of all hydropower in the IHS regions (to divide over in second step)
    for region in range(0,REGIONS):
        if region in phs_regions[column]:
            phs_projections_IMAGE.iloc[:,region] = phs_projections.iloc[:,column] * (Gcap_hydro.iloc[:,region]/sum_data)

# Then fill the years after 2030 (end of IHS projections) according to the Gcap annual growth rate (assuming a fixed percentage of Hydro dams will be built with Pumped hydro capabilities after )
if SENS_ANALYSIS == 'high_stor':
   phs_projections_IMAGE.loc[2030:YEAR_OUT] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:YEAR_OUT]/Gcap_hydro.loc[2030:YEAR_OUT])  # no growth after 2030 in the high_stor sensitivity variant
else:
   phs_projections_IMAGE.loc[2030:YEAR_OUT] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:YEAR_OUT]/Gcap_hydro.loc[2030])

# Calculate the fractions of the storage capacity that is provided through pumped hydro-storage, electric vehicles or other storage (larger than 1 means the capacity superseeds the demand for energy storage, in terms of power in MW or enery in MWh) 
phs_storage_fraction = phs_projections_IMAGE.divide(storage_power.loc[:YEAR_OUT]).clip(upper=1)      # the phs storage fraction deployed to fulfill storage demand, both phs & storage_power here are expressed in MW
storage_remaining = storage.loc[:YEAR_OUT] * (1 - phs_storage_fraction)

if SENS_ANALYSIS == 'high_stor':
   oth_storage_fraction = 0.5 * storage_remaining 
   oth_storage_fraction += ((storage_remaining * 0.5) - storage_vehicles).clip(lower=0)    
   oth_storage_fraction = oth_storage_fraction.divide(storage).where(oth_storage_fraction > 0, 0).clip(lower=0) 
   evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
else: 
   oth_storage_fraction = (storage_remaining - storage_vehicles).clip(lower=0)    
   oth_storage_fraction = oth_storage_fraction.divide(storage.loc[:YEAR_OUT]).where(oth_storage_fraction > 0, 0).clip(lower=0)      
   evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
   
checksum = phs_storage_fraction + evs_storage_fraction + oth_storage_fraction   # should be 1 for all fields

# absolute storage capacity (MWh)
phs_storage_theoretical = phs_projections_IMAGE.divide(storage_power) * storage.loc[:YEAR_OUT]       # theoretically available PHS storage (MWh; fraction * total) only used in the graphs that show surplus capacity
phs_storage = phs_storage_fraction * storage.loc[:YEAR_OUT]
evs_storage = evs_storage_fraction * storage.loc[:YEAR_OUT]
oth_storage = oth_storage_fraction * storage.loc[:YEAR_OUT]

#output for Main text figure 2 (storage reservoir, in MWh for 3 storage types)
storage_out_phs = pd.concat([phs_storage], keys=['phs'], names=['type']) 
storage_out_evs = pd.concat([evs_storage], keys=['evs'], names=['type']) 
storage_out_oth = pd.concat([oth_storage], keys=['oth'], names=['type']) 
storage_out = pd.concat([storage_out_phs, storage_out_evs, storage_out_oth])
storage_out.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test'  / 'storage_by_type_MWh.csv')        # in MWh

# derive inflow & outflow (in MWh) for PHS, for later use in the material calculations 
PHS_kg_perkWh = 26.8   # kg per kWh storage capacity (as weight addition to existing hydro plants to make them pumped) 
phs_storage_stock_tail   = stock_tail(phs_storage.astype(float))
storage_lifetime_PHS = storage_lifetime['PHS'].reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1)), axis=0).interpolate(limit_direction='both')


###########################################################################################################
#%%% 2.4.1) Prep_data File
###########################################################################################################



###########################################################################################################
#%%% 2.4.2) Stock Modelling
###########################################################################################################
# TODO: move this to electricity.py


# Next step: stock modelling
# phs_storage_inflow, phs_storage_outflow, phs_storage_stock  = inflow_outflow(phs_storage_stock_tail, storage_lifetime_PHS, stor_materials_interpol.loc[idx[:,:],'PHS'].unstack() * PHS_kg_perkWh * 1000, 'PHS')    # PHS lifetime is fixed at 60 yrs anyway so, we simply select 1 value
# inflow_by_tech, stock_cohorts, outflow_cohorts = stock_share_calc(oth_storage, storage_market_share, 'Deep-cycle Lead-Acid', list(storage_lifetime_interpol.columns)) # run the function that calculates stock shares from total stock & inflow shares























###########################################################################################################
###########################################################################################################
#%% 3) TRANSMISSION GRID 
###########################################################################################################
###########################################################################################################


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


# 1. External Data ======================================================================================== 

grid_length_Hv = pd.read_csv(path_external_data_standard /'grid_length_Hv.csv', index_col=0, names=None).transpose()    # lenght of the High-voltage (Hv) lines in the grid, based on Open Street Map (OSM) analysis (km)
ratio_Hv = pd.read_csv(path_external_data_standard / 'Hv_ratio.csv', index_col=0)                                        # Ratio between the length of Medium-voltage (Mv) and Low-voltage (Lv) lines in relation to Hv lines (km Lv /km Hv) & (km Mv/ km Hv)
underground_ratio = pd.read_csv(path_external_data_standard / 'underground_ratio.csv', index_col=[0,1])           # these contain the definition of the constants in the linear function used to determine the relation between income & the percentage of underground power-lines (% underground = mult * gdp/cap + add)
grid_additions = pd.read_csv(path_external_data_standard / 'grid_additions.csv', index_col=0)                            # Transformers & substations per km of grid (Hv, Mv & Lv, in units/km)


# dynamic or scenario-dependent data (lifetimes & material intensity)

lifetime_grid_elements = pd.read_csv(path_external_data_scenario  / 'operational_lifetime_grid.csv', index_col=0)         # Average lifetime in years of grid elements

# dynamic material intensity files (kg/km or kg/unit)
materials_grid = pd.read_csv(path_external_data_scenario / 'Materials_grid_dynamic.csv', index_col=[0,1])                # Material intensity of grid lines specific material content for Hv, Mv & Lv lines, & specific for underground vs. aboveground lines. (kg/km)
materials_grid_additions = pd.read_csv(path_external_data_scenario / 'Materials_grid_additions.csv', index_col=[0,1])    # (not part of the SA yet) Additional infrastructure required for grid connections, such as transformers & substations (material compositin in kg/unit)

# IMAGE file: GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
gdp_pc = pd.read_csv(path_external_data_scenario / 'gdp_pc.csv', index_col=0)  # TODO: check why it says this is an IMAGE file (why is it .csv?)


# 2. IMAGE/TIMER files ====================================================================================

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data = read_mym_df(path_image_output / 'Gcap.out')
# gcap_BL_data = read_mym_df('SSP2\\SSP2_BL\\Gcap.out') # baseline scenario? TODO: what is the purpose of reading in the scneario + the baseline?
gcap_BL_data = read_mym_df(path_image_output / 'Gcap.out')


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.2) Prepare model specific variables
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

gcap_BL_data = gcap_BL_data.loc[~gcap_BL_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_BL = pd.pivot_table(gcap_BL_data[gcap_BL_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

region_list = list(grid_length_Hv.columns.values)
material_list = list(materials_grid.columns.values)

# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap_techlist = ['Solar PV', 'Solar Decentral', 'CSP', 'Wind onshore', 'Wind offshore', 'Wave', 'Hydro', 'Other Renewables', 'Geothermal', 'Hydrogen', 'Nuclear', '<EMPTY>', 'Conv. Coal', 'Conv. Oil', 'Conv. Natural Gas', 'Waste', 'IGCC', 'OGCC', 'NG CC', 'Biomass CC', 'Coal + CCS', 'Oil/Coal + CCS', 'Natural Gas + CCS', 'Biomass + CCS', 'CHP Coal', 'CHP Oil', 'CHP Natural Gas', 'CHP Biomass', 'CHP Coal + CCS', 'CHP Oil + CCS', 'CHP Natural Gas + CCS', 'CHP Biomass + CCS', 'CHP Geothermal', 'CHP Hydrogen']
gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
gcap_BL.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
gcap.columns = gcap_techlist
gcap_BL.columns = gcap_techlist

gdp_pc.columns = region_list
# gdp_pc = gdp_pc.drop([1970]).drop(list(range(YEAR_END+1,YEAR_LAST+1)))


# length calculations ----------------------------------------------------------------------------

# only the regional total (peak) generation capacity is used as a proxy for the grid growth (BL to 2016, then BL or 450)
gcap_BL_total = gcap_BL.sum(axis=1).unstack()
gcap_BL_total = gcap_BL_total[region_list]               # re-order columns to the original TIMER order
gcap_growth = gcap_BL_total / gcap_BL_total.loc[2016]    # define growth according to 2016 as base year
gcap_total = gcap.sum(axis=1).unstack()
gcap_total = gcap_total[region_list]                     # re-order columns to the original TIMER order
gcap_growth.loc[2016:YEAR_END] = gcap_total.loc[2016:YEAR_END] / gcap_total.loc[2016]        # define growth according to 2016 as base year

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
grid_length_Mv = grid_length_Hv.mul(ratio_Hv['HV to MV'])
grid_length_Lv = grid_length_Hv.mul(ratio_Hv['HV to LV'])

# define grid length over time (fixed in 2016, growth according to gcap)
grid_length_Hv_time = pd.DataFrame().reindex_like(gcap_total)
grid_length_Mv_time = pd.DataFrame().reindex_like(gcap_total)
grid_length_Lv_time = pd.DataFrame().reindex_like(gcap_total)

#implement growth correction (sensitivity variant)
if SENS_ANALYSIS == 'high_grid':
   gcap_growth_HV = gcap_growth.add(add_growth.reindex_like(gcap_growth)).subtract(red_growth.reindex_like(gcap_growth))
else: 
   gcap_growth_HV = gcap_growth

for year in range(YEAR_START, YEAR_END+1):
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
    function_Hv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'HV'] + underground_ratio.loc[idx[select_proxy,'add'],'HV']
    function_Mv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'MV'] + underground_ratio.loc[idx[select_proxy,'add'],'MV']
    function_Lv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'LV'] + underground_ratio.loc[idx[select_proxy,'add'],'LV']

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
grid_length_HV_out.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test' / 'grid_length_HV_km.csv') # in km

grid_length_Mv_above = grid_length_Mv_time * (1 - function_Mv_under/100)
grid_length_Mv_under = grid_length_Mv_time * function_Mv_under/100
grid_length_Mv_total = grid_length_Mv_above + grid_length_Mv_under

grid_length_Lv_above = grid_length_Lv_time * (1 - function_Lv_under/100)
grid_length_Lv_under = grid_length_Lv_time * function_Lv_under/100
grid_length_Lv_total = grid_length_Lv_above + grid_length_Lv_under

grid_subst_Hv = grid_length_Hv_total.mul(grid_additions.loc['Substations','HV'])        # number of substations on HV network
grid_subst_Mv = grid_length_Mv_total.mul(grid_additions.loc['Substations','MV'])        # # of substations
grid_subst_Lv = grid_length_Lv_total.mul(grid_additions.loc['Substations','LV'])        # # of substations
grid_trans_Hv = grid_length_Hv_total.mul(grid_additions.loc['Transformers','HV'])       # number of transformers on the HV network
grid_trans_Mv = grid_length_Mv_total.mul(grid_additions.loc['Transformers','MV'])       # # of transformers
grid_trans_Lv = grid_length_Lv_total.mul(grid_additions.loc['Transformers','LV'])       # # of transformers




##################
# Interpolations #
##################

# Interpolate material intensities (dynamic content from 1926 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID,YEAR_END+1)), list(materials_grid.index.levels[1])])
materials_grid_interpol = pd.DataFrame(index=index, columns=materials_grid.columns)
materials_grid_additions_interpol = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID,YEAR_END+1)), list(materials_grid_additions.index.levels[1])]), columns=materials_grid_additions.columns)

for cat in list(materials_grid.index.levels[1]):
   materials_grid_1st   = materials_grid.loc[idx[materials_grid.index[0][0], cat],:]
   materials_grid_interpol.loc[idx[YEAR_FIRST_GRID ,cat],:] = materials_grid_1st                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   materials_grid_interpol.loc[idx[materials_grid.index.levels[0].min(),cat],:] = materials_grid.loc[idx[materials_grid.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   materials_grid_interpol.loc[idx[materials_grid.index.levels[0].max(),cat],:] = materials_grid.loc[idx[materials_grid.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   materials_grid_interpol.loc[idx[:,cat],:] = materials_grid_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(YEAR_FIRST_GRID,YEAR_END+1)), level=0).interpolate()

for cat in list(materials_grid_additions.index.levels[1]):
   materials_grid_additions_1st   = materials_grid_additions.loc[idx[materials_grid_additions.index[0][0], cat],:]
   materials_grid_additions_interpol.loc[idx[YEAR_FIRST_GRID ,cat],:] = materials_grid_additions_1st          # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   materials_grid_additions_interpol.loc[idx[materials_grid_additions.index.levels[0].min(),cat],:] = materials_grid_additions.loc[idx[materials_grid_additions.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   materials_grid_additions_interpol.loc[idx[materials_grid_additions.index.levels[0].max(),cat],:] = materials_grid_additions.loc[idx[materials_grid_additions.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   materials_grid_additions_interpol.loc[idx[:,cat],:] = materials_grid_additions_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(YEAR_FIRST_GRID,YEAR_END+1)), level=0).interpolate()


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


#############
# Lifetimes #
#############

# data only for lines, substations and transformer -> bring in knowledge_graph format: HV - Lines, MV - Lines, LV - Lines, HV - Transformers, etc.
expanded_data = {}
for typ in ["Lines", "Transformers", "Substations"]:
    for level in ["HV", "MV", "LV"]:
        new_col = f"{level} - {typ}"
        expanded_data[new_col] = lifetime_grid_elements[typ]
lifetime_grid_elements = pd.DataFrame(expanded_data, index=lifetime_grid_elements.index)
lifetime_grid_elements.rename_axis('Year', inplace=True)

# no differentiation between HV, MV & LV lines as well as between aboveground and belowground
# Types: lines, transformers, substations
lifetime_grid_elements.loc[YEAR_FIRST_GRID,:] = lifetime_grid_elements.loc[lifetime_grid_elements.first_valid_index(),:]
lifetime_grid_elements.loc[YEAR_OUT,:]         = lifetime_grid_elements.loc[lifetime_grid_elements.last_valid_index(),:]
lifetime_grid_elements                        = lifetime_grid_elements.reindex(list(range(YEAR_FIRST_GRID, YEAR_OUT+1))).interpolate()
# TODO: check why lifetime for lines is interpoltaed from 2020 - 40yrs to 2050 - 48 yrs and then back to 2060 - 40 yrs -> should stay at 48 yrs?


#######################################################################################################
#%%%% NEW

# lifetimes
df_mean = lifetime_grid_elements.copy()
df_stdev = df_mean * STD_LIFETIMES_ELECTR
df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
lifetime_grid_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns


# Materials

# harmonize units of MI -> then materials_grid and materials_grid_additions can be merged in one dataframe

# Grid Additions MIs ---
# for substations and transformer we multiply the MI (kg/unit of substation or transformer) with the number of substations or transformers per kilometer of grid length
materials_grid_additions_kgperkm = materials_grid_additions_to_kgperkm(materials_grid_additions_interpol, grid_additions)
# material intensities: (years, tech. type) index and materials as columns -> years as index and (tech. type, materials) as columns
materials_grid_additions_kgperkm.index.names = ["Year", "Type"]
materials_grid_additions_kgperkm = materials_grid_additions_kgperkm.unstack(level='Type') # bring tech. type from row index to column header
materials_grid_additions_kgperkm.columns = materials_grid_additions_kgperkm.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
materials_grid_additions_kgperkm = materials_grid_additions_kgperkm.sort_index(axis=1)
# rename columns to match knowledge graph
new_level_0 = [col.replace(' ', ' - ', 1) for col in materials_grid_additions_kgperkm.columns.get_level_values(0)]
new_columns = pd.MultiIndex.from_arrays([
    new_level_0,
    materials_grid_additions_kgperkm.columns.get_level_values(1)
], names=materials_grid_additions_kgperkm.columns.names)
materials_grid_additions_kgperkm.columns = new_columns

# Grid MIs ---
materials_grid_kgperkm = materials_grid_interpol.copy() # copy the interpolated material intensities
materials_grid_kgperkm.index.names = ["Year", "Type"]
materials_grid_kgperkm = materials_grid_kgperkm.unstack(level='Type') # bring tech. type from row index to column header
materials_grid_kgperkm.columns = materials_grid_kgperkm.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
materials_grid_kgperkm = materials_grid_kgperkm.sort_index(axis=1)
# rename columns to match knowledge graph
new_level_0 = [col.replace(' ', ' - Lines - ', 1) for col in materials_grid_kgperkm.columns.get_level_values(0)]
new_columns = pd.MultiIndex.from_arrays([
    new_level_0,
    materials_grid_kgperkm.columns.get_level_values(1)
], names=materials_grid_kgperkm.columns.names)
materials_grid_kgperkm.columns = new_columns

# print_df_info(materials_grid_kgperkm, "Materials Grid kg/km")
# print_df_info(materials_grid_additions_kgperkm, "Materials Grid Additions kg/km")

materials_grid_combined_kgperkm = pd.concat([materials_grid_kgperkm, materials_grid_additions_kgperkm], axis=1) # 


# Stocks

grid_dict = dict({
        'HV - Lines - Overhead': grid_length_Hv_above_new,
        'HV - Lines - Underground': grid_length_Hv_under_new,
        'HV - Substations': grid_subst_Hv_new,
        'HV - Transformers': grid_trans_Hv_new,
        'MV - Lines - Overhead': grid_length_Mv_above_new,
        'MV - Lines - Underground': grid_length_Mv_under_new,
        'MV - Substations': grid_subst_Mv_new,
        'MV - Transformers': grid_trans_Mv_new,
        'LV - Lines - Overhead': grid_length_Lv_above_new,
        'LV - Lines - Underground': grid_length_Lv_under_new,
        'LV - Substations': grid_subst_Lv_new,
        'LV - Transformers': grid_trans_Lv_new
    })
    
grid_stock = pd.concat(grid_dict, axis=1) # Concatenate with keys to create MultiIndex ('Name', 'Region')
# grid_stock.columns = grid_stock.columns.swaplevel(0, 1) # Swap levels so Region comes first
grid_stock = grid_stock.sort_index(axis=1)

print_df_info(grid_stock, "grid_stock")

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.3) Prep_data File
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


ureg = pint.UnitRegistry(force_ndarray_like=True)
# Define the units for each dimension
unit_mapping = {
    'time': ureg.year,
    'year': ureg.year,
    'Year': ureg.year,
    'kg': ureg.kilogram,
    'yr': ureg.year,
    '%': ureg.percent,
    't': ureg.tonne,
    'MW': ureg.megawatt, #added
    'GW': ureg.gigawatt, #added
}

# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "grid_stock": (["Time"], ["Type", "Region"],),
    "materials_grid_combined_kgperkm": (["Cohort"], ["Type", "material"],)
    # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
}


# results_dict = {
#         'total_nr_vehicles_simple': total_nr_vehicles_simple,
#         'material_fractions_simple': material_fractions_simple,
#         'material_fractions_typical': material_fractions_typical,
#         'vehicle_weights_simple': vehicle_weights_simple,
#         'vehicle_weights_typical': vehicle_weights_typical,
#         'lifetimes': lifetimes_vehicles,
#         'battery_weights_typical': battery_weights_typical,
#         'battery_materials': battery_materials,
#         'battery_shares': battery_shares,
#         'weight_boats': weight_boats,
#         'vehicle_shares_typical': vehicle_shares_typical
#     }
results_dict = {
        'grid_stock': grid_stock,
        'materials_grid_combined_kgperkm': materials_grid_combined_kgperkm,
        'lifetime_grid_distr': lifetime_grid_distr,
}

# df = gcap_materials_interpol.copy()


# Convert the DataFrames to xarray Datasets and apply units
preprocessing_results_xarray = {}


for df_name, df in results_dict.items():
    if df_name in conversion_table:
        data_xar_dataset = pandas_to_xarray(df, unit_mapping)
        data_xarray = dataset_to_array(data_xar_dataset, *conversion_table[df_name])
    else:
        # lifetimes_vehicles does not need to be converted in the same way.
        data_xarray = pandas_to_xarray(df, unit_mapping)
    preprocessing_results_xarray[df_name] = data_xarray



preprocessing_results_xarray["lifetimes"] = convert_life_time_vehicles(preprocessing_results_xarray["lifetime_grid_distr"])
preprocessing_results_xarray["stocks"] = preprocessing_results_xarray.pop("grid_stock")
preprocessing_results_xarray["material_intensities"] = preprocessing_results_xarray.pop("materials_grid_combined_kgperkm")
# preprocessing_results_xarray["shares"] = preprocessing_results_xarray.pop("vehicle_shares")

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.4) Run Stock Model New
###########################################################################################################
# TODO: move this to electricity.py
#----------------------------------------------------------------------------------------------------------



prep_data = preprocessing_results_xarray.copy()

# # Define the complete timeline, including historic tail
time_start = prep_data["stocks"].coords["Time"].min().values
# time_start = 1960
complete_timeline = prism.Timeline(time_start, 2060, 1)
simulation_timeline = prism.Timeline(1970, 2060, 1)

# Define the coordinates of all dimensions.
Region = list(prep_data["stocks"].coords["Region"].values)
Time = [t for t in complete_timeline]
Cohort = Time
Type = list(prep_data["stocks"].coords["Type"].values)
material = list(prep_data["material_intensities"].coords["material"].values)

# Create
# main_model_normal = GenericMainModel(
#     complete_timeline, Region=Region, Time=Time, Cohort=Cohort, Type=Type, prep_data=prep_data,
#     compute_materials=True, compute_battery_materials=False, compute_maintenance_materials=False, 
#     material=material)

new_prep_data = prep_data.copy()
new_prep_data["knowledge_graph"] = create_electricity_graph()
new_prep_data["shares"] = None

sec_elctr_trans = Sector("elctr_trans", new_prep_data)


main_model_factory = ModelFactory(
    sec_elctr_trans, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()


main_model_factory.simulate(simulation_timeline)
list(main_model_factory.elctr_trans)



path_test_plots = Path(path_base, "imagematerials", "electricity", "out_test", "Figures")

###########################################################################################################
#%%% 3.5) Visualize Stocks 


da_stocks = main_model_factory.stocks.copy()


# TOTAL STOCKS

da_stocks = da_stocks.sum(dim="Region")
types_top = da_stocks.Type.values[[0, 1, 4, 5, 8, 9]]   # Types 1–10
types_bottom = da_stocks.Type.values[[2,3,6,7,10,11]]  # Types 11–20

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True, sharey=True)
i=0
for t in types_top:
    axes[0].plot(da_stocks.sel(Type=t), label=t)
# axes.set_title(f"{region} (Types 1–10)")
axes[0].legend()

# Bottom row: Types 11–20
for t in types_bottom:
    axes[1].plot(da_stocks.sel(Type=t), label=t)
# axes.set_title(f"{region} (Types 11–20)")
axes[1].set_xlabel("Time")
axes[1].legend(loc ='upper left')

# Label only left side with y-axis label
axes[1].set_ylabel("Value")
axes[1].set_ylabel("Value")
plt.suptitle("Electricity Grid - Stocks", fontsize=16)

plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_stocks_world.png", dpi=300)
plt.show()


# # Two regions
# regions = da_stocks.Region.values[:2]  # First 2 regions
# types_top = da_stocks.Type.values[[0, 1, 4, 5, 8, 9]]   # Types 1–10
# types_bottom = da_stocks.Type.values[[2,3,6,7,10,11]]  # Types 11–20

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=True, sharey=True)

# for i, region in enumerate(regions):
#     # Top row: Types 1–10
#     for t in types_top:
#         da_stocks.sel(Type=t, Region=region).plot(ax=axes[0, i], label=t)
#     axes[0, i].set_title(f"{region} (Types 1–10)")
#     axes[0, i].legend()

#     # Bottom row: Types 11–20
#     for t in types_bottom:
#         da_stocks.sel(Type=t, Region=region).plot(ax=axes[1, i], label=t)
#     axes[1, i].set_title(f"{region} (Types 11–20)")
#     axes[1, i].set_xlabel("Time")
#     axes[1, i].legend(loc ='upper left')

# # Label only left side with y-axis label
# axes[0, 0].set_ylabel("Value")
# axes[1, 0].set_ylabel("Value")
# plt.suptitle("Electricity Grid - Stocks", fontsize=16)

# plt.tight_layout()
# # fig.savefig(path_test_plots / "Grid_stocks_Brazil-CEurope.png", dpi=300)
# plt.show()



###########################################################################################################
#%% Visualize INFLOW Materials - World per type Lines, Trans., Subst.


materials = ["Steel", "Concrete", "Aluminium", "Cu"]
dict_grid_colors = {
    #'Lines Overhead': 'FF9B85',
    #'Lines Underground': 'FFD97D',
    'Lines': '#8cb369', #'#007f5f',
    'Transformers': '#f4a259', #'#aacc00',
    'Substations': '#bc4b51' #'#55a630'
}

# data IEA ---------------------------------------------------------------------
# values for APS  scenario (NZE scenario)

# Cu -------------
years = np.arange(2012, 2051)
values = np.concatenate([
    np.full(10, 5e9),       # 2012–2021 # 5 Mt = 5*10e9 kg
    np.full(9, 5.5e9),       # 2022–2030
    np.full(10, np.nan),    # 2031–2040 (gap)
    np.full(10, 9e9),       # 2041–2050 (12)
])
df_iea_lt_cu = pd.DataFrame({ # lines & transformers, copper
    'Year': years,
    'Cu': values
})
df_iea_lt_cu.set_index('Year', inplace=True)

# Alu -------------
years = np.arange(2012, 2051)
values = np.concatenate([
    np.full(10, 12e9),       # 2012–2021 # 12 Mt = 12*10e9 kg
    np.full(9, 13e9),        # 2022–2030
    np.full(10, np.nan),    # 2031–2040 (gap)
    np.full(10, 21e9),       # 2041–2050 (27)
])
df_iea_lt_alu = pd.DataFrame({ # lines & transformers, aluminium
    'Year': years,
    'Aluminium': values
})
df_iea_lt_alu.set_index('Year', inplace=True)

# Steel -------------
years = np.arange(2012, 2051)
values = np.concatenate([
    np.full(10, 5e9),       # 2012–2021
    np.full(9, np.nan),     # 2022–2030
    np.full(10, 9e9),       # 2031–2040 (13)
    np.full(10, np.nan),    # 2041–2050
])
df_iea_t_steel = pd.DataFrame({ # transformers, steel
    'Year': years,
    'Steel': values
})
df_iea_t_steel.set_index('Year', inplace=True)
# ----------------------------------------------------------------------------------

da_x = main_model_factory.inflow_materials.to_array().sum('Region')
da_x = da_x.sel(time=slice(1971, None))
da_x_sum = da_x.sum(dim='Type')

lines_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10), sharex=True)

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes[i], color = dict_grid_colors['Lines'], label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes[i], color = dict_grid_colors['Transformers'], label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes[i], color = dict_grid_colors['Substations'], label="Substations")
    da_x_sum.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--', linewidth=3)

    if mat == "Cu":
        # Add IEA data for copper
        axes[i].plot(df_iea_lt_cu.index, df_iea_lt_cu['Cu'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    if mat == "Aluminium":
        # Add IEA data for aluminium
        axes[i].plot(df_iea_lt_alu.index, df_iea_lt_alu['Aluminium'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    if mat == "Steel":
        # Add IEA data for steel
        axes[i].plot(df_iea_t_steel.index, df_iea_t_steel['Steel'], label="IEA T", color='#00a5cf', linestyle=':', linewidth=4)
    
    axes[i].set_title(f"{mat}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Inflow [kg]")
    axes[i].legend()
    axes[i].grid(alpha=0.3, linestyle='--')

axes[-1].set_xlabel("Time")

plt.suptitle("Electricity Grid Inflow Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_inflow-materials_world.svg")
fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.pdf")
fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.png")
fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.svg")
plt.show()



###########################################################################################################
#%% Visualize STOCK Materials - World per type Lines, Trans., Subst.



materials = ["Steel", "Concrete", "Aluminium", "Cu"]

da_x = main_model_factory.stock_by_cohort_materials.sum('Region')
da_x_sum = main_model_factory.stock_by_cohort_materials.sum('Region').sum(dim='Type')

lines_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=len(materials), ncols=1, figsize=(12, 10), sharex=True)

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes[i], label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes[i], label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes[i], label="Substations")
    da_x_sum.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--')
    
    axes[i].set_title(f"{mat}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Stock [kg]")
    axes[i].legend()
    axes[i].grid(alpha=0.3, linestyle='--')

axes[-1].set_xlabel("Time")

plt.suptitle("Electricity Grid Stocks", fontsize=16)
plt.tight_layout()
fig.savefig(path_test_plots / "Grid_stocks-bulkmaterials_per-type_world.pdf", dpi=300)
plt.show()

#-------------

materials = ["Aluminium"]

da_x = main_model_factory.stock_by_cohort_materials.sum('Region')
da_x_sum = main_model_factory.stock_by_cohort_materials.sum('Region').sum(dim='Type')

lines_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=len(materials), ncols=1, figsize=(8, 6), sharex=True)

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes, label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes, label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes, label="Substations")
    da_x_sum.sel(material=mat).plot(ax=axes, label="Total", color='red', alpha=0.8, linestyle='--')
    
    axes.set_title(f"{mat}")
    axes.set_xlabel(" ")
    axes.set_ylabel("Stock [kg]")
    axes.legend()
    axes.grid(alpha=0.3, linestyle='--')

axes.set_xlabel("Time")
axes.axvline(x=2050, color='grey', linestyle='--', alpha=0.5)

plt.suptitle("Electricity Grid Stocks", fontsize=16)
plt.tight_layout()
fig.savefig(path_test_plots / "Grid_stocks-Alu_per-type_world.pdf", dpi=300)
fig.savefig(path_test_plots / "Grid_stocks-Alu_per-type_world.png", dpi=300)
plt.show()


#------------

materials = ["Aluminium"]

fig, axes = plt.subplots(nrows=len(materials), ncols=1, figsize=(8, 4), sharex=True)

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes, label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes, label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes, label="Substations")
    da_x_sum.sel(material=mat).plot(ax=axes, label="Total", color='red', alpha=0.8, linestyle='--')
    
    axes.set_title(f"{mat}")
    axes.set_xlabel(" ")
    axes.set_ylabel("Stock [kg]")
    axes.legend()
    axes.grid(alpha=0.3, linestyle='--')

axes.set_xlabel("Time")
axes.set_xlim(1990, 2050)

plt.suptitle("Electricity Grid Stocks", fontsize=16)
plt.tight_layout()
fig.savefig(path_test_plots / "Grid_stocks-Alu_per-type_world_1990-2050.png",dpi=300)
plt.show()

###########################################################################################################
#%%%%

# da_x = main_model_factory.inflow.to_array()
da_x = main_model_factory.inflow_materials.to_array()
da_x = main_model_factory.inflow_materials.to_array().sum('Type')

regions = da_x.Region.values[:2]  # First 2 regions
# types_top = da_x.material.values[1:6]   # Types 1–10
# types_bottom = da_x.material.values[6:12]  # Types 11–20
types_level1 = [m for m in da_x.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in da_x.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in da_x.material.values if m not in (types_level1 + types_level2)]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=True)

axes[0, 1].sharey(axes[0, 0])
axes[1, 1].sharey(axes[1, 0])
axes[2, 1].sharey(axes[2, 0])

for i, region in enumerate(regions):
    # Top row: 
    for t in types_level1:
        da_x.sel(material=t, Region=region).plot(ax=axes[0, i], label=t)
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel("Time")
    axes[0, i].legend()

    # Middle row: 
    for t in types_level2:
        da_x.sel(material=t, Region=region).plot(ax=axes[1, i], label=t)
    axes[1, i].set_title(f"{region}")
    axes[1, i].set_xlabel("Time")
    axes[1, i].legend(loc ='upper left')

    # Bottom row: 
    for t in types_level3:
        da_x.sel(material=t, Region=region).plot(ax=axes[2, i], label=t)
    axes[2, i].set_title(f"{region}")
    axes[2, i].set_xlabel("Time")
    axes[2, i].legend(loc ='upper left')

# Label only left side with y-axis label
axes[0, 0].set_ylabel("Value")
axes[1, 0].set_ylabel("Value")
axes[2, 0].set_ylabel("Value")

plt.suptitle("Generation - Inflow Materials", fontsize=16)
plt.tight_layout()
fig.savefig(path_test_plots / "Gen_inflow-materials_Brazil-CEurope.png", dpi=300)
plt.show()




# # HV inflow/outflow (in kgs)
# Hv_lines_above_in, Hv_lines_above_out, Hv_lines_above_stock = inflow_outflow(grid_length_Hv_above_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'HV overhead'],:].droplevel(1))    
# Hv_lines_under_in, Hv_lines_under_out, Hv_lines_under_stock = inflow_outflow(grid_length_Hv_under_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'HV underground'],:].droplevel(1)) 
# Hv_subst_in, Hv_subst_out, Hv_subst_stock                   = inflow_outflow(grid_subst_Hv_new,        lifetime_grid_elements['substations'],  materials_grid_additions_interpol.loc[idx[:,'Hv Substation'],:].droplevel(1))              
# Hv_trans_in, Hv_trans_out, Hv_trans_stock                   = inflow_outflow(grid_trans_Hv_new,        lifetime_grid_elements['transformers'], materials_grid_additions_interpol.loc[idx[:,'Hv Transformer'],:].droplevel(1))            

# #MV inflow/outflow (in kgs)
# Mv_lines_above_in, Mv_lines_above_out, Mv_lines_above_stock = inflow_outflow(grid_length_Mv_above_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'MV overhead'],:].droplevel(1))  # For now, the same lifetime assumptions for HV elements are applied to MV & LV
# Mv_lines_under_in, Mv_lines_under_out, Mv_lines_under_stock = inflow_outflow(grid_length_Mv_under_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'MV underground'],:].droplevel(1))
# Mv_subst_in, Mv_subst_out, Mv_subst_stock                   = inflow_outflow(grid_subst_Mv_new,        lifetime_grid_elements['substations'],  materials_grid_additions_interpol.loc[idx[:,'Mv Substation'],:].droplevel(1))
# Mv_trans_in, Mv_trans_out, Mv_trans_stock                   = inflow_outflow(grid_trans_Mv_new,        lifetime_grid_elements['transformers'], materials_grid_additions_interpol.loc[idx[:,'Mv Transformer'],:].droplevel(1))

# #LV inflow/outflow (in kgs)
# Lv_lines_above_in, Lv_lines_above_out, Lv_lines_above_stock = inflow_outflow(grid_length_Lv_above_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'LV overhead'],:].droplevel(1))
# Lv_lines_under_in, Lv_lines_under_out, Lv_lines_under_stock = inflow_outflow(grid_length_Lv_under_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'LV underground'],:].droplevel(1))
# Lv_subst_in, Lv_subst_out, Lv_subst_stock                   = inflow_outflow(grid_subst_Lv_new,        lifetime_grid_elements['substations'],  materials_grid_additions_interpol.loc[idx[:,'Lv Substation'],:].droplevel(1))
# Lv_trans_in, Lv_trans_out, Lv_trans_stock                   = inflow_outflow(grid_trans_Lv_new,        lifetime_grid_elements['transformers'], materials_grid_additions_interpol.loc[idx[:,'Lv Transformer'],:].droplevel(1))










# %%
