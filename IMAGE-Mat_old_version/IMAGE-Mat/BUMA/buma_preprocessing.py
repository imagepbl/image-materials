#%% GENERAL SETTING & STATEMENTS
import pandas as pd
import numpy as np
import os
import math
from pathlib import Path

idx = pd.IndexSlice

from read_mym import read_mym_df

from constants_BUMA import(
    SCENARIO_SELECT, FILE_ADDITION,
    REGIONS, REGIONS_RANGE,
    START_YEAR, END_YEAR, HIST_YEAR, YEARS, YEAR_LIST_SVA,
    INFLATION, 
    FLAG_ALPHA, FLAG_EXPDEC,
    LOWCOMM,
    GOMPERTZ_EXPDEC,
    MINIMUM_COM
)

#%% # Reading all csv files for buildings that are external to IMAGE
base_dir = Path(os.getcwd())
files_db_data_path = base_dir.joinpath('files_DB\\' + SCENARIO_SELECT)
files_IMAGE = base_dir.joinpath('files_IMAGE\\' + SCENARIO_SELECT)

# 1) scenario independent data
# Avg_m2_cap; unit: m2/capita; meaning: average square meters per person (by region & rural/urban) 
avg_m2_cap: pd.DataFrame = pd.read_csv(base_dir.joinpath('files_DB\\'). joinpath('Average_m2_per_cap.csv')) 

# 1) scenario dependent data 
# Housing_type; unit: %; meaning: the share of the PEOPLE living in a particular building type (by region & by area) 
housing_type_new: pd.DataFrame = pd.read_csv(files_db_data_path. joinpath('Housing_type_dynamic.csv'), index_col = [0,1,2]) 
# Building_materials; unit: kg/m2; meaning: the average material use per square meter (by building type, by region & by area)
building_materials: pd.DataFrame = pd.read_csv(files_db_data_path. joinpath('Building_materials' + FILE_ADDITION + '.csv'), index_col = [0,1,2]) 
# 7 building materials in 4 commercial building types; unit: kg/m2; meaning: the average material use per square meter (by commercial building type)
materials_commercial: pd.DataFrame = pd.read_csv(files_db_data_path. joinpath('materials_commercial' + FILE_ADDITION + '.csv'), index_col = [0,1])  

# load IMAGE data-files (MyM file format)
floorspace: pd.DataFrame = read_mym_df(files_IMAGE.joinpath("res_Floorspace.out"))
floorspace = floorspace[['time','DIM_1',2,3]].rename(columns={"DIM_1": "Region", 'time':'t', 2:'Urban', 3:'Rural'})
#the other columns are average per capita floorspace per quintile (we also exclude the average per capita floorspace of the total population in column 1, because we use the urban & rural specific totals)
floorspace = floorspace[floorspace.Region != REGIONS + 1] #removing region 27
floorspace = floorspace[floorspace['t'].isin(list(range(START_YEAR, END_YEAR+1)))]
#remove all data beyond 2060 to save runtime, we have not yet generated scenario results beyond 2060

# Pop; unit: million of people; meaning: global population (over time, by region)             
pop: pd.DataFrame = pd.read_csv(files_IMAGE. joinpath('pop.csv'), index_col = [0]) 
# rurpop; unit: %; meaning: the share of people living in rural areas (over time, by region)
rurpop: pd.DataFrame = pd.read_csv(files_IMAGE. joinpath('rurpop.csv'), index_col = [0])
# we use the inflation corrected SVA to adjust for the fact that IMAGE provides gdp/cap in 2005 US$
sva_pc_2005: pd.DataFrame = pd.read_csv(files_IMAGE. joinpath('sva_pc.csv'), index_col = [0])
sva_pc = sva_pc_2005 * INFLATION

# load historic population development
hist_pop = pd.read_csv('files_initial_stock\hist_pop.csv', index_col = [0])  # initial population as a percentage of the 1970 population; unit: %; according to the Maddison Project Database (MPD) 2018 (Groningen University)

# Load fitted regression parameters
if FLAG_ALPHA == 0:
    gompertz = pd.read_csv('files_commercial/Gompertz_parameters.csv', index_col = [0])
else:
    gompertz = pd.read_csv('files_commercial/Gompertz_parameters_alpha.csv', index_col = [0])

#%% 
# added cubic interpolation to the sva_pc (presumed linear interpolation between 5-year original data caused sawtooth demand/inflow throughout the scenario projection after 2025)

sva_pc      = sva_pc.loc[YEAR_LIST_SVA,:].reindex(list(range(1970, END_YEAR + 1,1))).interpolate(method='cubic')      

# Interpolate population and rural population data (fills in missing years with cubic interpolation)
rurpop2 = rurpop.reindex(YEARS).interpolate(method='cubic')
pop2 = pop.reindex(YEARS).interpolate(method='cubic')

# also interpolate housing type data
index_ht = pd.MultiIndex.from_product([list(range(HIST_YEAR, END_YEAR + 1)), 
                                       list(range(1,REGIONS + 1)), 
                                       ['Urban', 'Rural'] ]) 
housing_type_new2 = pd.DataFrame(np.nan, index=index_ht, columns=housing_type_new.columns)

for year in list(housing_type_new.index.levels[0]):
    housing_type_new2.loc[idx[year,:,:],:] = housing_type_new.loc[idx[year,:,:],:]
    
for region in list(range(1,REGIONS + 1)):
    for area in ['Urban', 'Rural']:
        housing_types_interpolated = housing_type_new2.loc[idx[:,region,area],:].interpolate(method='linear', limit_direction='both')
        housing_type_new2.loc[idx[:,region,area],:] = housing_types_interpolated.values
        
# Remove 1st year, to ensure same Table size as floorspace data (from 1971)
pop2 = pop2.iloc[1:]
rurpop2 = rurpop2.iloc[1:]

#pre-calculate urban population
urbpop = 1 - rurpop2                                                           # urban population is 1 - the fraction of people living in rural areas (rurpop)
      
# Restructure the tables to regions as columns; for floorspace
floorspace_rur = floorspace.pivot(index="t", columns="Region", values="Rural")    # floorspace (m2) per capita
floorspace_urb = floorspace.pivot(index="t", columns="Region", values="Urban")

# Restructuring for square meters (m2/cap)
avg_m2_cap_urb = avg_m2_cap.loc[avg_m2_cap['Area'] == 'Urban'].drop('Area', 1).T  # Remove area column & Transpose
avg_m2_cap_urb.columns = list(map(int,avg_m2_cap_urb.iloc[0]))                      # name columns according to the row containing the region-labels
avg_m2_cap_urb2 = avg_m2_cap_urb.drop(['Region'])                                 # Remove idle row 

avg_m2_cap_rur = avg_m2_cap.loc[avg_m2_cap['Area'] == 'Rural'].drop('Area', 1).T  # Remove area column & Transpose
avg_m2_cap_rur.columns = list(map(int,avg_m2_cap_rur.iloc[0]))                      # name columns according to the row containing the region-labels
avg_m2_cap_rur2 = avg_m2_cap_rur.drop(['Region'])                                 # Remove idle row 

#%% COMMERCIAL building space demand (stock) calculated from Gompertz curve (fitted, using separate regression model)

# Select gompertz curve paramaters for the total commercial m2 demand (stock)
alpha, beta, gamma = (
    (gompertz['All']['a'], gompertz['All']['b'], gompertz['All']['c'])
    if FLAG_EXPDEC == 0 else GOMPERTZ_EXPDEC          
)
alpha_low = alpha * LOWCOMM                                    # alpha multiplied with a factor, lowering the maximum per capita commecrial floorspace between (2020 and 2050) 

# find the total commercial m2 stock (in Millions of m2)
commercial_m2_cap = pd.DataFrame(index=YEARS, columns=REGIONS_RANGE)
commercial_m2_cap_low = commercial_m2_cap.copy()

# Compute commercial floorspace using Gompertz curves
for year in YEARS:
    for region in REGIONS_RANGE:
        exp_factor = math.exp((-gamma/1000) * sva_pc[str(region)][year])
        if FLAG_EXPDEC == 0:
            commercial_m2_cap[region][year] = alpha * math.exp(-beta * exp_factor)
            commercial_m2_cap_low[region][year] = alpha_low * math.exp(-beta * exp_factor)
        else:
            commercial_m2_cap[region][year] = max(0.542, alpha - beta * exp_factor)

# commercial floorspace is scaled here (in case lowComm is not 1)
scale_comm = pd.Series([1.0, 0.0], index=[2020, 2060], name='time').reindex(YEARS).interpolate(method='linear', limit=300, limit_direction='both')
commercial_m2_cap    = commercial_m2_cap.mul(scale_comm, axis=0) + commercial_m2_cap_low.mul((1-scale_comm), axis=0)

# Subdivide the total across Offices, Retail+, Govt+ & Hotels+
types = ["Office", "Retail+", "Hotels+", "Govt+"]
index = pd.MultiIndex.from_product([types, REGIONS_RANGE, YEARS], names=["Type", "Region", "Year"])
commercial_m2_cap = pd.DataFrame(index=index, columns=["m2_per_cap"]).fillna(0)

#TODO move to a util file
# Define a function to calculate Gompertz growth
def gompertz_value(category, region, year, sva_data):
    """Calculate the Gompertz value for a given category, region, and year."""
    params = gompertz[category]
    return params['a'] * math.exp(-params['b'] * math.exp((-params['c'] / 1000) * sva_data[str(region)][year]))

for year in YEARS:
    for region in REGIONS_RANGE:
        # Calculate floorspace for all types and update the minimum values
        floorspace_commercial_list = {}
        for type_ in types:
            value = gompertz_value(type_, region, year, sva_pc)
            floorspace_commercial_list[type_] = value
            MINIMUM_COM[type_] = min(MINIMUM_COM[type_], value)

        # Sum all floorspace values for normalization
        commercial_sum = sum(floorspace_commercial_list.values())

        # Calculate and assign the floorspace for each type
        for type_ in types:
            commercial_m2_cap.loc[(type_, region, year), "m2_per_cap"] = (
                commercial_m2_cap.loc[(type_, region, year), "m2_per_cap"] * (floorspace_commercial_list[type_] / commercial_sum)
            )

        
#%% Add historic tail (1720-1970) + 100 yr initial --------------------------------------------

# Determine the historical average global trend in floorspace/cap  & the regional rural population share based on the last 10 years of IMAGE data
floorspace_urb_trend_by_region = [0 for j in range(0,26)]
floorspace_rur_trend_by_region = [0 for j in range(0,26)]
rurpop_trend_by_region = [0 for j in range(0,26)]
commercial_m2_cap_office_trend = [0 for j in range(0,26)]
commercial_m2_cap_retail_trend = [0 for j in range(0,26)]
commercial_m2_cap_hotels_trend = [0 for j in range(0,26)]
commercial_m2_cap_govern_trend = [0 for j in range(0,26)]

# For the RESIDENTIAL & COMMERCIAL floorspace: Derive the annual trend (in m2/cap) over the initial 10 years of IMAGE data
for region in range(1,27):
    floorspace_urb_trend_by_year = [0 for i in range(0,10)]
    floorspace_rur_trend_by_year = [0 for i in range(0,10)]
    commercial_m2_cap_office_trend_by_year = [0 for j in range(0,10)]    
    commercial_m2_cap_retail_trend_by_year = [0 for i in range(0,10)]   
    commercial_m2_cap_hotels_trend_by_year = [0 for j in range(0,10)]
    commercial_m2_cap_govern_trend_by_year = [0 for i in range(0,10)]
    
    # Get the growth by year (for the first 10 years)
    for year in range(1970,1980):
        floorspace_urb_trend_by_year[year-1970] = floorspace_urb[region][year+1]/floorspace_urb[region][year+2]
        floorspace_rur_trend_by_year[year-1970] = floorspace_rur[region][year+1]/floorspace_rur[region][year+2]
        commercial_m2_cap_office_trend_by_year[year-1970] = commercial_m2_cap_office[region][year+1]/commercial_m2_cap_office[region][year+2]
        commercial_m2_cap_retail_trend_by_year[year-1970] = commercial_m2_cap_retail[region][year+1]/commercial_m2_cap_retail[region][year+2] 
        commercial_m2_cap_hotels_trend_by_year[year-1970] = commercial_m2_cap_hotels[region][year+1]/commercial_m2_cap_hotels[region][year+2]
        commercial_m2_cap_govern_trend_by_year[year-1970] = commercial_m2_cap_govern[region][year+1]/commercial_m2_cap_govern[region][year+2]
        
    rurpop_trend_by_region[region-1] = ((1-(rurpop[str(region)][1980]/rurpop[str(region)][1970]))/10)*100
    floorspace_urb_trend_by_region[region-1] = sum(floorspace_urb_trend_by_year)/10
    floorspace_rur_trend_by_region[region-1] = sum(floorspace_rur_trend_by_year)/10
    commercial_m2_cap_office_trend[region-1] = sum(commercial_m2_cap_office_trend_by_year)/10
    commercial_m2_cap_retail_trend[region-1] = sum(commercial_m2_cap_retail_trend_by_year)/10
    commercial_m2_cap_hotels_trend[region-1] = sum(commercial_m2_cap_hotels_trend_by_year)/10
    commercial_m2_cap_govern_trend[region-1] = sum(commercial_m2_cap_govern_trend_by_year)/10

# Average global annual decline in floorspace/cap in %, rural: 1%; urban 1.2%;  commercial: 1.26-2.18% /yr   
floorspace_urb_trend_global = (1-(sum(floorspace_urb_trend_by_region)/26))*100              # in % decrease per annum
floorspace_rur_trend_global = (1-(sum(floorspace_rur_trend_by_region)/26))*100              # in % decrease per annum
commercial_m2_cap_office_trend_global = (1-(sum(commercial_m2_cap_office_trend)/26))*100    # in % decrease per annum
commercial_m2_cap_retail_trend_global = (1-(sum(commercial_m2_cap_retail_trend)/26))*100    # in % decrease per annum
commercial_m2_cap_hotels_trend_global = (1-(sum(commercial_m2_cap_hotels_trend)/26))*100    # in % decrease per annum
commercial_m2_cap_govern_trend_global = (1-(sum(commercial_m2_cap_govern_trend)/26))*100    # in % decrease per annum


# define historic floorspace (1820-1970) in m2/cap
floorspace_urb_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=floorspace_urb.columns)
floorspace_rur_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=floorspace_rur.columns)
rurpop_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=rurpop.columns)
pop_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=pop2.columns)
commercial_m2_cap_office_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_office.columns)
commercial_m2_cap_retail_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_retail.columns)
commercial_m2_cap_hotels_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_hotels.columns)
commercial_m2_cap_govern_1820_1970 = pd.DataFrame(index=range(1820,1971), columns=commercial_m2_cap_govern.columns)

# Find minumum or maximum values in the original IMAGE data (Just for residential, commercial minimum values have been calculated above)
minimum_urb_fs = floorspace_urb.values.min()    # Region 20: China
minimum_rur_fs = floorspace_rur.values.min()    # Region 20: China
maximum_rurpop = rurpop.values.max()            # Region 9 : Eastern Africa

# Calculate the actual values used between 1820 & 1970, given the trends & the min/max values
for region in range(1,REGIONS+1):
    for year in range(1820,1971):
        # MAX of 1) the MINimum value & 2) the calculated value
        floorspace_urb_1820_1970[region][year] = max(minimum_urb_fs, floorspace_urb[region][1971] * ((100-floorspace_urb_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        floorspace_rur_1820_1970[region][year] = max(minimum_rur_fs, floorspace_rur[region][1971] * ((100-floorspace_rur_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        commercial_m2_cap_office_1820_1970[region][year] = max(minimum_com_office, commercial_m2_cap_office[region][1971] * ((100-commercial_m2_cap_office_trend_global)/100)**(1971-year))  # single global value for average annual Decrease  
        commercial_m2_cap_retail_1820_1970[region][year] = max(minimum_com_retail, commercial_m2_cap_retail[region][1971] * ((100-commercial_m2_cap_retail_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        commercial_m2_cap_hotels_1820_1970[region][year] = max(minimum_com_hotels, commercial_m2_cap_hotels[region][1971] * ((100-commercial_m2_cap_hotels_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        commercial_m2_cap_govern_1820_1970[region][year] = max(minimum_com_govern, commercial_m2_cap_govern[region][1971] * ((100-commercial_m2_cap_govern_trend_global)/100)**(1971-year))  # single global value for average annual Decrease
        # MIN of 1) the MAXimum value & 2) the calculated value        
        rurpop_1820_1970[str(region)][year] = min(maximum_rurpop, rurpop[str(region)][1970] * ((100+rurpop_trend_by_region[region-1])/100)**(1970-year))  # average annual INcrease by region
        # just add the tail to the population (no min/max & trend is pre-calculated in hist_pop)        
        pop_1820_1970[str(region)][year] = hist_pop[str(region)][year] * pop[str(region)][1970]

urbpop_1820_1970 = 1 - rurpop_1820_1970

# To avoid full model setup in 1820 (all required stock gets built in yr 1) we assume another tail that linearly increases to the 1820 value over a 100 year time period, so 1720 = 0
floorspace_urb_1721_1820 = pd.DataFrame(index=range(HIST_YEAR, 1820), columns=floorspace_urb.columns)
floorspace_rur_1721_1820 = pd.DataFrame(index=range(HIST_YEAR,1820), columns=floorspace_rur.columns)
rurpop_1721_1820 = pd.DataFrame(index=range(HIST_YEAR, 1820), columns=rurpop.columns)
urbpop_1721_1820 = pd.DataFrame(index=range(HIST_YEAR, 1820), columns=urbpop.columns)
pop_1721_1820 = pd.DataFrame(index=range(HIST_YEAR, 1820), columns=pop2.columns)
commercial_m2_cap_office_1721_1820 = pd.DataFrame(index=range(HIST_YEAR, 1820), columns=commercial_m2_cap_office.columns)
commercial_m2_cap_retail_1721_1820 = pd.DataFrame(index=range(HIST_YEAR, 1820), columns=commercial_m2_cap_retail.columns)
commercial_m2_cap_hotels_1721_1820 = pd.DataFrame(index=range(HIST_YEAR, 1820), columns=commercial_m2_cap_hotels.columns)
commercial_m2_cap_govern_1721_1820 = pd.DataFrame(index=range(HIST_YEAR, 1820), columns=commercial_m2_cap_govern.columns)

for region in range(1,27):
    for time in range(HIST_YEAR, 1820):
        #                                                        MAX(0,...) Because of floating point deviations, leading to negative stock in some cases
        floorspace_urb_1721_1820[int(region)][time]            = max(0.0, floorspace_urb_1820_1970[int(region)][1820] - (floorspace_urb_1820_1970[int(region)][1820]/100)*(1820-time))
        floorspace_rur_1721_1820[int(region)][time]            = max(0.0, floorspace_rur_1820_1970[int(region)][1820] - (floorspace_rur_1820_1970[int(region)][1820]/100)*(1820-time))
        rurpop_1721_1820[str(region)][time]                    = max(0.0, rurpop_1820_1970[str(region)][1820] - (rurpop_1820_1970[str(region)][1820]/100)*(1820-time))
        urbpop_1721_1820[str(region)][time]                    = max(0.0, urbpop_1820_1970[str(region)][1820] - (urbpop_1820_1970[str(region)][1820]/100)*(1820-time))
        pop_1721_1820[str(region)][time]                       = max(0.0, pop_1820_1970[str(region)][1820] - (pop_1820_1970[str(region)][1820]/100)*(1820-time))
        commercial_m2_cap_office_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_office_1820_1970[region][1820] - (commercial_m2_cap_office_1820_1970[region][1820]/100)*(1820-time))
        commercial_m2_cap_retail_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_retail_1820_1970[region][1820] - (commercial_m2_cap_retail_1820_1970[region][1820]/100)*(1820-time))
        commercial_m2_cap_hotels_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_hotels_1820_1970[region][1820] - (commercial_m2_cap_hotels_1820_1970[region][1820]/100)*(1820-time))
        commercial_m2_cap_govern_1721_1820[int(region)][time]  = max(0.0, commercial_m2_cap_govern_1820_1970[region][1820] - (commercial_m2_cap_govern_1820_1970[region][1820]/100)*(1820-time))

# combine historic with IMAGE data here
rurpop_tail                     = rurpop_1820_1970.append(rurpop2, ignore_index=False)
urbpop_tail                     = urbpop_1820_1970.append(urbpop, ignore_index=False)
pop_tail                        = pop_1820_1970.append(pop2, ignore_index=False)
floorspace_urb_tail             = floorspace_urb_1820_1970.append(floorspace_urb, ignore_index=False)
floorspace_rur_tail             = floorspace_rur_1820_1970.append(floorspace_rur, ignore_index=False)
commercial_m2_cap_office_tail   = commercial_m2_cap_office_1820_1970.append(commercial_m2_cap_office, ignore_index=False)
commercial_m2_cap_retail_tail   = commercial_m2_cap_retail_1820_1970.append(commercial_m2_cap_retail, ignore_index=False)
commercial_m2_cap_hotels_tail   = commercial_m2_cap_hotels_1820_1970.append(commercial_m2_cap_hotels, ignore_index=False)
commercial_m2_cap_govern_tail   = commercial_m2_cap_govern_1820_1970.append(commercial_m2_cap_govern, ignore_index=False)

rurpop_tail                     = rurpop_1721_1820.append(rurpop_1820_1970.append(rurpop2, ignore_index=False), ignore_index=False)
urbpop_tail                     = urbpop_1721_1820.append(urbpop_1820_1970.append(urbpop, ignore_index=False), ignore_index=False)
pop_tail                        = pop_1721_1820.append(pop_1820_1970.append(pop2, ignore_index=False), ignore_index=False)
floorspace_urb_tail             = floorspace_urb_1721_1820.append(floorspace_urb_1820_1970.append(floorspace_urb, ignore_index=False), ignore_index=False)
floorspace_rur_tail             = floorspace_rur_1721_1820.append(floorspace_rur_1820_1970.append(floorspace_rur, ignore_index=False), ignore_index=False)
commercial_m2_cap_office_tail   = commercial_m2_cap_office_1721_1820.append(commercial_m2_cap_office_1820_1970.append(commercial_m2_cap_office, ignore_index=False), ignore_index=False)
commercial_m2_cap_retail_tail   = commercial_m2_cap_retail_1721_1820.append(commercial_m2_cap_retail_1820_1970.append(commercial_m2_cap_retail, ignore_index=False), ignore_index=False)
commercial_m2_cap_hotels_tail   = commercial_m2_cap_hotels_1721_1820.append(commercial_m2_cap_hotels_1820_1970.append(commercial_m2_cap_hotels, ignore_index=False), ignore_index=False)
commercial_m2_cap_govern_tail   = commercial_m2_cap_govern_1721_1820.append(commercial_m2_cap_govern_1820_1970.append(commercial_m2_cap_govern, ignore_index=False), ignore_index=False)

#%% SQUARE METER Calculations (requires dynamic stock model to disaggregate building types) -----------------------------------------------------------

# share in new construction (from Fishman 2021)
housing_type_rur_new = housing_type_new2.loc[idx[:,:,'Rural'],:].droplevel(2)
housing_type_urb_new = housing_type_new2.loc[idx[:,:,'Urban'],:].droplevel(2)

if flag_Normal == 0:
    lifetimes_DB   = pd.read_csv('files_lifetimes\\' + SCENARIO_SELECT + '\\lifetimes.csv', index_col = [0,1,2,3])   # Weibull parameter database for residential buildings (shape & scale parameters given by region, area & building-type)
    lifetimes_comm = pd.read_csv('files_lifetimes\\' + SCENARIO_SELECT + '\\lifetimes_comm.csv', index_col = [0,1])  # Weibull parameter database for commercial buildings (shape & scale parameters given by region, area & building-type)
else:
    lifetimes_DB = pd.read_csv('files_lifetimes\lifetimes_normal.csv')  # Normal distribution database (Mean & StDev parameters given by region, area & building-type, though only defined by region for now)

# interpolate lifetime data
lifetimes_comm_shape = lifetimes_comm['Shape'].unstack().reindex(list(range(HIST_YEAR, END_YEAR + 1))).interpolate(limit=300, limit_direction='both')
lifetimes_comm_scale = lifetimes_comm['Scale'].unstack().reindex(list(range(HIST_YEAR, END_YEAR + 1))).interpolate(limit=300, limit_direction='both')

lifetimes_shape = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(HIST_YEAR, END_YEAR + 1)), list(lifetimes_DB.index.levels[2]), list(lifetimes_DB.index.levels[3])]), columns=lifetimes_DB.index.levels[1])
lifetimes_scale = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(HIST_YEAR, END_YEAR + 1)), list(lifetimes_DB.index.levels[2]), list(lifetimes_DB.index.levels[3])]), columns=lifetimes_DB.index.levels[1])
for building in list(lifetimes_DB.index.levels[2]):
    for area in list(lifetimes_DB.index.levels[3]):
        lifetimes_shape.loc[idx[:,building,area],:] = lifetimes_DB['Shape'].unstack(level=1).loc[:,building,area].reindex(list(range(HIST_YEAR, END_YEAR + 1))).interpolate(method='linear', limit=300, limit_direction='both').values
        lifetimes_scale.loc[idx[:,building,area],:] = lifetimes_DB['Scale'].unstack(level=1).loc[:,building,area].reindex(list(range(HIST_YEAR, END_YEAR + 1))).interpolate(method='linear', limit=300, limit_direction='both').values


# calculte the total rural/urban population in millions (pop2 = millions of people, rurpop2 = % of people living in rural areas)
people_rur = pd.DataFrame(rurpop_tail.values * pop_tail.values, columns=pop_tail.columns.astype('int'), index=pop_tail.index)
people_urb = pd.DataFrame(urbpop_tail.values * pop_tail.values, columns=pop_tail.columns.astype('int'), index=pop_tail.index)

# re-calculate the total floorspace (IMAGE), including historic tails (in MILLIONS of m2)
m2_rur = floorspace_rur_tail.mul(people_rur.values)
m2_urb = floorspace_urb_tail.mul(people_urb.values)

# define the relative importance of housing type wthin the stock of floorspace 
# (not just accounting for the changing share of people living accross housing types, 
# but also acknowledging that some housing types typically involve a higher floorspace per capita)
# the avg_m2 (per capita) only affects the relative allocation of total IMAGE floorsapce over building types
relative_rur_det = avg_m2_cap_rur2.loc['1',:] * housing_type_rur_new['Detached'].unstack()
relative_rur_sem = avg_m2_cap_rur2.loc['2',:] * housing_type_rur_new['Semi-detached'].unstack()
relative_rur_app = avg_m2_cap_rur2.loc['3',:] * housing_type_rur_new['Appartment'].unstack()
relative_rur_hig = avg_m2_cap_rur2.loc['4',:] * housing_type_rur_new['High-rise'].unstack()
total_rur        = relative_rur_det + relative_rur_sem + relative_rur_app + relative_rur_hig

relative_urb_det = avg_m2_cap_urb2.loc['1',:] * housing_type_urb_new['Detached'].unstack()
relative_urb_sem = avg_m2_cap_urb2.loc['2',:] * housing_type_urb_new['Semi-detached'].unstack()
relative_urb_app = avg_m2_cap_urb2.loc['3',:] * housing_type_urb_new['Appartment'].unstack()
relative_urb_hig = avg_m2_cap_urb2.loc['4',:] * housing_type_urb_new['High-rise'].unstack()
total_urb        = relative_urb_det + relative_urb_sem + relative_urb_app + relative_urb_hig

stock_share_rur_det = relative_rur_det / total_rur
stock_share_rur_sem = relative_rur_sem / total_rur
stock_share_rur_app = relative_rur_app / total_rur
stock_share_rur_hig = relative_rur_hig / total_rur

stock_share_urb_det = relative_urb_det / total_urb
stock_share_urb_sem = relative_urb_sem / total_urb
stock_share_urb_app = relative_urb_app / total_urb
stock_share_urb_hig = relative_urb_hig / total_urb

# checksum (should be all 1's)
checksum_rur = stock_share_rur_det + stock_share_rur_sem + stock_share_rur_app + stock_share_rur_hig
checksum_urb = stock_share_urb_det + stock_share_urb_sem + stock_share_urb_app + stock_share_urb_hig

# All m2 by region (in millions), Building_type & year (using the correction factor, to comply with IMAGE avg m2/cap)
m2_det_rur = pd.DataFrame(stock_share_rur_det.values * m2_rur.values, columns=people_rur.columns, index=people_rur.index)
m2_sem_rur = pd.DataFrame(stock_share_rur_sem.values * m2_rur.values, columns=people_rur.columns, index=people_rur.index)
m2_app_rur = pd.DataFrame(stock_share_rur_app.values * m2_rur.values, columns=people_rur.columns, index=people_rur.index)
m2_hig_rur = pd.DataFrame(stock_share_rur_hig.values * m2_rur.values, columns=people_rur.columns, index=people_rur.index)

m2_det_urb = pd.DataFrame(stock_share_urb_det.values * m2_urb.values, columns=people_urb.columns, index=people_urb.index)
m2_sem_urb = pd.DataFrame(stock_share_urb_sem.values * m2_urb.values, columns=people_urb.columns, index=people_urb.index)
m2_app_urb = pd.DataFrame(stock_share_urb_app.values * m2_urb.values, columns=people_urb.columns, index=people_urb.index)
m2_hig_urb = pd.DataFrame(stock_share_urb_hig.values * m2_urb.values, columns=people_urb.columns, index=people_urb.index)
# %%
#%% Old code
commercial_m2_cap_office = pd.DataFrame(index=range(1971, END_YEAR + 1), columns=range(1,27))    # Offices
commercial_m2_cap_retail = pd.DataFrame(index=range(1971, END_YEAR + 1), columns=range(1,27))    # Retail & Warehouses
commercial_m2_cap_hotels = pd.DataFrame(index=range(1971, END_YEAR + 1), columns=range(1,27))    # Hotels & Restaurants
commercial_m2_cap_govern = pd.DataFrame(index=range(1971, END_YEAR + 1), columns=range(1,27))    # Hospitals, Education, Government & Transportation

minimum_com_office = 25
minimum_com_retail = 25
minimum_com_hotels = 25
minimum_com_govern = 25
for year in YEARS:
    for region in REGIONS:
        
        # get the square meter per capita floorspace for 4 commercial applications
        office = gompertz['Office']['a'] * math.exp(-gompertz['Office']['b'] * math.exp((-gompertz['Office']['c']/1000) * sva_pc[str(region)][year]))
        retail = gompertz['Retail+']['a'] * math.exp(-gompertz['Retail+']['b'] * math.exp((-gompertz['Retail+']['c']/1000) * sva_pc[str(region)][year]))
        hotels = gompertz['Hotels+']['a'] * math.exp(-gompertz['Hotels+']['b'] * math.exp((-gompertz['Hotels+']['c']/1000) * sva_pc[str(region)][year]))
        govern = gompertz['Govt+']['a'] * math.exp(-gompertz['Govt+']['b'] * math.exp((-gompertz['Govt+']['c']/1000) * sva_pc[str(region)][year]))

        #calculate minimum values for later use in historic tail(Region 20: China @ 134 $/cap SVA)
        minimum_com_office = office if office < minimum_com_office else minimum_com_office      
        minimum_com_retail = retail if retail < minimum_com_retail else minimum_com_retail
        minimum_com_hotels = hotels if hotels < minimum_com_hotels else minimum_com_hotels
        minimum_com_govern = govern if govern < minimum_com_govern else minimum_com_govern
        
        # Then use the ratio's to subdivide the total commercial floorspace into 4 categories      
        commercial_sum = office + retail + hotels + govern
        
        commercial_m2_cap_office[region][year] = commercial_m2_cap[region][year] * (office/commercial_sum)
        commercial_m2_cap_retail[region][year] = commercial_m2_cap[region][year] * (retail/commercial_sum)
        commercial_m2_cap_hotels[region][year] = commercial_m2_cap[region][year] * (hotels/commercial_sum)
        commercial_m2_cap_govern[region][year] = commercial_m2_cap[region][year] * (govern/commercial_sum)

"""
# debugging population figure (used to determine a solution of interpolation issue based on 5yr data input format)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Population (million)', rotation='vertical', y=0.8, fontsize=10)
ax1.margins(x=0)
ax1.plot(np.array(pop2.index), list(pop2.sum(axis=1)), label=["pop"], color='black')
ax1.plot(np.array(pop2.index), list(pop2_new.sum(axis=1)), label=["pop_new"], color='blue')
ax1.plot(np.array(rurpop2.index), list((rurpop2 * pop2).sum(axis=1)), '--', label=["rur_pop"], color='black')
ax1.plot(np.array(rurpop2.index), list((rurpop2_new * pop2_new).sum(axis=1)), '--', label=["rur_pop_new"], color='blue')
ax1.plot(np.array(urbpop.index), list((urbpop * pop2).sum(axis=1)), '*', label=["urb_pop"], color='black')
ax1.plot(np.array(urbpop.index), list((urbpop_new * pop2_new).sum(axis=1)), '*', label=["urb_pop_new"], color='blue')
plt.show()

sva_pc_diff = [0, 0, 0]
sva_pc_diff_new = [0, 0, 0]
region_diff = '2'
for year in range(1972,2060):
    #rural_diff.append( (rurpop2[region_diff][year] * pop2[region_diff][year]) - (rurpop2[region_diff][year-1] * pop2[region_diff][year-1]) )
    sva_pc_diff.append( sva_pc[region_diff][year] - sva_pc[region_diff][year-1] )
    sva_pc_diff_new.append( sva_pc_new[region_diff][year] - sva_pc_new[region_diff][year-1] )

fig, ax1 = plt.subplots()
ax1.set_ylabel('Population (million)', rotation='vertical', y=0.8, fontsize=10)
ax1.margins(x=0)
ax1.plot(np.array(sva_pc_new.index), sva_pc_diff, label=["diff"], color='red')
ax1.plot(np.array(sva_pc_new.index), sva_pc_diff_new,  '--', label=["diff_new"], color='red')
plt.show()
"""