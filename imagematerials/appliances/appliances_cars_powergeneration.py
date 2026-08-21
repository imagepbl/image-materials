"""
Author: Sebastiaan Deetmen, 2018

Code is supplement of:
Deetman, S., Pauliuk, S., Van Vuuren, D. P., Van Der Voet, E., & Tukker, A. (2018). Scenarios for 
demand growth of metals in electricity generation technologies, cars, and electronic appliances. 
Environmental science & technology, 52(8), 4950.
"""
# define imports, counters & general settings
import sys, math
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

cohorts = 50
applian = 11
startyear = 1971
endyear = 2050
years = endyear - startyear + 1
metals = 5
vehicles = 25
categories = 5
entech = 28 # the number of energy technologies
products = categories + applian + 1 # the number of products considered (car categories + nr of appliances + 1 other appliances)
moving_average_window = 5
# load input files (6 TIMER files + weibull parameters & metal composition)
filenames = ['kilometrage', # Assumption: average nr of kilometers driven annually per car, by region & time
    'weibull', # Assumption: weibull shape & scale parameters of each appliance
    'composition', # Assumption: metal composition of appliances (gram/appliance), cars (gram/car) & energy technologies (gram/MW)
    'diffusion', # Scenario_input: diffusion of appliances per household (#/household)
    'households', # Scenario_input: number of households by region (specified for rural & urban by income quintile, we only use the TOTAL nr of households specified in col 1)
    'loadfactor', # Scenario_input: average load factor (passengers/car) by region --> carpooling!
    'passengerkms', # Scenario_input: demand for passenger kilometers (trillion pkm)
    'applianceEU', # Scenario_input: energy use for appliances
    'vehicleshare', # Scenario_input: fraction of the car type in the total vehicle fleet (by region)
    'Gcap_new'] # Scenario_input: newly installed capacity of power generating technologies (MW, by region, by technology, by year)

# read in files, split them based on line-end
kilometrage = open(inputpath + filenames[0] + '.csv', 'r').read().split('\n') # Assumptions
weibull = open(inputpath + filenames[1] + '.csv', 'r').read().split('\n') # Assumptions
composition = open(inputpath + filenames[2] + '.csv', 'r').read().split('\n') # Assumptions
diffusion = open(scen_path + filenames[3] + '.csv', 'r').read().split('\n') # given TIMER scenario output
households = open(scen_path + filenames[4] + '.csv', 'r').read().split('\n') # given TIMER scenario output
loadfactor = open(scen_path + filenames[5] + '.csv', 'r').read().split('\n') # given TIMER scenario output
passengerkms = open(scen_path + filenames[6] + '.csv', 'r').read().split('\n') # given TIMER scenario output
ApplianceEU = open(scen_path + filenames[7] + '.csv', 'r').read().split('\n') # given TIMER scenario output
vehicleshare = open(scen_path + filenames[8] + '.csv', 'r').read().split('\n') # given TIMER scenario output
Gcap_new = open(scen_path + filenames[9] + '.csv', 'r').read().split('\n') # given TIMER scenario output

# define the names of columns (if possible, based on the file headers)
metalnames = composition[0].split(',')
regionnames = kilometrage[0].split(',')
appliancenames = ['Fan', 'Air-cooler', 'Air-conditioning', 'Refridgerator', 'Microwave', 
                  'Washing Machine', 'Tumbler dryer', 'Dish washer', 'Television', 'VCR/DVD', 
                  'PCs & Laptops', 'Other small appliances']
gentechnames = ['Solar PV', 'CSP', 'Wind onshore', 'Wind offshore', 'Hydro', 'Other Renewables', 
                'Nuclear', '<EMPTY>', 'Conv. Coal', 'Conv. Oil', 'Conv. Natural Gas', 'Waste', 
                'IGCC', 'OGCC', 'NG CC', 'Biomass CC', 'Coal + CCS', 'Oil/Coal + CCS', 
                'Natural Gas + CCS', 'Biomass + CCS', 'CHP Coal', 'CHP Oil', 'CHP Natural Gas', 
                'CHP Biomass', 'CHP Coal + CCS', 'CHP Oil + CCS', 'CHP Natural Gas + CCS', 
                'CHP Biomass + CCS']

# delete first line (column names) & last line (empty line) of all datasets
for item in diffusion, households, loadfactor, passengerkms, kilometrage, vehicleshare, weibull, composition, ApplianceEU, Gcap_new:
    del item[0]
    del item[-1]

# split the lines using comma separator (because they are .csv files)
diffusion1 = [i.split(',') for i in diffusion]
households1 = [i.split(',') for i in households]
loadfactor1 = [i.split(',') for i in loadfactor]
passengerkms1 = [i.split(',') for i in passengerkms]
kilometrage1 = [i.split(',') for i in kilometrage]
vehicleshare1 = [i.split(',') for i in vehicleshare]
weibull1 = [i.split(',') for i in weibull]
composition1 = [i.split(',') for i in composition]
ApplianceEU1 = [i.split(',') for i in ApplianceEU]
GenCapacity1 = [i.split(',') for i in Gcap_new]
GenCapacity2 = [[0 for i in range(len(GenCapacity1[0]))] for i in range(len(GenCapacity1))]

for line in range(0, len(GenCapacity1)):
    for item in range(0, len(GenCapacity1[0])):
        if item < 2:
            GenCapacity2[line][item] = int(GenCapacity1[line][item])
        else:
            GenCapacity2[line][item] = float(GenCapacity1[line][item])

# function to organize list into dimensions, because used multiple times (from name[year,region,product] to name[region][product][year])
def make_useful_list(arg1,y,r,p,s = 1971):
    # return a list organized by dimension, based on input [year, region, list]
    result = [[[0 for k in range(y)] for j in range(p)] for i in range(r)]
    for line in arg1:
        for year in range(y):
            for region in range(r):
                for prod in range(p):
                    if line[0] == year+s and line[1] == region+1:
                        result[region][prod][year] = line[prod+2]
    return result


####################################################################################################
#%% Appliances
# calculate the number of appliances in use based on the appliance diffusion and the number of households
appliances = []
households_total = [0 for i in range(years)]
for hh in households1:
    for diff in diffusion1:
        if hh[0] == diff[0] and hh[1] == diff[1] and diff[2] == '1':
            count = float(hh[2]) # the first column of households contains the total
            fans = float(diff[3]) * count
            airCooler = float(diff[4]) * count
            airConditioner = float(diff[5]) * count
            refridgerator = float(diff[6]) * count
            microwave = float(diff[7]) * count
            washingMachine = float(diff[8]) * count
            tumblerDryer = float(diff[9]) * count
            dishWasher = float(diff[10]) * count
            television = float(diff[11]) * count
            vcr_dvd = float(diff[12]) * count
            pc_computers = float(diff[13]) * count
            parselist = [int(hh[0]), int(hh[1]), fans, airCooler, airConditioner, refridgerator, microwave, washingMachine, tumblerDryer, dishWasher, television, vcr_dvd, pc_computers]
            appliances.append(parselist)
            households_total[int(hh[0])-startyear] += count

# arrange the appliance stock into a usefull list
appliances_all = make_useful_list(appliances,y=years,r=regions,p=applian)
households_useful = make_useful_list(households1,y=years,r=regions,p=applian)

# one additional appliance type ('other') is derived from the energy consumption (no data on diffusion available)
# first select the relevant appliances, being PCs (as a reference) and the 'other' category (to be calculated here)
other_EU = []
for line in ApplianceEU1:
    if line[2] == '1': # select income class 1, representing the total energy consumption (instead of by urban/rural or by quintile)
        other_EU.append([int(line[0]),int(line[1]),float(line[13]),float(line[14])]) # line 13 = PCs, line 14 = other

# calculate energy use per PC in 2015 (as a reference)
other_EU1 = make_useful_list(other_EU,y=years,r=regions,p=2)
other_EU_global = np.sum(other_EU1, axis=0)
EU_perPC = other_EU_global[0] / np.sum(appliances_all, axis=0)[10] # appliances_all[10] = other_EU[0] = PCs

# calculate number of other appliances using the Energy Use per PC (EU_perPC), this is used later on
appliances_other = [[0 for k in range(years)] for i in range(regions)]
for region in range(regions):
    appliances_other[region] = other_EU1[region][1] / EU_perPC[2010-startyear] # assuming similar energy use as PCs in 2010 the number of 'other' appliances is calculated from its total energy consumption


####################################################################################################
#%% vehicles

# calculate the nr of vehicle kms from the passenger kms
vehicle_kms = []
for pas in passengerkms1:
    for load in loadfactor1:
        if pas[0] == load[0] and pas[1] == load[1]:
            vehiclekm = float(pas[6]) / max(float(load[5]),1) # in passengerkms1 the 6th column represents cars, in loadfactor1 that is the 4th, but loadfator seems to go below 1, hence the max() statement
            vehicle = (vehiclekm * 1000000000000) # * trillion to yield actual cars
            vehicle_kms.append([int(load[0]), int(load[1]), vehicle])

# calculate the nr of vehicles (stock/registered) required to fullfill the demand for vehicle kms
vehicles_all = []
for region in range(regions):
    for veh in vehicle_kms:
        for kms in kilometrage1:
            if veh[0] == int(kms[0]) and veh[1] == region+1:
                vehicles_parse = veh[2] / float(kms[1+region])
                vehicles_all.append([int(veh[0]),region+1, vehicles_parse])
            if veh[0] > 2008 and veh[1] == region+1:
                vehicles_parse = veh[2] / float(kilometrage1[-1][1+region]) #last year in set
                vehicles_all.append([int(veh[0]),region+1, vehicles_parse])

# multiply by the vehicleshare for cars to get the nr of vehicles by type
vehicles_by_type = []
for veh in vehicles_all:
    for fraction in vehicleshare1:
        if veh[0] == int(fraction[0]) and veh[1] == int(fraction[1]):
            parsevar2 = []
            for type in range(vehicles):
                parsevar1 = veh[2] * float(fraction[2+type])
                parsevar2.append(parsevar1)
            vehicles_by_type.append([veh[0], veh[1], parsevar2])

# categorisation of original IMAGE car types (25) to broader categories (5)
carnames =['ICE', 'HEV', 'FCV', 'PHEV', 'BEV']
vehicles_by_category =[]
for line in vehicles_by_type:
    vehicles_ICE = sum(line[2][0:7]) # sum over all Internal Combustion Engine vehicles
    vehicles_HEV = sum(line[2][7:12]) # sum over all Hybrid Electric Vehicles
    vehicles_FCV = sum(line[2][12:15]) # sum over all Fuel Cell Vehicles
    vehicles_PHEV= sum(line[2][15:21]) # sum over all Plugin Hybrid Electric Vehicles
    vehicles_BEV = sum(line[2][21:25]) # sum over all Battery Electric Vehicles
    vehicles_by_category.append([line[0], line[1], vehicles_ICE, vehicles_HEV, vehicles_FCV, vehicles_PHEV, vehicles_BEV])



####################################################################################################
#%% Arrange cars & appliances

# make useful list for cars first
vehicles_by_cat = make_useful_list(vehicles_by_category,y=years,r=regions,p=categories)
vehicle_kms_all = make_useful_list(vehicle_kms, y=years,r=regions, p=1) #[region][product][year], product is only 1 dimension in this case
Gcap_new_by_cat = make_useful_list(GenCapacity2,y=years,r=regions+1,p=entech)

# Get global sums for output indicators on cars
vehicle_kms_tot = np.sum(vehicle_kms_all, axis=0)
vehicles_tot_by_cat = np.sum(vehicles_by_cat, axis=0)
vehicles_tot = np.sum(np.sum(vehicles_by_cat, axis=0), axis=0)

# Get global sums for output indicators on appliances
appliances_tot = np.sum(np.sum(appliances_all, axis=0), axis=0) + np.sum(appliances_other, axis=0)
appliances_tot_by_cat = np.sum(appliances_all, axis=0) # sum of all regions [appl][yrs]
appliances_tot_other = np.sum(appliances_other, axis=0) # sum of all regions [yrs]
appliances_tot_by_cat_all = np.transpose(np.column_stack((appliances_tot_by_cat.T, appliances_tot_other)))

# get averages for output indicators (2010-2015 & 2045-2050):
avg_nr_households_2015 = 0
avg_appliances_tot_2015 = 0
avg_appliances_tot_by_cat_2015 = [0 for i in range(applian + 1)]
avg_vehiclekms_tot_2015 = 0
avg_vehicles_tot_2015 = 0
avg_vehicles_tot_by_cat_2015 = [0 for i in range(categories)]
avg_nr_households_2050 = 0
avg_appliances_tot_2050 = 0
avg_appliances_tot_by_cat_2050 = [0 for i in range(applian + 1)]
avg_vehiclekms_tot_2050 = 0
avg_vehicles_tot_2050 = 0
avg_vehicles_tot_by_cat_2050 = [0 for i in range(categories)]

for year in range(2010-startyear,2016-startyear):
    avg_nr_households_2015 = avg_nr_households_2015 + households_total[year]/6
    avg_appliances_tot_2015 = avg_appliances_tot_2015 + appliances_tot[year]/6
    avg_vehiclekms_tot_2015 = avg_vehiclekms_tot_2015 + vehicle_kms_tot[0][year]/6
    avg_vehicles_tot_2015 = avg_vehicles_tot_2015 + vehicles_tot[year]/6
    for cat in range(categories):
        avg_vehicles_tot_by_cat_2015[cat] = avg_vehicles_tot_by_cat_2015[cat] + vehicles_tot_by_cat[cat][year]/6
    for appl in range(applian + 1):
        avg_appliances_tot_by_cat_2015[appl] = avg_appliances_tot_by_cat_2015[appl] + appliances_tot_by_cat_all[appl][year]/6

for year in range(2045-startyear,2051-startyear):
    avg_nr_households_2050 = avg_nr_households_2050 + households_total[year]/6
    avg_appliances_tot_2050 = avg_appliances_tot_2050 + appliances_tot[year]/6
    avg_vehiclekms_tot_2050 = avg_vehiclekms_tot_2050 + vehicle_kms_tot[0][year]/6
    avg_vehicles_tot_2050 = avg_vehicles_tot_2050 + vehicles_tot[year]/6
    for cat in range(categories):
        avg_vehicles_tot_by_cat_2050[cat] = avg_vehicles_tot_by_cat_2050[cat] + vehicles_tot_by_cat[cat][year]/6
    for appl in range(applian + 1):
        avg_appliances_tot_by_cat_2050[appl] = avg_appliances_tot_by_cat_2050[appl] + appliances_tot_by_cat_all[appl][year]/6

# add appliances & vehicles into a common products list (not the generation technologies, as their new capacity is already known!)
products_all = [[[0 for k in range(years)] for j in range(products)] for i in range(regions)]
for region in range(regions):
    for prod in range(products):
        if prod < applian:
            products_all[region][prod] = appliances_all[region][prod]
        elif prod == applian:
            products_all[region][prod] = appliances_other[region]
        else:
            products_all[region][prod] = vehicles_by_cat[region][(prod-(applian+1))]


####################################################################################################
#%% define weibull

# define cohorts and weibull distribution based on assumed weibull parameters
weibull_SF = [[0 for col in range(cohorts)] for row in range(products)]
weibull_PDF = [[0 for col in range(cohorts)] for row in range(products)]
weibull_fractions = [[0 for col in range(cohorts)] for row in range(products)]

for prod in range(products):
    for cohort in range(0,cohorts):
        Wshape = float(weibull1[prod][0])
        Wscale = float(weibull1[prod][1])
        weibull_SF[prod][cohort] = weibull_min.sf(cohort,Wshape,0,Wscale)
        weibull_PDF[prod][cohort] = weibull_min.pdf(cohort,Wshape,0,Wscale)


#%% apply Weibull

#define fractions per cohort for the first year stock, based on weibull distributions
for prod in range(products):
    for cohort in range(cohorts):
        weibull_fractions[prod][cohort] = weibull_SF[prod][cohort] / sum(weibull_SF[prod])

# assign initial 1971 stock to cohorts using weibull distribution
products_init = [[[0 for k in range(cohorts)] for j in range(products)] for i in range(regions)]
for prod in range(products):
    for cohort in range(cohorts):
        for region in range(regions):
            products_init[region][prod][cohort] = products_all[region][prod][0] * weibull_fractions[prod][cohorts-1-cohort] # reverse the order of age cohorts, because the DSM-class expects the newest cohort to be the last in the list

# start calculation of cohorts by year
# first, calculate what the stock should have been in the years before the initial cohort distribution in year t0
products_backward_inflow = [[0 for k in range(products)] for j in range(regions)]
products_backward_stock = [[0 for k in range(products)] for j in range(regions)]
for prod in range(products):
    for region in range(regions):
        DSMbackward = DSM(t = np.arange(1,cohorts+1,1), lt = {'Type': 'Weibull', 'Shape': np.array([float(weibull1[prod][0])]), 'Scale': np.array([float(weibull1[prod][1])])})
        products_backward_inflow[region][prod], Exitflag = DSMbackward.compute_i_from_s(products_init[region][prod])
        CheckStr, ExitFlag = DSMbackward.dimension_check()
        S_C1, ExitFlag = DSMbackward.compute_s_c_inflow_driven()
        products_backward_stock[region][prod], Exitflag = DSMbackward.compute_stock_total()

# add this 'tail' to the actual stock & compute its rolling mean (MOVING AVERAGE)
products_all_extended = [[[0 for k in range(years+cohorts-1)] for j in range(products)] for i in range(regions)]
products_all_extended_smooth = [[[0 for k in range(years+cohorts-1)] for j in range(products)] for i in range(regions)]
for region in range(regions):
    for prod in range(products):
        for year in range(years+cohorts):
            if year < cohorts:
                products_all_extended[region][prod][year] = products_backward_stock[region][prod][year]
            else:
                products_all_extended[region][prod][year-1] = products_all[region][prod][year-cohorts]
        products_all_extended_smooth[region][prod] = pd.rolling_mean(np.array(products_all_extended[region][prod]), window=moving_average_window).tolist()

# remove NaN values from the rolling mean calculation, replace by 0
for region in range(regions):
    for prod in range(products):
        for year in range(years):
            if math.isnan(float(products_all_extended_smooth[region][prod][year])):
                products_all_extended_smooth[region][prod][year] = 0.0

# use the stock driven model to calculate inflows & outflows over the full time-series
Stock_by_cohort = [[[0 for k in range(years+cohorts)] for j in range(products)] for i in range(regions)]
Outflow_by_cohort = [[[0 for k in range(years+cohorts)] for j in range(products)] for i in range(regions)]
Inflow = [[[0 for k in range(years+cohorts)] for j in range(products)] for i in range(regions)]
for prod in range(products):
    for region in range(regions):
        TestDSM = DSM(t = np.arange(1,years+cohorts,1), s = np.array(products_all_extended[region][prod]), lt = {'Type': 'Weibull', 'Shape': np.array([float(weibull1[prod][0])]), 'Scale': np.array([float(weibull1[prod][1])])})
        Stock_by_cohort[region][prod], Outflow_by_cohort[region][prod], Inflow[region][prod], ExitFlag = TestDSM.compute_stock_driven_model()

# Add the new energy generation capacity to the Inflow var, this is also where the years before 1971 are cut off
Inflow_all = [[[0 for k in range(years)] for j in range(products + entech)] for i in range(regions)]
for year in range(years):
    for region in range(regions):
        for prod in range(products + entech):
            if prod < products:
                Inflow_all[region][prod][year] = Inflow[region][prod][year+cohorts-1]
            else:
                Inflow_all[region][prod][year] = Gcap_new_by_cat[region][prod - products][year]

# determine sums of the Inflow of appliances & cars (global) for use in results table
inflow_total_by_product = [[0 for k in range(years)] for i in range(products + entech)]
inflow_total_by_product = np.sum(Inflow_all, axis=0)

# Determine total inflow of metals
Metal_inflow = [[[[0 for k in range(years)] for j in range(products + entech)] for i in range(regions)] for m in range(metals)]
for year in range(years):
    for region in range(regions):
        for prod in range(products + entech):
            for metal in range(metals):
                Metal_inflow[metal][region][prod][year] = Inflow_all[region][prod][year] * float(composition1[prod][metal])

# determine sums by region or product, or global sums of metal use
Metal_inflow_total_by_region = [[[0 for k in range(years)] for i in range(regions)] for m in range(metals)]
Metal_inflow_total_by_product = [[[0 for k in range(years)] for i in range(products + entech)] for m in range(metals)]
Metal_inflow_total = [[0 for k in range(years)] for m in range(metals)]
for metal in range(metals):
    for prod in range(products + entech):
        Metal_inflow_total_by_product[metal] = np.sum(Metal_inflow[metal], axis=0)
    for region in range(regions):
        Metal_inflow_total_by_region[metal][region] = np.sum(Metal_inflow[metal][region], axis=0)
    Metal_inflow_total[metal] = np.sum(Metal_inflow_total_by_region[metal], axis=0)

# calculate the annual metal demand growth index per category for 6-year averages 2010-2015 and averages of 2045-2050
index_inflow_metal = [[[0 for j in range(3)] for i in range(3)] for m in range(metals)]
for metal in range(metals):
    for time in range(2):
        for prod in range(0,12):
            index_inflow_metal[metal][0][time] = index_inflow_metal[metal][0][time] + sum(Metal_inflow_total_by_product[metal][prod][(2010 + time*35) - startyear : (2016 + time*35) - startyear])/6
        for prod in range(12,17):
            index_inflow_metal[metal][1][time] = index_inflow_metal[metal][1][time] + sum(Metal_inflow_total_by_product[metal][prod][(2010 + time*35) - startyear : (2016 + time*35) - startyear])/6
        for prod in range(17,44):
            index_inflow_metal[metal][2][time] = index_inflow_metal[metal][2][time] + sum(Metal_inflow_total_by_product[metal][prod][(2010 + time*35) - startyear : (2016 + time*35) - startyear])/6
    for cat in range(3):
        if index_inflow_metal[metal][cat][0] != 0:
            index_inflow_metal[metal][cat][2] = index_inflow_metal[metal][cat][1]/index_inflow_metal[metal][cat][0]

# calculate the annual product demand for averages 2010-2015 and averages of 2045-2050
avg_inflow_prod = [[0 for i in range(2)] for m in range(products + entech)]
for prod in range(products + entech):
    for time in range(2):
        avg_inflow_prod[prod][time] = avg_inflow_prod[prod][time] + sum(inflow_total_by_product[prod][(2010 + time*35) - startyear : (2016 + time*35) - startyear])/6

# calculate the average shares of car & energytech purchases in 2010-2015 vs. 2045-2050
purchase_perc_appl = [[0 for i in range(2)] for m in range(applian + 1)]
purchase_perc_cars = [[0 for i in range(2)] for m in range(categories)]
purchase_perc_etech = [[0 for i in range(2)] for m in range(entech)]
purchase_sum_appl = [0,0]
purchase_sum_cars = [0,0]
purchase_sum_etech = [0,0]
for time in range(2):
    for prod in range(0,12):
        purchase_sum_appl[time] = purchase_sum_appl[time] + avg_inflow_prod[prod][time]
    for prod in range(12,17):
        purchase_sum_cars[time] = purchase_sum_cars[time] + avg_inflow_prod[prod][time]
    for prod in range(17,44):
        purchase_sum_etech[time] = purchase_sum_etech[time] + avg_inflow_prod[prod][time]



for time in range(2):
    for prod in range(0,12):
        purchase_perc_appl[prod][time] = avg_inflow_prod[prod][time] / purchase_sum_appl[time]
    for prod in range(12,17):
        purchase_perc_cars[prod - 12][time] = avg_inflow_prod[prod][time] / purchase_sum_cars[time]
    for prod in range(17,44):
        purchase_perc_etech[prod - 17][time] = avg_inflow_prod[prod][time] / purchase_sum_etech[time]