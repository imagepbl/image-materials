os.chdir('/Users/Arp00003/Coding/IMAGE-Mat/IMAGE-Mat')   # SET YOUR PATH HERE

st = time.time()

# settings & constants
START_YEAR = 1971            # start year of historic IMAGE data
END_YEAR = 2060
OUT_YEAR = 2060              # year of output generation
REGIONS = 26
idx = pd.IndexSlice          # needed for slicing multi-index
FIRST_YEAR_BOATS = 1900
LOAD_FACTOR = 1.6            # reference loadfactor of cars in TIMER (the trp_trvl_Load.out file is relative to this BASE loadfcator (persons/car))
FIRST_YEAR = first_year_vehicle.values.min()  # start year of the full model period (including stock-development from scratch, which needs to be the oldest year of any vehicle, all stock calculations are initiated in this year, so this has an effect on runtime)

# scenario settings
SCEN    = 'SSP2'
VARIANT = '2D_RE'               # CP or 2D (Add '_RE' for Resource Efficiency)
PROJECT = 'mock_project'
FOLDER  = SCEN + '_' + VARIANT
IMAGE_FOLDER = PROJECT + '/' + SCEN
OUTPUT_FOLDER ='output/' + PROJECT + '/' + FOLDER

#%% Reading all csv files for vehicles and ships that are external to IMAGE

# 1) scenario independent data
load = pd.read_csv('input/vehicles/standard_data/load_pass_and_tonnes.csv')             #TODO: add description again here!
loadfactor = pd.read_csv('input/vehicles/standard_data/loadfactor_percentages.csv')     # Percentage of the maximum load that is on average 
market_share = pd.read_csv('input/vehicles/standard_data/fraction_tkm_pkm.csv')         # Percentage of tonne-/passengerkilometres
first_year_vehicle = pd.read_csv('input/vehicles/standard_data/first_year_vehicle.csv') # first year of operation per vehicle-type
battery_shares_full = pd.read_csv('input/vehicles/standard_data/battery_share_inflow.csv', index_col=0)  # The share of the battery market (8 battery types used in vehicles), this data is based on a Multi-Nomial-Logit market model & costs in https://doi.org/10.1016/j.resconrec.2020.105200 - since this is scenario dependent it's placed under the 'IMAGE' scenario folder

# Files related to the international shipping
nr_of_boats  = pd.read_csv('input/vehicles/standard_data/ships/number_of_boats.csv', index_col='t').sort_index(axis=0)           # number of boats in the global merchant fleet (2005-2018)   changing Data by EQUASIS
cap_of_boats        = pd.read_csv('input/vehicles/standard_data/ships/capacity_ton_boats.csv', index_col='t').sort_index(axis=0)        # boat capacity in tons                                      changing Data is a combination of EQUASIS Gross Tonnage and UNCTAD Dead-Weigh-Tonnage per Gross Tonnage
loadfactor_boats    = pd.read_csv('input/vehicles/standard_data/ships/loadfactor_boats.csv', index_col='t').sort_index(axis=0)          # loadfactor of boats (fraction)                             fixed    Data is based on Ecoinvent report 14 on Transport (Table 8-19)
mileage_boats       = pd.read_csv('input/vehicles/standard_data/ships/mileage_kmyr_boats.csv', index_col='t').sort_index(axis=0)        # mileage of boats in km/yr (per ship)                       fixed    Data is based on Ecoinvent report 14 on Transport (Table 8-19)
weight_boats        = pd.read_csv('input/vehicles/standard_data/ships/weight_percofcap_boats.csv', index_col='t').sort_index(axis=0)    # weight of boats as a percentage of the capacity (%)        fixed    Data is based on Ecoinvent report 14 on Transport (section 8.4.1)

# 2) scenario dependent data
lifetimes_vehicles = pd.read_csv('input/vehicles/' + FOLDER + '/lifetimes_years.csv',  index_col=[0,1])   # Average End-of-Life of vehicles in years, this file also contains the setting for the choice of distribution and other lifetime related settings (standard devition, or alternative parameterisation)
kilometrage = pd.read_csv('input/vehicles/' + FOLDER + '/kilometrage.csv',      index_col='t')      # kilometrage of passenger cars in kms/yr
kilometrage_midi_bus = pd.read_csv('input/vehicles/' + FOLDER + '/kilometrage_midi.csv', index_col='t')      # kilometrage of midi-buses in kms/yr
kilometrage_bus = pd.read_csv('input/vehicles/' + FOLDER + '/kilometrage_bus.csv',  index_col='t')     # kilometrage of regular buses in kms/yr
mileages = pd.read_csv('input/vehicles/' + FOLDER + '/mileages_km_per_year.csv', index_col='t') # Km/year of all the vehicles (buses & cars have region-specific files)

# weight and materials related data
vehicle_weight_kg_simple = pd.read_csv('input/vehicles/' + FOLDER + '/vehicle_weight_kg_simple.csv',   index_col=0)       # Weight of a single vehicle of each type in kg
vehicle_weight_kg_typical = pd.read_csv('input/vehicles/' + FOLDER + '/vehicle_weight_kg_typical.csv',  index_col=[0,1])   # Weight of a single vehicle of each type in kg
material_fractions = pd.read_csv('input/vehicles/' + FOLDER + '/material_fractions_simple.csv',  index_col=[0,1])   # Material fractions in percentages
material_fractions_type = pd.read_csv('input/vehicles/' + FOLDER + '/material_fractions_typical.csv', index_col=[0,1], header=[0,1])   # Material fractions in percentages, by vehicle sub-type
battery_weights = pd.read_csv('input/vehicles/' + FOLDER + '/battery_weights_kg.csv', index_col=[0,1])   # Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is determined
battery_materials = pd.read_csv('input/vehicles/' + FOLDER + '/battery_materials.csv', index_col=[0,1])   # The material fraction of storage technologies (used to get the vehicle battery composition)

#%% Reading all out files for vehicles and ships that are internal to IMAGE

# IMAGE scenario files (total demand in Tkms & Pkms + vehicle shares)
tonkms_Mtkms        = read_mym_df('image/' + IMAGE_FOLDER + '/trp_frgt_Tkm.out')              # The tonne kilometres of freight vehicles of the IMAGE/TIMER SSP2 (in Mega Tkm)
passengerkms_Tpkms  = read_mym_df('image/' + IMAGE_FOLDER + '/trp_trvl_pkm.out')              # The passenger kilometres from the IMAGE/TIMER SSP2 (in Tera Pkm)
buses_vshares       = read_mym_df('image/' + IMAGE_FOLDER + '/trp_trvl_Vshare_bus.out')       # The vehicle shares of buses of the SSP2                            MIND! FOR the BL this is still the OLD SSP2 file REPLACE LATER
car_vshares         = read_mym_df('image/' + IMAGE_FOLDER + '/trp_trvl_Vshare_car.out')       # The vehicle shares of passenger cars of the SSP2 
medtruck_vshares    = read_mym_df('image/' + IMAGE_FOLDER + '/trp_frgt_Vshare_MedTruck.out')  # The vehicle shares of trucks (medium) of the SSP2 
hvytruck_vshares    = read_mym_df('image/' + IMAGE_FOLDER + '/trp_frgt_Vshare_HvyTruck.out')  # The vehicle shares of trucks (heavy) of the SSP2 
loadfactor_car_data = read_mym_df('image/' + IMAGE_FOLDER + '/trp_trvl_Load.out')             # The loadfactor of passenger vehicles (occupation in nr of people/vehicle) in reference to the base loadfactor (see constants above)

#%%
"""
preprocessing of the IMAGE files & files with additional assumptions on vehcile materials (renaming, removing 27th region, adding labels etc.)
"""
range_interpol       = list(range(START_YEAR, END_YEAR + 1))
kilometrage          = kilometrage.reindex(range_interpol).interpolate(limit_direction='both')
kilometrage_bus      = kilometrage_bus.reindex(range_interpol).interpolate(limit_direction='both')
kilometrage_midi_bus = kilometrage_midi_bus.reindex(range_interpol).interpolate(limit_direction='both')
mileages             = mileages.reindex(range_interpol).interpolate(limit_direction='both')

region_list          = list(kilometrage.columns.values)     # get a list with region names

# select loadfactor for cars
car_loadfactor = loadfactor_car_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1').droplevel(level=0, axis=1)  * LOAD_FACTOR # loadfactor for cars (in persons per vehicle) * LOAD_FACTOR to correct with te TIMER reference
car_loadfactor = car_loadfactor.apply(lambda x: [y if y >= 1 else 1 for y in x])                      # To avoid car load (person/vehicle) values ever going below 1, replace all values below 1 with 1
car_loadfactor = car_loadfactor.loc[list(range(START_YEAR, END_YEAR+1)),:]     # remove years beyond LAST_YEAR
car_loadfactor.columns = region_list

# select data only for requested output years
tonkms_Mtkms        = tonkms_Mtkms[tonkms_Mtkms['time'].isin(list(range(START_YEAR, END_YEAR+1)))]
passengerkms_Tpkms  = passengerkms_Tpkms[passengerkms_Tpkms['time'].isin(list(range(START_YEAR, END_YEAR+1)))] 
buses_vshares       = buses_vshares[buses_vshares['time'].isin(list(range(START_YEAR, END_YEAR+1)))] 
car_vshares         = car_vshares[car_vshares['time'].isin(list(range(START_YEAR, END_YEAR+1)))] 
medtruck_vshares    = medtruck_vshares[medtruck_vshares['time'].isin(list(range(START_YEAR, END_YEAR+1)))] 
hvytruck_vshares    = hvytruck_vshares[hvytruck_vshares['time'].isin(list(range(START_YEAR, END_YEAR+1)))] 
battery_shares_full = battery_shares_full.loc[list(range(START_YEAR, END_YEAR+1))]

#set multi-index based on the first two columns
tonkms_Mtkms.set_index(['time', 'DIM_1'], inplace=True)
passengerkms_Tpkms.set_index(['time', 'DIM_1'], inplace=True)
buses_vshares.set_index(['time', 'DIM_1'], inplace=True)
car_vshares.set_index(['time', 'DIM_1'], inplace=True)
medtruck_vshares.set_index(['time', 'DIM_1'], inplace=True)
hvytruck_vshares.set_index(['time', 'DIM_1'], inplace=True)

bus_label   = ['BusOil',	'BusBio',	'BusGas',	'BusElecTrolley',	'Bus Hybrid1',	'Bus Hybrid2',	'BusBattElectric', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
truck_label = ['Conv. ICE(2000)',	'Conv. ICE(2010)', 'Adv. ICEOil',	'Adv. ICEH2', 'Turbo-petrol IC', 'Diesel ICEOil', 'Diesel ICEBio', 'ICE-HEV-gasoline', 'ICE-HEV-diesel oil', 'ICE-HEV-H2', 'ICE-HEV-CNG Gas', 'ICE-HEV-diesel bio', 'FCV Oil', 'FCV Bio', 'FCV H2', 'PEV-10 OilElec.', 'PEV-30 OilElec.', 'PEV-60 OilElec.', 'PEV-10 BioElec.', 'PEV-30 BioElec.', 'PEV-60 BioElec.', 'BEV Elec.', '', '', '']
car_label   = ['Conv. ICE(2000)',	'Conv. ICE(2010)', 'Adv. ICEOil',	'Adv. ICEH2', 'Turbo-petrol IC', 'Diesel ICEOil', 'Diesel ICEBio', 'ICE-HEV-gasoline', 'ICE-HEV-diesel oil', 'ICE-HEV-H2', 'ICE-HEV-CNG Gas', 'ICE-HEV-diesel bio', 'FCV Oil', 'FCV Bio', 'FCV H2', 'PEV-10 OilElec.', 'PEV-30 OilElec.', 'PEV-60 OilElec.', 'PEV-10 BioElec.', 'PEV-30 BioElec.', 'PEV-60 BioElec.', 'BEV Elec.', 'PHEV_BEV', 'BEV', 'Gas car']
tkms_label  = ['inland shipping', 'freight train', 'medium truck', 'heavy truck', 'air cargo', 'international shipping', 'empty', 'total']
pkms_label  = ['walking', 'biking', 'bus', 'train', 'car', 'hst', 'air', 'total']
columns_vehcile_output = ['Buses','Trains','HST','Cars','Planes','Bikes','Trucks','Cargo Trains','Ships','Inland ships','Cargo Planes']

# insert column descriptions
tonkms_Mtkms.columns       = tkms_label
passengerkms_Tpkms.columns = pkms_label
medtruck_vshares.columns   = truck_label
hvytruck_vshares.columns   = truck_label
buses_vshares.columns      = bus_label

# output transport drivers to output folder for 450 vs Bl comparisson in overarching figures later on
tonkms_Mtkms.to_csv(OUTPUT_FOLDER + '/transport_tkms.csv', index=True)       # in Mega tkms
passengerkms_Tpkms.to_csv(OUTPUT_FOLDER + '/transport_pkms.csv', index=True) # in Tera pkms

# aggregate car types into 5 car types
BEV_collist  = [22, 24]
PHEV_collist = [23, 21, 20, 19, 18, 17, 16]
ICE_collist  = [1,2,3,4,5,6,7,25]             # Gas car is considered ICE
HEV_collist  = [8,9,10,11,12]
FCV_collist  = [13,14,15]
car_types = ['ICE','HEV','PHEV','BEV','FCV']

index = pd.MultiIndex.from_product([list(kilometrage.index), list(range(1,27))], names=['time', 'DIM_1'])
vehicleshare_cars = pd.DataFrame(index=index, columns=car_types)
vehicleshare_cars.loc[idx[:,:],'ICE']  = car_vshares[ICE_collist].sum(axis=1).to_numpy()
vehicleshare_cars.loc[idx[:,:],'HEV']  = car_vshares[HEV_collist].sum(axis=1).to_numpy()
vehicleshare_cars.loc[idx[:,:],'PHEV'] = car_vshares[PHEV_collist].sum(axis=1).to_numpy()
vehicleshare_cars.loc[idx[:,:],'BEV']  = car_vshares[BEV_collist].sum(axis=1).to_numpy()
vehicleshare_cars.loc[idx[:,:],'FCV']  = car_vshares[FCV_collist].sum(axis=1).to_numpy()

# output to IRP
vehicleshare_cars.to_csv(OUTPUT_FOLDER + '\\car_type_share_regional.csv', index=True)

# labels etc.
x_graphs        = [i for i in range(START_YEAR, END_YEAR, 1)]                           # this is used as an x-axis for the years in graphs
labels_pas      = ['bicycle', 'rail_reg','rail_hst','midi_bus','reg_bus','air_pas','ICE','HEV','PHEV','BEV', 'FCV']             # Names used to shorten plots
labels_fre      = ['inland_shipping', 'rail_freight','LCV', 'MFT', 'HFT', 'air_freight', 'sea_shipping_small', 'sea_shipping_med', 'sea_shipping_large', 'sea_shipping_vl'] #names used to shorten plots
labels_materials= ['Steel', 'Aluminium', 'Cu', 'Plastics', 'Glass', 'Ti', 'Wood', 'Rubber', 'Li','Co','Ni', 'Mn','Nd','Pb']
labels_ev_batt  = ['NiMH','LMO','NMC','NCA','LFP','Lithium Sulfur','Lithium Ceramic','Lithium-air']


#%% For dynamic variables, apply interpolation and extend over the whole timeframe

# use the add_history_and_future() function to speciyfy dynamic variables over the entire scenario period
# complete & interpolate the vehicle weight data
vehicle_weight_kg_air_pas     = interpolate(pd.DataFrame(vehicle_weight_kg_simple["air_pas"]),         change='no')
vehicle_weight_kg_air_frgt    = interpolate(pd.DataFrame(vehicle_weight_kg_simple["air_freight"]),     change='no')
vehicle_weight_kg_rail_reg    = interpolate(pd.DataFrame(vehicle_weight_kg_simple["rail_reg"]),        change='no')
vehicle_weight_kg_rail_hst    = interpolate(pd.DataFrame(vehicle_weight_kg_simple["rail_hst"]),        change='no')
vehicle_weight_kg_rail_frgt   = interpolate(pd.DataFrame(vehicle_weight_kg_simple["rail_freight"]),    change='no')
vehicle_weight_kg_inland_ship = interpolate(pd.DataFrame(vehicle_weight_kg_simple["inland_shipping"]), change='no')
vehicle_weight_kg_bicycle     = interpolate(pd.DataFrame(vehicle_weight_kg_simple["bicycle"]),         change='no')

vehicle_weight_kg_car         = interpolate(vehicle_weight_kg_typical["car"].unstack(),      change='no')
vehicle_weight_kg_LCV         = interpolate(vehicle_weight_kg_typical["LCV"].unstack(),      change='no')
vehicle_weight_kg_MFT         = interpolate(vehicle_weight_kg_typical["MFT"].unstack(),      change='no')
vehicle_weight_kg_HFT         = interpolate(vehicle_weight_kg_typical["HFT"].unstack(),      change='no')
vehicle_weight_kg_bus         = interpolate(vehicle_weight_kg_typical["reg_bus"].unstack(),  change='no')
vehicle_weight_kg_midi        = interpolate(vehicle_weight_kg_typical["midi_bus"].unstack(), change='no')

# complete & interpolate the vehicle composition data (simple first)
material_fractions_air_pas     = interpolate(material_fractions['air_pas'].unstack(),            change='no')
material_fractions_air_frgt    = interpolate(material_fractions['air_freight'].unstack(),        change='no')
material_fractions_rail_reg    = interpolate(material_fractions['rail_reg'].unstack(),           change='no')
material_fractions_rail_hst    = interpolate(material_fractions['rail_hst'].unstack(),           change='no')
material_fractions_rail_frgt   = interpolate(material_fractions['rail_freight'].unstack(),       change='no')
material_fractions_ship_small  = interpolate(material_fractions['sea_shipping_small'].unstack(), change='no')
material_fractions_ship_medium = interpolate(material_fractions['sea_shipping_med'].unstack(),   change='no')
material_fractions_ship_large  = interpolate(material_fractions['sea_shipping_large'].unstack(), change='no')
material_fractions_ship_vlarge = interpolate(material_fractions['sea_shipping_vl'].unstack(),    change='no')
material_fractions_inland_ship = interpolate(material_fractions['inland_shipping'].unstack(),    change='no')
material_fractions_bicycle     = interpolate(material_fractions['bicycle'].unstack(),            change='no')

# complete & interpolate the vehicle composition data (by vehicle sub-type second)
material_fractions_car        = interpolate(material_fractions_type['car'].unstack(),     change='no')
material_fractions_bus_reg    = interpolate(material_fractions_type['reg_bus'].unstack(), change='no')
material_fractions_bus_midi   = interpolate(material_fractions_type['midi_bus'].unstack(),change='no')
material_fractions_truck_HFT  = interpolate(material_fractions_type['HFT'].unstack(),     change='no')
material_fractions_truck_MFT  = interpolate(material_fractions_type['MFT'].unstack(),     change='no')
material_fractions_truck_LCV  = interpolate(material_fractions_type['LCV'].unstack(),     change='no')

# interpolate & complete series for battery weights, shares & composition too
battery_weights_full    = interpolate(battery_weights.unstack(),   change='no')
battery_materials_full  = interpolate(battery_materials.unstack(), change='no')
battery_shares_full     = interpolate(battery_shares_full, change='no')

# same for lifetime data
lifetimes_vehicles_mean  = interpolate(lifetimes_vehicles.loc[idx[:,"mean"],:].droplevel(["data"]),  change='no')
lifetimes_vehicles_stdev = interpolate(lifetimes_vehicles.loc[idx[:,"stdev"],:].droplevel(["data"]), change='no')
lifetimes_vehicles_shape = interpolate(lifetimes_vehicles.loc[idx[:,"shape"],:].droplevel(["data"]), change='no')
lifetimes_vehicles_scale = interpolate(lifetimes_vehicles.loc[idx[:,"scale"],:].droplevel(["data"]), change='no')

#%% Caculating the tonnekilometres for all freight and passenger vehicle types (adjustments are made to: freight air, trucks, and buses)

# Trucks are calculated differently because the IMAGE model does not account for LCV trucks, which concerns a large portion of the material requirements of road freight
# the total trucks Tkms remain the same, but a LCV fraction is substracted, and the remainder is re-assigned to medium and heavy trucks according to their original ratio
trucks_total_tkm       = tonkms_Mtkms['medium truck'].unstack() +  tonkms_Mtkms['heavy truck'].unstack()
trucks_LCV_tkm         = trucks_total_tkm * 0.04                                    # 0.04 is the fraction of the tkms driven by light commercial vehicles according to the IEA
MFT_percshare_tkm      = tonkms_Mtkms['medium truck'].unstack() / trucks_total_tkm  # the MFT fraction of the total 
HFT_percshare_tkm      = tonkms_Mtkms['heavy truck'].unstack() / trucks_total_tkm   # the HFT fraction of the total 
trucks_min_LCV         = trucks_total_tkm - trucks_LCV_tkm
trucks_MFT_tkm         = trucks_min_LCV.mul(MFT_percshare_tkm)                      
trucks_HFT_tkm         = trucks_min_LCV.mul(HFT_percshare_tkm)

# demand for freight planes is reduced by 50% because about half of the air freight is transported as cargo on passenger planes 
air_freight_tkms       = tonkms_Mtkms['air cargo'].unstack() * market_share['air_freight'].values[0]

# Buses are adjusted to account for the higher material intensity of mini-buses
bus_regl_pkms          = passengerkms_Tpkms['bus'].unstack() * market_share['reg_bus'].values[0]   # in tera pkms
bus_midi_pkms          = passengerkms_Tpkms['bus'].unstack() * market_share['midi_bus'].values[0]  # in tera pkms

# Select tkms of passenger cars (which will be adjusted to represent 5 types: ICE, HEV, PHEV, BEV & FCV)
car_pkms               = passengerkms_Tpkms['car'].unstack()
car_pkms               = car_pkms.drop([27, 28], axis=1)    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies  # in tera pkms                            
car_pkms.columns       = region_list                                  

#%% Calculate the NUMBER OF VEHICLES (stock, on the road) to fulfull the ton-kilometers transport demand
 
#calculate the number of vehicles on the road (first passenger, then freight)
air_pas_nr      = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['air'].unstack(),    mileages['air_pas'],  load['air_pas'].values[0],    loadfactor['air_pas'].values[0])
rail_reg_nr     = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['train'].unstack(),  mileages['rail_reg'], load['rail_reg'].values[0],   loadfactor['rail_reg'].values[0])
rail_hst_nr     = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['hst'].unstack(),    mileages['rail_hst'], load['rail_hst'].values[0],   loadfactor['rail_hst'].values[0])
bikes_nr        = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['biking'].unstack(), mileages['bicycle'],  load['bicycle'].values[0],    loadfactor['bicycle'].values[0])

# original ton kilometers are in Mega-ton-kms, div by 1000000 to harmonize with pkms which are in Tera pkms
trucks_HFT_nr   = tkms_to_nr_of_vehicles_fixed(trucks_HFT_tkm/1000000,   mileages['HFT'],          load['HFT'].values[0],         loadfactor['HFT'].values[0])
trucks_MFT_nr   = tkms_to_nr_of_vehicles_fixed(trucks_MFT_tkm/1000000,   mileages['MFT'],          load['MFT'].values[0],         loadfactor['MFT'].values[0])
trucks_LCV_nr   = tkms_to_nr_of_vehicles_fixed(trucks_LCV_tkm/1000000,   mileages['LCV'],          load['LCV'].values[0],         loadfactor['LCV'].values[0])
air_freight_nr  = tkms_to_nr_of_vehicles_fixed(air_freight_tkms/1000000,  mileages['air_freight'], load['air_freight'].values[0], loadfactor['air_freight'].values[0])
rail_freight_nr = tkms_to_nr_of_vehicles_fixed(tonkms_Mtkms['freight train'].unstack()/1000000, mileages['rail_freight'], load['rail_freight'].values[0],         loadfactor['rail_freight'].values[0])
inland_ship_nr  = tkms_to_nr_of_vehicles_fixed(tonkms_Mtkms['inland shipping'].unstack()/1000000, mileages['inland_shipping'], load['inland_shipping'].values[0], loadfactor['inland_shipping'].values[0])

# passenger cars and buses are calculated separately (due to regional & changeing mileage & load), first the totals
car_total_vkms  = car_pkms.div(car_loadfactor) * 1000000000000    # now in kms
car_total_nr    = car_total_vkms.div(kilometrage)                 # total number of cars
car_total_nr.columns = list(range(1,27))                          # remove region labels (for use in functions later on)

# for buses do the same, but first remove region 27 & 28 (empty & world total) & kilometrage column names
bus_regl_pkms  = bus_regl_pkms.drop([27, 28], axis=1) 
kilometrage_bus.columns = list(range(1,27))
bus_regl_vkms  = bus_regl_pkms.div(load['reg_bus'].values[0] * loadfactor['reg_bus'].values[0]) * 1000000000000    # now in kms
bus_regl_nr    = bus_regl_vkms.div(kilometrage_bus)                              # total number of regular buses

bus_midi_pkms  = bus_midi_pkms.drop([27, 28], axis=1) 
kilometrage_midi_bus.columns = list(range(1,27))   
bus_midi_vkms  = bus_midi_pkms.div(load['midi_bus'].values[0] * loadfactor['midi_bus'].values[0]) * 1000000000000   # now in kms
bus_midi_nr    = bus_midi_vkms.div(kilometrage_midi_bus)                         # total number of regular buses


#%% for INTERNATIONAL SHIPPING the number of vehicles is calculated differently 

cap_adjustment  = [1, 1, 1, 1]
mile_adjustment = [1, 1, 1, 1]

#pre-calculate the shares of the boats based on the number of boats, before adding history/future
share_of_boats       = nr_of_boats.div(nr_of_boats.sum(axis=1), axis=0)

share_of_boats_yrs    =  interpolate(share_of_boats,   change='yes')   # does change based on trend in original data
cap_of_boats_yrs      =  interpolate(cap_of_boats,     change='no')    # could be 'yes' based on better data
loadfactor_boats_yrs  =  interpolate(loadfactor_boats, change='no')   
mileage_boats_yrs     =  interpolate(mileage_boats,    change='no')  
weight_frac_boats_yrs =  interpolate(weight_boats,     change='no')  

# normalize the share of boats to 1 & adjust the capacity & mileage for smaller ships 
share_of_boats_yrs   = share_of_boats_yrs.div(share_of_boats_yrs.sum(axis=1), axis=0)
cap_of_boats_yrs     = cap_of_boats_yrs.mul(cap_adjustment, axis=1)
mileage_boats_yrs    = mileage_boats_yrs.mul(mile_adjustment, axis=1)

# now derive the number of ships for 4 ship types in four steps: 
# 1) get the share of the ship types in the Tkms shipped (what % of total tkms shipped goes by what ship type?)
share_of_boats_tkm_all   = share_of_boats_yrs * cap_of_boats_yrs * loadfactor_boats_yrs * mileage_boats_yrs
share_of_boats_tkm       = share_of_boats_tkm_all.div(share_of_boats_tkm_all.sum(axis=1), axis=0)

# 2) get the total tkms shipped by ship-type. (The shares are pre-calculated from 1900 onwards, so a selection from 1971-onwards is applied here)
ship_small_tkm  = tonkms_Mtkms['international shipping'].unstack().mul(share_of_boats_tkm['Small'].loc[START_YEAR:], axis=0)
ship_medium_tkm = tonkms_Mtkms['international shipping'].unstack().mul(share_of_boats_tkm['Medium'].loc[START_YEAR:], axis=0)
ship_large_tkm  = tonkms_Mtkms['international shipping'].unstack().mul(share_of_boats_tkm['Large'].loc[START_YEAR:], axis=0)
ship_vlarge_tkm = tonkms_Mtkms['international shipping'].unstack().mul(share_of_boats_tkm['Very Large'].loc[START_YEAR:], axis=0)

# 3) get the vehicle-kms by ship type (multiply by 1000000 to get from Mega-tkm to tkm)
ship_small_vehkm  = ship_small_tkm.mul(1000000).div(cap_of_boats_yrs['Small'].loc[START_YEAR:], axis=0)
ship_medium_vehkm = ship_medium_tkm.mul(1000000).div(cap_of_boats_yrs['Medium'].loc[START_YEAR:], axis=0)  
ship_large_vehkm  = ship_large_tkm.mul(1000000).div(cap_of_boats_yrs['Large'].loc[START_YEAR:], axis=0)  
ship_vlarge_vehkm = ship_vlarge_tkm.mul(1000000).div(cap_of_boats_yrs['Very Large'].loc[START_YEAR:], axis=0) 

# 4) get the number of ships (stock) by dividing with the mileage
ship_small_nr  = ship_small_vehkm.div(mileage_boats_yrs['Small'].loc[START_YEAR:], axis=0)
ship_medium_nr = ship_medium_vehkm.div(mileage_boats_yrs['Medium'].loc[START_YEAR:], axis=0)  
ship_large_nr  = ship_large_vehkm.div(mileage_boats_yrs['Large'].loc[START_YEAR:], axis=0)  
ship_vlarge_nr = ship_vlarge_vehkm.div(mileage_boats_yrs['Very Large'].loc[START_YEAR:], axis=0) 

# for comparison, we find the difference of the known and the calculated nr of ships (global total) in the period 2005-2018
diff_ships = pd.DataFrame().reindex_like(nr_of_boats)
diff_ships['Small']      =  ship_small_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Small'])
diff_ships['Medium']     =  ship_medium_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Medium'])
diff_ships['Large']      =  ship_large_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Large'])
diff_ships['Very Large'] =  ship_vlarge_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Very Large'])

total_nr_of_ships = ship_small_nr + ship_medium_nr + ship_large_nr + ship_vlarge_nr
diff_ships_total = total_nr_of_ships.loc[list(range(2005,2018+1)), 28].div(nr_of_boats.sum(axis=1))

#%% Export intermediate indicators (a.o. files on nr. of vehicles, pkms/tkms)

# Export total global number of vehicles in the fleet (stock) as csv
region_list = list(range(1,27))
index = pd.MultiIndex.from_product([list(total_nr_of_ships.index), region_list], names = ['years','regions'])
total_nr_vehicles = pd.DataFrame(index=index, columns=columns_vehcile_output)
total_nr_vehicles['Buses']        = bus_regl_nr[region_list].stack() + bus_midi_nr[region_list].stack()
total_nr_vehicles['Trains']       = rail_reg_nr[region_list].stack()
total_nr_vehicles['HST']          = rail_hst_nr[region_list].stack()
total_nr_vehicles['Cars']         = car_total_nr[region_list].stack()
total_nr_vehicles['Planes']       = air_pas_nr[region_list].stack()
total_nr_vehicles['Bikes']        = bikes_nr[region_list].stack()
total_nr_vehicles['Trucks']       = trucks_HFT_nr[region_list].stack() + trucks_MFT_nr[region_list].stack() + trucks_LCV_nr[region_list].stack()
total_nr_vehicles['Cargo Trains'] = rail_freight_nr[region_list].stack()
total_nr_vehicles['Ships']        = total_nr_of_ships[region_list].stack()
total_nr_vehicles['Inland ships'] = inland_ship_nr[region_list].stack()  
total_nr_vehicles['Cargo Planes'] = air_freight_nr[region_list].stack() 

total_nr_vehicles.sum(axis=0, level=0).to_csv(OUTPUT_FOLDER + '\\global_vehicle_nr.csv', index=True) # total global nr of vehicles 
total_nr_vehicles.to_csv(OUTPUT_FOLDER +  '\\region_vehicle_nr.csv', index=True) # regional nr of vehicles 

# Generate csv output file on pkms & tkms (same format as files on number of vehicles (also used later on)), unit: pkm or tkm 
region_list = list(range(1,27))
car_pkms.columns = list(range(1,27))      # remove region labels 
index = pd.MultiIndex.from_product([list(total_nr_of_ships.index), region_list], names = ['years','regions'])
total_pkm_tkm = pd.DataFrame(index=index, columns=columns_vehcile_output)
total_pkm_tkm['Buses']        = (bus_regl_pkms[region_list].stack() + bus_midi_pkms[region_list].stack()) * 1000000000000
total_pkm_tkm['Trains']       = passengerkms_Tpkms['train']   * 1000000000000                              
total_pkm_tkm['HST']          = passengerkms_Tpkms['hst']     * 1000000000000
total_pkm_tkm['Cars']         = car_pkms[region_list].stack() * 1000000000000
total_pkm_tkm['Planes']       = passengerkms_Tpkms['air']     * 1000000000000
total_pkm_tkm['Bikes']        = passengerkms_Tpkms['biking']  * 1000000000000
total_pkm_tkm['Trucks']       = (trucks_HFT_tkm[region_list].stack() + trucks_MFT_tkm[region_list].stack() + trucks_LCV_tkm[region_list].stack()) * 1000000
total_pkm_tkm['Cargo Trains'] = tonkms_Mtkms['freight train'] * 1000000
total_pkm_tkm['Ships']        = tonkms_Mtkms['international shipping'] * 1000000 
total_pkm_tkm['Inland ships'] = tonkms_Mtkms['inland shipping'] * 1000000
total_pkm_tkm['Cargo Planes'] = air_freight_tkms[region_list].stack() * 1000000     # mind that these are the tkms flows with cargo planes, real demand for air cargo is higher due to 50% hitching with passenger flights

total_pkm_tkm.sum(axis=0, level=0).to_csv(OUTPUT_FOLDER + '\\global_pkm_tkm.csv', index=True) # total global pkms & tkms 
total_pkm_tkm.to_csv(OUTPUT_FOLDER + '\\region_pkm_tkm.csv', index=True)  # regional pkms & tkms 

# some indicators for the model accuracy comparison
inland_ship_nr[[2,11,16,20]].loc[2015].sum()           # inland shipping in 2015 (China, Russia, Europe & US)
car_total_nr[[11,12]].loc[2018].sum()
rail_reg_nr[[1,2,11,12,18,20,23]].loc[2017].sum()      # India, Canada, China, United States, Europe, Japan
rail_freight_nr[[1,2,11,12,18,20,16]].loc[2016].sum()  # India, Canada, China, United States, Europe, Russia
