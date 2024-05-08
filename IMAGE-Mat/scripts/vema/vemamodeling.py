

#%% INFLOW-OUTFLOW calculations using the ODYM Dynamic Stock Model (DSM) as a function

# In order to calculate inflow & outflow smoothly (without peaks for the initial years), 
def stock_tail(stock, first_year_veh, choice='stock'):
    """
    In order to avoid an initial inflow shock, this function calculates the size of the initial development of the vehicle stock (in number of vehicles) before the first scenario year of the IMAGE model (1971).
    It does so by adding a so-called historic tail to the stock, by adding a 0 value for first year of operation (e.g. 1926), then linear interpolation of values towards 1971    
    """
    if choice=='stock':
       zero_value = [0 for i in range(0, REGIONS)]
    else:
       zero_value = stock.head(1).values[0]
       
    stock_used = pd.DataFrame(stock).reindex_like(stock)
    stock_used.loc[first_year_veh] = zero_value  # set all regions to 0 in the year of initial operation
    stock_new  = stock_used.reindex(list(range(first_year_veh, END_YEAR + 1))).interpolate()
    return stock_new

# lifetimes also need a tail, but this doesn't start at 0, but remains constant at the first known value (assumed)
def lifetime_tail(lifetime_vehicles_in, vehicle):
    first_year = first_year_vehicle[vehicle].values[0]
    lifetime_vehicles_out = lifetime_vehicles_in[vehicle]
    for year in range(first_year, START_YEAR):
        lifetime_vehicles_out[year] = lifetime_vehicles_in[vehicle][START_YEAR]
    return lifetime_vehicles_out.reindex(list(range(first_year, END_YEAR + 1)))

# apply the dynamic stock model
def inflow_outflow_dynamic_np(stock, fact1, fact2, distribution):
    """
    In this function the stock-driven DSM is applied to return inflow & outflow for all regions
    IN: stock as a dataframe (nr of vehicles by year & region), lifetime (in years), stdev_mult (multiplier to derive the standard deviation from the mean)
    OUT: 3 dataframes of stock, inflow & outflow (nr of vehicles by year^2 & region)
    
    This DYNAMIC variant of the DSM calculates & returns the full unadjusted (cohort specific) matrix (YES, very memory intensive, but neccesary for dynamic assesmment, i.e. changing vehicle weights & compositions)
    """
    inflow         = np.empty((len(stock[0]), len(stock)))
    outflow_cohort = np.empty((len(stock[0]), len(stock), len(stock))) 
    stock_cohort   = np.empty((len(stock[0]), len(stock), len(stock))) 

    # define mean (fact1) & standard deviation (fact2) (applied accross all regions) (For Weibull fact1 = shape & fact2 = scale)
                     
    # fact 1 is a list with the mean lifetime (for Weibull: it's the shape parameter), given for every timestep (e.g. 1900-2100), needed for the DSM as it allows to change lifetime for different cohort
    # fact 2 is a list with the standarddeviation as a fraction of the mean lifetime (for Weibull: it's the scale parameter)
    if distribution == 'FoldedNormal':
       fact2_list = fact1.mul(fact2)   # for the (Folded) normal distribution, the standard_deviation is calculated as the mean times the multiplier (i.e. the standard deviation as a fraction of the mean)
    else:
       fact2_list = fact2

    for region in range(0,len(stock[0])):
        # define and run the DSM   
        if distribution == 'FoldedNormal':
           DSMforward = DSM(t = np.arange(0,len(stock[:,region]),1), s=stock[:,region], lt = {'Type': 'FoldedNormal', 'Mean': np.array(fact1), 'StdDev': np.array(fact2_list)})  # definition of the DSM based on a folded normal distribution
        else:
           DSMforward = DSM(t = np.arange(0,len(stock[:,region]),1), s=stock[:,region], lt = {'Type': 'Weibull', 'Shape': np.array(fact1), 'Scale': np.array(fact2_list)})       # definition of the DSM based on a Weibull distribution
           
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model_surplus()                                                                 # run the DSM, to give 3 outputs: stock_by_cohort, outflow_by_cohort & inflow_per_year
        
        # store the regional results in the return dataframe (full range of years (e.g. 1900 onwards), for later use in material calculations  & no moving average applied here)
        outflow_cohort[region,:,:] = out_oc                                                                                                      # sum the outflow by cohort to get the total outflow per year
        stock_cohort[region,:,:]   = out_sc
        inflow[region,:]           = out_i
         
    return inflow.T, outflow_cohort, stock_cohort     


# Pre-calculate the inflow (aka. market-)share corresponding to the (known) share of vehicles in stock (from IMAGE)
def inflow_outflow_typical_np(stock, fact1, fact2, distribution, stock_share):
   """
   In this function the dynamic inflow & outflow calculations are applied to vehicles with sub-types. To return inflow & outflow for all regions & vehicle types
   IN: stock as a dataframe (nr of vehicles by year, region and type), lifetime (in years), stdev_mult (multiplier to derive the standard deviation from the mean)
   OUT: 3 dataframes of stock, inflow & outflow (nr of vehicles by year^2 & region & type)
   
   This TYPICAL variant of the DSM calculates & returns the full unadjusted (cohort specific) matrix (time*time) for every region AND vehicle type (YES, very memory intensive, but neccesary for dynamic assesmment, i.e. changing vehicle weights & compositions)
   """
   initial_year        = stock.first_valid_index()
   initial_year_shares = stock_share.first_valid_index()[0]
   
   inflow              = np.zeros((len(stock_share.iloc[0]), len(stock.iloc[0]), len(stock)))
   outflow_cohort      = np.zeros((len(stock_share.iloc[0]), len(stock.iloc[0]), len(stock), len(stock)))
   stock_cohort        = np.zeros((len(stock_share.iloc[0]), len(stock.iloc[0]), len(stock), len(stock)))

   index = pd.MultiIndex.from_product([stock_share.columns, stock.index], names=['type', 'time'])
   stock_by_vtype      = pd.DataFrame(0, index=index, columns=stock.columns)
   stock_share_used    = stock_share.unstack().stack(level=0).reorder_levels([1,0]).sort_index()
   
   # before running the DSM: 
   # 1) extend the historic coverage of the stock shares (assuming constant pre-1971), and
   for vtype in list(stock_share.columns):
      stock_share_used.loc[idx[vtype, initial_year],:] = stock_share[vtype].unstack().loc[initial_year_shares]
   stock_share_used = stock_share_used.reindex(index).interpolate()
   
   # 2) calculate the stocks of individual (vehicle) types (in nr of vehicles)
   for vtype in list(stock_share.columns):
      stock_by_vtype.loc[idx[vtype,:],:] = stock.mul(stock_share_used.loc[idx[vtype,:],:])

   vtype_list  = list(stock_by_vtype.index.unique('type'))
   vtype_count = list(range(0,len(stock_share.columns))) 
   vtype_dict   = dict(zip(vtype_count, vtype_list))

   # Then run the original DSM for each vehicle type & add it to the inflow, ouflow & stock containers, (for stock: index = vtype, time; columns = region, time)
   for vtype in range(0,len(stock_share.columns)): 
      if stock_share.iloc[:,vtype].sum() > 0.001:
         dsm_inflow, dsm_outflow_coh, dsm_stock_coh = inflow_outflow_dynamic_np(stock_by_vtype.loc[idx[vtype_dict[vtype],:],:].to_numpy(), fact1, fact2, distribution)
         
         inflow[vtype,:,:]           = dsm_inflow.T
         outflow_cohort[vtype,:,:,:] = dsm_outflow_coh
         stock_cohort[vtype,:,:,:]   = dsm_stock_coh
      
      else:
         pass
        
   return inflow, outflow_cohort, stock_cohort

# Calculate the historic tail (& reduce regions to 26)
air_pas_nr     = stock_tail(air_pas_nr[list(range(1,REGIONS+1))],  first_year_vehicle['air_pas'].values[0])
rail_reg_nr    = stock_tail(rail_reg_nr[list(range(1,REGIONS+1))], first_year_vehicle['rail_reg'].values[0])
rail_hst_nr    = stock_tail(rail_hst_nr[list(range(1,REGIONS+1))], first_year_vehicle['rail_hst'].values[0])
bikes_nr       = stock_tail(bikes_nr[list(range(1,REGIONS+1))],    first_year_vehicle['bicycle'].values[0])

air_freight_nr  = stock_tail(air_freight_nr[list(range(1, REGIONS+1))],  first_year_vehicle['air_freight'].values[0])
rail_freight_nr = stock_tail(rail_freight_nr[list(range(1, REGIONS+1))], first_year_vehicle['rail_freight'].values[0])
inland_ship_nr  = stock_tail(inland_ship_nr[list(range(1, REGIONS+1))],  first_year_vehicle['inland_shipping'].values[0])
ship_small_nr   = stock_tail(ship_small_nr[list(range(1, REGIONS+1))],   1900)
ship_medium_nr  = stock_tail(ship_medium_nr[list(range(1, REGIONS+1))],  1900)
ship_large_nr   = stock_tail(ship_large_nr[list(range(1, REGIONS+1))],   1900)
ship_vlarge_nr  = stock_tail(ship_vlarge_nr[list(range(1, REGIONS+1))],  1900)

bus_regl_nr     = stock_tail(bus_regl_nr[list(range(1, REGIONS+1))],  first_year_vehicle['reg_bus'].values[0])
bus_midi_nr     = stock_tail(bus_midi_nr[list(range(1, REGIONS+1))],  first_year_vehicle['midi_bus'].values[0])
car_total_nr    = stock_tail(car_total_nr[list(range(1, REGIONS+1))], first_year_vehicle['car'].values[0])

trucks_HFT_nr   = stock_tail(trucks_HFT_nr[list(range(1, REGIONS+1))], first_year_vehicle['HFT'].values[0])
trucks_MFT_nr   = stock_tail(trucks_HFT_nr[list(range(1, REGIONS+1))], first_year_vehicle['MFT'].values[0])
trucks_LCV_nr   = stock_tail(trucks_LCV_nr[list(range(1, REGIONS+1))], first_year_vehicle['LCV'].values[0])

##################### DYNAMIC MODEL (runtime: ca. 30 sec) ########################################################################
# Calculate the NUMBER of vehicles, total for inflow & by cohort for stock & outflow, first only for simple vehicles

air_pas_in,      air_pas_out_coh,      air_pas_stock_coh      = inflow_outflow_dynamic_np(air_pas_nr.to_numpy(),    lifetime_tail(lifetimes_vehicles_mean,'air_pas'),  lifetime_tail(lifetimes_vehicles_stdev,'air_pas'),  'FoldedNormal')
rail_reg_in,     rail_reg_out_coh,     rail_reg_stock_coh     = inflow_outflow_dynamic_np(rail_reg_nr.to_numpy(),   lifetime_tail(lifetimes_vehicles_mean,'rail_reg'), lifetime_tail(lifetimes_vehicles_stdev,'rail_reg'), 'FoldedNormal')
rail_hst_in,     rail_hst_out_coh,     rail_hst_stock_coh     = inflow_outflow_dynamic_np(rail_hst_nr.to_numpy(),   lifetime_tail(lifetimes_vehicles_mean,'rail_hst'), lifetime_tail(lifetimes_vehicles_stdev,'rail_hst'), 'FoldedNormal')
bikes_in,        bikes_out_coh,        bikes_stock_coh        = inflow_outflow_dynamic_np(bikes_nr.to_numpy(),      lifetime_tail(lifetimes_vehicles_mean,'bicycle'),  lifetime_tail(lifetimes_vehicles_stdev,'bicycle'),  'FoldedNormal')

air_freight_in,  air_freight_out_coh,  air_freight_stock_coh  = inflow_outflow_dynamic_np(air_freight_nr.to_numpy(),  lifetime_tail(lifetimes_vehicles_mean,'air_freight'),        lifetime_tail(lifetimes_vehicles_stdev,'air_freight'),        'FoldedNormal')
rail_freight_in, rail_freight_out_coh, rail_freight_stock_coh = inflow_outflow_dynamic_np(rail_freight_nr.to_numpy(), lifetime_tail(lifetimes_vehicles_mean,'rail_freight'),       lifetime_tail(lifetimes_vehicles_stdev,'rail_freight'),       'FoldedNormal')
inland_ship_in,  inland_ship_out_coh,  inland_ship_stock_coh  = inflow_outflow_dynamic_np(inland_ship_nr.to_numpy(),  lifetime_tail(lifetimes_vehicles_mean,'inland_shipping'),    lifetime_tail(lifetimes_vehicles_stdev,'inland_shipping'),    'FoldedNormal')
ship_small_in,   ship_small_out_coh,   ship_small_stock_coh   = inflow_outflow_dynamic_np(ship_small_nr.to_numpy(),   lifetime_tail(lifetimes_vehicles_mean,'sea_shipping_small'), lifetime_tail(lifetimes_vehicles_stdev,'sea_shipping_small'), 'FoldedNormal')
ship_medium_in,  ship_medium_out_coh,  ship_medium_stock_coh  = inflow_outflow_dynamic_np(ship_medium_nr.to_numpy(),  lifetime_tail(lifetimes_vehicles_mean,'sea_shipping_med'),   lifetime_tail(lifetimes_vehicles_stdev,'sea_shipping_med'),   'FoldedNormal')
ship_large_in,   ship_large_out_coh,   ship_large_stock_coh   = inflow_outflow_dynamic_np(ship_large_nr.to_numpy(),   lifetime_tail(lifetimes_vehicles_mean,'sea_shipping_large'), lifetime_tail(lifetimes_vehicles_stdev,'sea_shipping_large'), 'FoldedNormal')
ship_vlarge_in,  ship_vlarge_out_coh,  ship_vlarge_stock_coh  = inflow_outflow_dynamic_np(ship_vlarge_nr.to_numpy(),  lifetime_tail(lifetimes_vehicles_mean,'sea_shipping_vl'),    lifetime_tail(lifetimes_vehicles_stdev,'sea_shipping_vl'),    'FoldedNormal')

#%% BATTERY VEHICLE CALCULATIONS - Determine the fraction of the fleet that uses batteries, based on vehicle share files
# Batteries are relevant for 1) BUSES 2) TRUCKS
# We use fixed weight & material content assumptions, but we use the development of battery energy density (from the electricity storage calculations) to derive a changing battery capacity (and thus range)
# Battery weight is assumed to be in-addition to the regular vehicle weight

bus_label        = ['BusOil',	'BusBio',	'BusGas',	'BusElecTrolley',	'Bus Hybrid1',	'Bus Hybrid2',	'BusBattElectric', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
bus_label_ICE    = ['BusOil',	'BusBio',	'BusGas']
bus_label_HEV    = ['BusElecTrolley',	'Bus Hybrid1']
truck_label      = ['Conv. ICE(2000)',	'Conv. ICE(2010)', 'Adv. ICEOil',	'Adv. ICEH2', 'Turbo-petrol IC', 'Diesel ICEOil', 'Diesel ICEBio', 'ICE-HEV-gasoline', 'ICE-HEV-diesel oil', 'ICE-HEV-H2', 'ICE-HEV-CNG Gas', 'ICE-HEV-diesel bio', 'FCV Oil', 'FCV Bio', 'FCV H2', 'PEV-10 OilElec.', 'PEV-30 OilElec.', 'PEV-60 OilElec.', 'PEV-10 BioElec.', 'PEV-30 BioElec.', 'PEV-60 BioElec.', 'BEV Elec.', 'BEV Elec.', 'BEV Elec.', 'BEV Elec.']
truck_label_ICE  = ['Conv. ICE(2000)',	'Conv. ICE(2010)', 'Adv. ICEOil', 'Adv. ICEH2', 'Turbo-petrol IC', 'Diesel ICEOil', 'Diesel ICEBio']
truck_label_HEV  = ['ICE-HEV-gasoline', 'ICE-HEV-diesel oil', 'ICE-HEV-H2', 'ICE-HEV-CNG Gas', 'ICE-HEV-diesel bio']
truck_label_PHEV = ['PEV-10 OilElec.', 'PEV-30 OilElec.', 'PEV-60 OilElec.', 'PEV-10 BioElec.', 'PEV-30 BioElec.', 'PEV-60 BioElec.']
truck_label_BEV  = ['BEV Elec.', 'BEV Elec.', 'BEV Elec.']
truck_label_FCV  = ['FCV Oil', 'FCV Bio', 'FCV H2']
vshares_label    = ['ICE', 'HEV', 'PHEV', 'BEV', 'FCV', 'Trolley']

# 1) BUSES: original vehcile shares are distributed into two vehicle types (regular and small midi buses)
# vehicle shares are grouped as: a) ICE, b) HEV, c) trolley, d) BEV, but trolley buses are not relevant for midi busses, so the midi shares are re-calculated based on the sum without trolleys
midi_sum = buses_vshares[filter(lambda x: x != 'BusElecTrolley',bus_label)].sum(axis=1)     #sum of all except Trolleys

# regular buses are just grouped
buses_regl_vshares = pd.DataFrame(index=buses_vshares.index, columns=vshares_label)
buses_regl_vshares['ICE']     = buses_vshares[bus_label_ICE].sum(axis=1)
buses_regl_vshares['HEV']     = buses_vshares[bus_label_HEV].sum(axis=1)
buses_regl_vshares['PHEV']    = pd.DataFrame(0, index=buses_vshares.index, columns=['PHEV'])
buses_regl_vshares['BEV']     = buses_vshares['BusBattElectric']
buses_regl_vshares['FCV']     = pd.DataFrame(0, index=buses_vshares.index, columns=['FCV'])
buses_regl_vshares['Trolley'] = buses_vshares['BusElecTrolley']

# midi buses are grouped & divided by the sum of ICE, HEV & BEV, to adjust for the fact that Trolleys (or FCV or PHEV) are not an option for midi buses
buses_midi_vshares = pd.DataFrame(index=buses_vshares.index, columns=vshares_label)
buses_midi_vshares['ICE']     = buses_vshares[bus_label_ICE].sum(axis=1).div(midi_sum)
buses_midi_vshares['HEV']     = buses_vshares[bus_label_HEV].sum(axis=1).div(midi_sum)
buses_midi_vshares['PHEV']    = pd.DataFrame(0, index=buses_vshares.index, columns=['PHEV'])
buses_midi_vshares['BEV']     = buses_vshares['BusBattElectric'].div(midi_sum)
buses_midi_vshares['FCV']     = pd.DataFrame(0, index=buses_vshares.index, columns=['FCV'])
buses_midi_vshares['Trolley'] = pd.DataFrame(0, index=buses_vshares.index, columns=['Trolley'])

# 2) TRUCKS
# vehicle shares are grouped as: a) ICE, b) HEV, c) PHEV, d) BEV, e) FCV
# LCV vehicle shares are determined based on the medium trucks (so not calculated explicitly)

# medium trucks
trucks_MFT_vshares = pd.DataFrame(index=medtruck_vshares.index, columns=vshares_label)
trucks_MFT_vshares['ICE']     = medtruck_vshares[truck_label_ICE].sum(axis=1)
trucks_MFT_vshares['HEV']     = medtruck_vshares[truck_label_HEV].sum(axis=1)
trucks_MFT_vshares['PHEV']    = medtruck_vshares[truck_label_PHEV].sum(axis=1)
trucks_MFT_vshares['BEV']     = medtruck_vshares[truck_label_BEV].sum(axis=1)
trucks_MFT_vshares['FCV']     = medtruck_vshares[truck_label_FCV].sum(axis=1)
trucks_MFT_vshares['Trolley'] = pd.DataFrame(0, index=index, columns=['Trolley'])                    # No trolley trucks 

# heavy trucks
trucks_HFT_vshares = pd.DataFrame(index=hvytruck_vshares.index, columns=vshares_label)
trucks_HFT_vshares['ICE']     = hvytruck_vshares[truck_label_ICE].sum(axis=1)
trucks_HFT_vshares['HEV']     = hvytruck_vshares[truck_label_HEV].sum(axis=1)
trucks_HFT_vshares['PHEV']    = hvytruck_vshares[truck_label_PHEV].sum(axis=1)
trucks_HFT_vshares['BEV']     = hvytruck_vshares[truck_label_BEV].sum(axis=1)
trucks_HFT_vshares['FCV']     = hvytruck_vshares[truck_label_FCV].sum(axis=1)
trucks_HFT_vshares['Trolley'] = pd.DataFrame(0, index=hvytruck_vshares.index, columns=['Trolley'])   # No trolley trucks 

### Then calculate the inflow & outflow for typical vehicles (vehicles with relevant sub types) as well (runtime appr. 1 min.)
bus_regl_in,     bus_regl_out_coh,     bus_regl_stock_coh     = inflow_outflow_typical_np(bus_regl_nr,    lifetime_tail(lifetimes_vehicles_mean, 'reg_bus'),  lifetime_tail(lifetimes_vehicles_stdev, 'reg_bus'),  'FoldedNormal', buses_regl_vshares)
bus_midi_in,     bus_midi_out_coh,     bus_midi_stock_coh     = inflow_outflow_typical_np(bus_midi_nr,    lifetime_tail(lifetimes_vehicles_mean, 'midi_bus'), lifetime_tail(lifetimes_vehicles_stdev, 'midi_bus'), 'FoldedNormal', buses_midi_vshares)
car_in,          car_out_coh,          car_stock_coh          = inflow_outflow_typical_np(car_total_nr,   lifetime_tail(lifetimes_vehicles_shape, 'car'),     lifetime_tail(lifetimes_vehicles_scale, 'car'),      'Weibull',      vehicleshare_cars)

trucks_HFT_in,   trucks_HFT_out_coh,   trucks_HFT_stock_coh   = inflow_outflow_typical_np(trucks_HFT_nr,  lifetime_tail(lifetimes_vehicles_mean, 'HFT'),     lifetime_tail(lifetimes_vehicles_stdev, 'HFT'),      'FoldedNormal',  trucks_HFT_vshares)
trucks_MFT_in,   trucks_MFT_out_coh,   trucks_MFT_stock_coh   = inflow_outflow_typical_np(trucks_MFT_nr,  lifetime_tail(lifetimes_vehicles_mean, 'MFT'),     lifetime_tail(lifetimes_vehicles_stdev, 'MFT'),      'FoldedNormal',  trucks_MFT_vshares)
trucks_LCV_in,   trucks_LCV_out_coh,   trucks_LCV_stock_coh   = inflow_outflow_typical_np(trucks_LCV_nr,  lifetime_tail(lifetimes_vehicles_mean, 'LCV'),     lifetime_tail(lifetimes_vehicles_stdev, 'LCV'),      'FoldedNormal',  trucks_MFT_vshares)  # Assumption: used MFT as a market-share for LCVs

#%% Intermediate export of inflow & outflow of vehicles (for IRP database) ###############

region_list = list(range(1,27))
last_years  = END_YEAR+1 - 1971
index = pd.MultiIndex.from_product([list(total_nr_of_ships.index), region_list], names = ['years','regions'])
total_nr_vehicles_in = pd.DataFrame(index=index, columns=columns_vehcile_output)
total_nr_vehicles_in['Buses']        = bus_regl_in.sum(0)[:,-last_years:].flatten(order='F') + bus_midi_in.sum(0)[:,-last_years:].flatten(order='F')  #flattening a numpy array in the expected order (year columns first)
total_nr_vehicles_in['Trains']       = rail_reg_in[-last_years:,:].flatten(order='C')  # for simple arrays (no vehicle types) the column order is reversed
total_nr_vehicles_in['HST']          = rail_hst_in[-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Cars']         = car_in.sum(0)[:,-last_years:].flatten(order='F')
total_nr_vehicles_in['Planes']       = air_pas_in[-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Bikes']        = bikes_in[-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Trucks']       = trucks_HFT_in.sum(0)[:,-last_years:].flatten(order='F') + trucks_MFT_in.sum(0)[:,-last_years:].flatten(order='F') + trucks_LCV_in.sum(0)[:,-last_years:].flatten(order='F')
total_nr_vehicles_in['Cargo Trains'] = rail_freight_in[-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Ships']        = ship_small_in[-last_years:,:].flatten(order='C') + ship_medium_in[-last_years:,:].flatten(order='C') + ship_large_in[-last_years:,:].flatten(order='C') + ship_vlarge_in[-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Inland ships'] = inland_ship_in[-last_years:,:].flatten(order='C')  
total_nr_vehicles_in['Cargo Planes'] = air_freight_in[-last_years:,:].flatten(order='C')

total_nr_vehicles_in.to_csv(OUTPUT_FOLDER + '\\region_vehicle_in.csv', index=True) # regional nr of vehicles sold (annually)

total_nr_vehicles_out = pd.DataFrame(index=index, columns=columns_vehcile_output)
total_nr_vehicles_out['Buses']        = bus_regl_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F') + bus_midi_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F')  #flattening a numpy array in the expected order (year columns first)
total_nr_vehicles_out['Trains']       = rail_reg_out_coh.sum(-1)[:,-last_years:].flatten(order='F')  # for simple arrays (no vehicle types) the column order is reversed
total_nr_vehicles_out['HST']          = rail_hst_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Cars']         = car_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Planes']       = air_pas_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Bikes']        = bikes_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Trucks']       = trucks_HFT_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F') + trucks_MFT_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F') + trucks_LCV_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Cargo Trains'] = rail_freight_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Ships']        = ship_small_out_coh.sum(-1)[:,-last_years:].flatten(order='F') + ship_medium_out_coh.sum(-1)[:,-last_years:].flatten(order='F') + ship_large_out_coh.sum(-1)[:,-last_years:].flatten(order='F') + ship_vlarge_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Inland ships'] = inland_ship_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Cargo Planes'] = air_freight_out_coh.sum(-1)[:,-last_years:].flatten(order='F')

total_nr_vehicles_out.to_csv(OUTPUT_FOLDER + '\\region_vehicle_out.csv', index=True) # regional nr of vehicles sold (annually)


#%% ################### MATERIAL CALCULATIONS ##########################################

# Material calculations for vehicles with only 1 relevant sub-type
def nr_by_cohorts_to_materials_simple_np(inflow, outflow_cohort, stock_cohort, weight, composition):
   """
   for those vehicles with only 1 relevant sub-type, we calculate the material stocks & flows as: 
   Nr * weight (kg) * composition (%)
   INPUT:  -------------- 
   inflow         = numpy array (time, regions)      - number of vehicles by region over time
   outflow_cohort = numpy array (region, time, time) - number of vehicles in outlfow by region over time, by built year
   stock_cohort   = numpy array (region, time, time) - number of vehicles in stock by region over time, by built year
   OUTPUT: --------------
   pd_inflow_mat  = pandas dataframe (index: material, time; column: regions) - inflow of materials in kg of material, by region & time
   pd_outflow_mat = pandas dataframe (index: material, time; column: regions) - inflow of materials in kg of material, by region & time
   pd_stock_mat   = pandas dataframe (index: material, time; column: regions) - inflow of materials in kg of material, by region & time
   """
   inflow_mat  = np.zeros((len(composition.columns), len(inflow[0]), len(inflow)))
   outflow_mat = np.zeros((len(composition.columns), len(inflow[0]), len(inflow)))
   stock_mat   = np.zeros((len(composition.columns), len(inflow[0]), len(inflow)))
   
   for material in range(0, len(composition.columns)): 
      # before running, check if the material is at all relevant in the vehicle (save calculation time)
      if composition.iloc[:,material].sum() > 0.001:          
         for region in range(0,len(inflow[0])):
            composition_used = composition.iloc[:,material].values
            inflow_mat[material,region,:]  = (inflow[:,region] * weight) * composition_used
            outflow_mat[material,region,:] = np.multiply(np.multiply(outflow_cohort[region,:,:].T, weight), composition_used).T.sum(axis=1)
            stock_mat[material,region,:]   = np.multiply(np.multiply(stock_cohort[region,:,:], weight), composition_used).sum(axis=1)
      else:
         pass

   length_materials = len(composition.columns)
   length_time      = END_YEAR + 1 - (END_YEAR + 1 - len(inflow))

   index          = pd.MultiIndex.from_product([composition.columns, range(END_YEAR + 1 - len(inflow), END_YEAR + 1)], names=['time', 'type'])
   pd_inflow_mat  = pd.DataFrame(inflow_mat.transpose(0,2,1).reshape((length_materials * length_time), REGIONS),  index=index, columns=range(1,len(inflow[0]) + 1))
   pd_outflow_mat = pd.DataFrame(outflow_mat.transpose(0,2,1).reshape((length_materials * length_time), REGIONS), index=index, columns=range(1,len(inflow[0]) + 1))
   pd_stock_mat   = pd.DataFrame(stock_mat.transpose(0,2,1).reshape((length_materials * length_time), REGIONS),   index=index, columns=range(1,len(inflow[0]) + 1))
      
   return pd_inflow_mat, pd_outflow_mat, pd_stock_mat


# Material calculations for vehicles with only multiple sub-types
def nr_by_cohorts_to_materials_typical_np(inflow, outflow_cohort, stock_cohort, weight, composition):
   """
   for those vehicles with multiple sub-type, we calculate the material stocks & flows in the same way, so:
   Nr * weight (kg) * composition (%) - but using data on specific vehicle sub-types
   """
   inflow_mat  = np.zeros((len(inflow), len(composition.columns.levels[1]), len(inflow[0]), len(inflow[0][0])))
   outflow_mat = np.zeros((len(inflow), len(composition.columns.levels[1]), len(inflow[0]), len(inflow[0][0])))
   stock_mat   = np.zeros((len(inflow), len(composition.columns.levels[1]), len(inflow[0]), len(inflow[0][0])))
   
   composition.columns.names = ['type', 'material']
   
   # get two dictionaries to keep tarck of the order of materials & vtypes while using numpy
   vtype_list   = list(composition.columns.unique('type'))  # columns.unique() keeps the original order, which is important here (same for materials)
   vtype_count  = list(range(0,len(inflow))) 
   vtype_dict   = dict(zip(vtype_count, vtype_list))
   print(vtype_dict)
   mater_list   = list(composition.columns.unique('material'))
   mater_count  = list(range(0,len(composition.columns.levels[1]))) 
   mater_dict   = dict(zip(mater_count, mater_list))
   print(mater_dict)
   
   for vtype in range(0, len(inflow)): 
      # before running, check if the vehicle type is at all relevant in the vehicle (save calculation time)
      weight_used = weight.loc[:,vtype_dict[vtype]].values
      if stock_cohort[vtype].sum() > 0.001:  
         for material in range(0, len(mater_list)): 
            composition_used = composition.loc[:,idx[vtype_dict[vtype],mater_dict[material]]].values
            # before running, check if the material is at all relevant in the vehicle (save calculation time)
            if composition_used.sum() > 0.001:          
               for region in range(0, len(inflow[0])):
                  inflow_mat[vtype, material, region, :]  = (inflow[vtype, region, :] * weight_used) * composition_used
                  outflow_mat[vtype, material, region, :] = np.multiply(np.multiply(outflow_cohort[vtype, region, :, :].T, weight_used), composition_used).T.sum(axis=1)
                  stock_mat[vtype, material, region, :]   = np.multiply(np.multiply(stock_cohort[vtype, region, :, :], weight_used), composition_used).sum(axis=1)
   
            else:
               pass
      else:
         pass
   
   length_materials = len(composition.columns.levels[1])
   length_time      = END_YEAR + 1 - (END_YEAR + 1 - len(inflow[0][0]))
   
   #return as pandas dataframe, just once
   index          = pd.MultiIndex.from_product([mater_list, list(range((END_YEAR + 1 - len(inflow[0][0])), END_YEAR + 1))], names=['material', 'time'])
   columns        = pd.MultiIndex.from_product([vtype_list, range(1,REGIONS+1)],      names=['type', 'region'])
   pd_inflow_mat  = pd.DataFrame(inflow_mat.transpose(1,3,0,2).reshape((length_materials * length_time), (len(inflow) * REGIONS)), index=index, columns=columns)
   pd_outflow_mat = pd.DataFrame(outflow_mat.transpose(1,3,0,2).reshape((length_materials * length_time),(len(inflow) * REGIONS)), index=index, columns=columns)
   pd_stock_mat   = pd.DataFrame(stock_mat.transpose(1,3,0,2).reshape((length_materials * length_time),  (len(inflow) * REGIONS)), index=index, columns=columns)
         
   return pd_inflow_mat, pd_outflow_mat, pd_stock_mat


# capacity of boats is in tonnes, the weight - expressed as a fraction of the capacity - is calculated in in kgs here
weight_boats  = weight_frac_boats_yrs * cap_of_boats_yrs * 1000 

#%% ############################################# RUNNING THE DYNAMIC STOCK FUNCTIONS  (runtime ca. 10 sec)  ###############################################


# run the simple material calculations on all vehicles                                                                                             # weight is passed as a series instead of a dataframe (pragmatic choice)
air_pas_mat_in,      air_pas_mat_out,      air_pas_mat_stock        = nr_by_cohorts_to_materials_simple_np(air_pas_in,      air_pas_out_coh,      air_pas_stock_coh,      vehicle_weight_kg_air_pas["air_pas"].to_numpy(),             material_fractions_air_pas)
rail_reg_mat_in,     rail_reg_mat_out,     rail_reg_mat_stock       = nr_by_cohorts_to_materials_simple_np(rail_reg_in,     rail_reg_out_coh,     rail_reg_stock_coh,     vehicle_weight_kg_rail_reg["rail_reg"].to_numpy(),           material_fractions_rail_reg)
rail_hst_mat_in,     rail_hst_mat_out,     rail_hst_mat_stock       = nr_by_cohorts_to_materials_simple_np(rail_hst_in,     rail_hst_out_coh,     rail_hst_stock_coh,     vehicle_weight_kg_rail_hst["rail_hst"].to_numpy(),           material_fractions_rail_hst)
bikes_mat_in,        bikes_mat_out,        bikes_mat_stock          = nr_by_cohorts_to_materials_simple_np(bikes_in,        bikes_out_coh,        bikes_stock_coh,        vehicle_weight_kg_bicycle["bicycle"].to_numpy(),             material_fractions_bicycle)

air_freight_mat_in,  air_freight_mat_out,  air_freight_mat_stock    = nr_by_cohorts_to_materials_simple_np(air_freight_in,  air_freight_out_coh,  air_freight_stock_coh,  vehicle_weight_kg_air_frgt["air_freight"].to_numpy(),        material_fractions_air_frgt)
rail_freight_mat_in, rail_freight_mat_out, rail_freight_mat_stock   = nr_by_cohorts_to_materials_simple_np(rail_freight_in, rail_freight_out_coh, rail_freight_stock_coh, vehicle_weight_kg_rail_frgt["rail_freight"].to_numpy(),      material_fractions_rail_frgt)
inland_ship_mat_in,  inland_ship_mat_out,  inland_ship_mat_stock    = nr_by_cohorts_to_materials_simple_np(inland_ship_in,  inland_ship_out_coh,  inland_ship_stock_coh,  vehicle_weight_kg_inland_ship["inland_shipping"].to_numpy(), material_fractions_inland_ship)

ship_small_mat_in,   ship_small_mat_out,   ship_small_mat_stock     = nr_by_cohorts_to_materials_simple_np(ship_small_in,   ship_small_out_coh,   ship_small_stock_coh,   weight_boats['Small'].to_numpy(),                            material_fractions_ship_small)
ship_medium_mat_in,  ship_medium_mat_out,  ship_medium_mat_stock    = nr_by_cohorts_to_materials_simple_np(ship_medium_in,  ship_medium_out_coh,  ship_medium_stock_coh,  weight_boats['Medium'].to_numpy(),                           material_fractions_ship_medium)
ship_large_mat_in,   ship_large_mat_out,   ship_large_mat_stock     = nr_by_cohorts_to_materials_simple_np(ship_large_in,   ship_large_out_coh,   ship_large_stock_coh,   weight_boats['Large'].to_numpy(),                            material_fractions_ship_large)
ship_vlarge_mat_in,  ship_vlarge_mat_out,  ship_vlarge_mat_stock    = nr_by_cohorts_to_materials_simple_np(ship_vlarge_in,  ship_vlarge_out_coh,  ship_vlarge_stock_coh,  weight_boats['Very Large'].to_numpy(),                       material_fractions_ship_vlarge)


# Calculate the weight of materials in the vehicles with sub-types: stock, inflow & outflow 
bus_regl_mat_in,     bus_regl_mat_out,     bus_regl_mat_stock       = nr_by_cohorts_to_materials_typical_np(bus_regl_in,   bus_regl_out_coh,   bus_regl_stock_coh,   vehicle_weight_kg_bus,   material_fractions_bus_reg)
bus_midi_mat_in,     bus_midi_mat_out,     bus_midi_mat_stock       = nr_by_cohorts_to_materials_typical_np(bus_midi_in,   bus_midi_out_coh,   bus_midi_stock_coh,   vehicle_weight_kg_midi,  material_fractions_bus_midi)
car_total_mat_in,    car_total_mat_out,    car_total_mat_stock      = nr_by_cohorts_to_materials_typical_np(car_in,        car_out_coh,        car_stock_coh,        vehicle_weight_kg_car,   material_fractions_car)    

trucks_HFT_mat_in,   trucks_HFT_mat_out,   trucks_HFT_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_HFT_in, trucks_HFT_out_coh, trucks_HFT_stock_coh, vehicle_weight_kg_HFT,   material_fractions_truck_HFT)
trucks_MFT_mat_in,   trucks_MFT_mat_out,   trucks_MFT_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_MFT_in, trucks_MFT_out_coh, trucks_MFT_stock_coh, vehicle_weight_kg_MFT,   material_fractions_truck_MFT)
trucks_LCV_mat_in,   trucks_LCV_mat_out,   trucks_LCV_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_LCV_in, trucks_LCV_out_coh, trucks_LCV_stock_coh, vehicle_weight_kg_LCV,   material_fractions_truck_LCV)


# Calculate the materials in batteries (in typical vehicles only) 
# For batteries this is a 2-step process, first (1) we pre-calculate the average material composition (of the batteries at inflow), based on a globally changing battery share & the changeing battery-specific material composition, both derived from our paper on (a.o.) storage in the electricity system.
# Then (2) we run the same function to derive the materials in vehicle batteries (based on changeing weight, composition & battery share)
# In doing so, we are no longer able to know the battery share per vehicle sub-type

battery_material_composition    = pd.DataFrame(index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=pd.MultiIndex.from_product([car_types + ['Trolley'], battery_materials_full.columns.levels[0]]))
battery_weight_total_in         = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=battery_weights_full.columns.levels[1])
battery_weight_total_stock      = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=battery_weights_full.columns.levels[1])

battery_weight_regional_stock   = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,REGIONS+1)))
battery_weight_regional_in      = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,REGIONS+1)))
battery_weight_regional_out     = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,REGIONS+1)))
 
# for now, 1 global battery market is assumed (so no difference between battery types for different vehicle sub-types), so the composition is duplicated over all vtypes
for vehicle in list(battery_weights_full.columns.levels[0]):
   for vtype in list(battery_weights_full.columns.levels[1]):
      for material in list(battery_materials_full.columns.levels[0]):
         battery_material_composition.loc[idx[vehicle,:],idx[vtype,material]] = battery_shares_full.mul(battery_materials_full.loc[:,idx[material,:]].droplevel(0,axis=1)).sum(axis=1).values

last_years = -(END_YEAR +1 - START_YEAR)
# Battery material calculations (these are made from 1971 onwwards, hence the selection in years) 
bus_regl_bat_in,     bus_regl_bat_out,     bus_regl_bat_stock       = nr_by_cohorts_to_materials_typical_np(bus_regl_in[:,:,last_years:],  bus_regl_out_coh[:,:,last_years:, last_years:], bus_regl_stock_coh[:,:,last_years:, last_years:], battery_weights_full['reg_bus'],   battery_material_composition.loc[idx['reg_bus',:],:].droplevel(0))
bus_midi_bat_in,     bus_midi_bat_out,     bus_midi_bat_stock       = nr_by_cohorts_to_materials_typical_np(bus_midi_in[:,:,last_years:],  bus_midi_out_coh[:,:,last_years:, last_years:], bus_midi_stock_coh[:,:,last_years:, last_years:], battery_weights_full['midi_bus'],  battery_material_composition.loc[idx['midi_bus',:],:].droplevel(0))
car_total_bat_in,    car_total_bat_out,    car_total_bat_stock      = nr_by_cohorts_to_materials_typical_np(car_in[:,:,last_years:],       car_out_coh[:,:,last_years:, last_years:],      car_stock_coh[:,:,last_years:, last_years:],      battery_weights_full['car'],       battery_material_composition.loc[idx['car',:],idx[car_types,:]].droplevel(0))    #mind that cars don't have Trolleys, hence the additional selection

trucks_HFT_bat_in,   trucks_HFT_bat_out,   trucks_HFT_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_HFT_in[:,:,last_years:], trucks_HFT_out_coh[:,:,last_years:, last_years:], trucks_HFT_stock_coh[:,:,last_years:, last_years:], battery_weights_full['HFT'],  battery_material_composition.loc[idx['HFT',:],:].droplevel(0))
trucks_MFT_bat_in,   trucks_MFT_bat_out,   trucks_MFT_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_MFT_in[:,:,last_years:], trucks_MFT_out_coh[:,:,last_years:, last_years:], trucks_MFT_stock_coh[:,:,last_years:, last_years:], battery_weights_full['MFT'],  battery_material_composition.loc[idx['MFT',:],:].droplevel(0))
trucks_LCV_bat_in,   trucks_LCV_bat_out,   trucks_LCV_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_LCV_in[:,:,last_years:], trucks_LCV_out_coh[:,:,last_years:, last_years:], trucks_LCV_stock_coh[:,:,last_years:, last_years:], battery_weights_full['LCV'],  battery_material_composition.loc[idx['LCV',:],:].droplevel(0))

# Sum the weight of the accounted materials (! so not total weight) in batteries by vehicle & vehicle type, output for figures
for vtype in list(battery_weights_full.columns.levels[1]):
      battery_weight_total_in.loc[idx['reg_bus',:],vtype]     = bus_regl_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_in.loc[idx['midi_bus',:],vtype]    = bus_midi_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_in.loc[idx['LCV',:],vtype]         = trucks_LCV_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_in.loc[idx['MFT',:],vtype]         = trucks_MFT_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_in.loc[idx['HFT',:],vtype]         = trucks_HFT_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      
      battery_weight_total_stock.loc[idx['reg_bus',:],vtype]  = bus_regl_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_stock.loc[idx['midi_bus',:],vtype] = bus_midi_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_stock.loc[idx['LCV',:],vtype]      = trucks_LCV_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_stock.loc[idx['MFT',:],vtype]      = trucks_MFT_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_stock.loc[idx['HFT',:],vtype]      = trucks_HFT_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values   
      
      if vtype == 'Trolley':
         pass 
      else:
         battery_weight_total_in.loc[idx['car',:],vtype]      = car_total_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
         battery_weight_total_stock.loc[idx['car',:],vtype]   = car_total_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values

battery_weight_total_in.to_csv(OUTPUT_FOLDER + '\\battery_weight_kg_in.csv', index=True)       # in kg
battery_weight_total_stock.to_csv(OUTPUT_FOLDER + '\\battery_weight_kg_stock.csv', index=True) # in kg

# Regional battery weight (only the accounted materials), used in graph later on
battery_weight_regional_stock = bus_regl_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + bus_midi_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_LCV_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_MFT_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_HFT_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + car_total_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1)
battery_weight_regional_in    = bus_regl_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + bus_midi_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_LCV_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_MFT_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_HFT_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + car_total_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)     
battery_weight_regional_out   = bus_regl_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + bus_midi_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_LCV_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_MFT_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_HFT_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + car_total_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)                                      

#%% ################################### Organise data for output ###########################################

year_select = list(range(START_YEAR, END_YEAR + 1))

# define 6 dataframes on materials in  stock, inflow & outflow X passenger vs. freight vehicles
index   = pd.MultiIndex.from_product([year_select, list(range(1,REGIONS+1)), ['vehicle','battery'], labels_materials], names=['year', 'region', 'part', 'materials'])
vehicle_materials_stock_passenger   = pd.DataFrame(index=index, columns=labels_pas)
vehicle_materials_stock_freight     = pd.DataFrame(index=index, columns=labels_fre)
vehicle_materials_inflow_passenger  = pd.DataFrame(index=index, columns=labels_pas)
vehicle_materials_inflow_freight    = pd.DataFrame(index=index, columns=labels_fre)
vehicle_materials_outflow_passenger = pd.DataFrame(index=index, columns=labels_pas)
vehicle_materials_outflow_freight   = pd.DataFrame(index=index, columns=labels_fre)

for material in labels_materials:

   ############## STARTING WITH SIMPLE VEHICLES ###########################
   
   # passenger stock, vehicles (in kg)
   vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'bicycle']       = bikes_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']      = rail_reg_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']      = rail_hst_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'air_pas']       = air_pas_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   # freight stock (in kg)
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']     = inland_ship_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'rail_freight']        = rail_freight_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'air_freight']         = air_freight_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small']  = ship_small_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']    = ship_medium_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large']  = ship_large_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']     = ship_vlarge_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   # passenger inflow (in kg)
   vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'bicycle']    = bikes_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']   = rail_reg_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']   = rail_hst_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'air_pas']    = air_pas_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   # freight inflow (in kg)
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']    = inland_ship_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'rail_freight']       = rail_freight_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'air_freight']        = air_freight_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small'] = ship_small_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']   = ship_medium_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large'] = ship_large_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']    = ship_vlarge_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   # passenger outflow (in kg)
   vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'bicycle']   = bikes_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']  = rail_reg_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']  = rail_hst_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'air_pas']   = air_pas_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   # freight outflow (in kg)
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']     = inland_ship_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'rail_freight']        = rail_freight_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'air_freight']         = air_freight_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small']  = ship_small_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']    = ship_medium_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large']  = ship_large_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']     = ship_vlarge_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   ############ CONTINUEING WITH TYPICAL VEHICLES (MATERIALS IN TRUCKS & BUSSES ARE SUMMED, FOR CARS DETAIL BY TYPE IS MAINTAINED) ##########################
   
   part = 'vehicle'
   
   # passenger stock, vehicles (in kg)
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'midi_bus']    = bus_midi_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'reg_bus']     = bus_regl_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'ICE']         = car_total_mat_stock.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'HEV']         = car_total_mat_stock.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values   
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'PHEV']        = car_total_mat_stock.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'BEV']         = car_total_mat_stock.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'FCV']         = car_total_mat_stock.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
   # freight stock (in kg)
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'LCV']           = trucks_LCV_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'MFT']           = trucks_MFT_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'HFT']           = trucks_HFT_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

   # passenger inflow (in kg)
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'midi_bus']   = bus_midi_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'reg_bus']    = bus_regl_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'ICE']        = car_total_mat_in.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'HEV']        = car_total_mat_in.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'PHEV']       = car_total_mat_in.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'BEV']        = car_total_mat_in.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'FCV']        = car_total_mat_in.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values        
 
   # freight inflow (in kg)
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'LCV']          = trucks_LCV_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'MFT']          = trucks_MFT_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'HFT']          = trucks_HFT_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values    

   # passenger outflow (in kg)
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'midi_bus']  = bus_midi_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'reg_bus']   = bus_regl_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'ICE']       = car_total_mat_out.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'HEV']       = car_total_mat_out.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'PHEV']      = car_total_mat_out.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'BEV']       = car_total_mat_out.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'FCV']       = car_total_mat_out.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
   # freight outflow (in kg)
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'LCV'] = trucks_LCV_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'MFT']= trucks_MFT_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'HFT'] = trucks_HFT_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

   ############ CONTINUEING WITH BATTERIES (MATERALS IN TRUCKS & BUSSES ARE SUMMED, FOR CARS DETAIL BY TYPE IS MAINTAINED) ##########################

   part = 'battery'
   
   # passenger stock, vehicles (in kg)
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'midi_bus']    = bus_midi_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'reg_bus']     = bus_regl_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'ICE']         = car_total_bat_stock.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'HEV']         = car_total_bat_stock.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values   
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'PHEV']        = car_total_bat_stock.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'BEV']         = car_total_bat_stock.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'FCV']         = car_total_bat_stock.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
   # freight stock (in kg)
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'LCV']           = trucks_LCV_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'MFT']           = trucks_MFT_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'HFT']           = trucks_HFT_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

   # passenger inflow (in kg)
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'midi_bus']   = bus_midi_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'reg_bus']    = bus_regl_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'ICE']        = car_total_bat_in.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'HEV']        = car_total_bat_in.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'PHEV']       = car_total_bat_in.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'BEV']        = car_total_bat_in.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'FCV']        = car_total_bat_in.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values        
 
   # freight inflow (in kg)
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'LCV']          = trucks_LCV_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'MFT']          = trucks_MFT_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'HFT']          = trucks_HFT_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values    

   # passenger outflow (in kg)
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'midi_bus']  = bus_midi_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'reg_bus']   = bus_regl_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'ICE']       = car_total_bat_out.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'HEV']       = car_total_bat_out.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'PHEV']      = car_total_bat_out.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'BEV']       = car_total_bat_out.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'FCV']       = car_total_bat_out.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
   # freight outflow (in kg)
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'LCV']         = trucks_LCV_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'MFT']         = trucks_MFT_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'HFT']         = trucks_HFT_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

