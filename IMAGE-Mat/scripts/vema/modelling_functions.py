import pandas as pd
import numpy as np
from constants import FIRST_YEAR, END_YEAR, REGIONS
from read_scripts.dynamic_stock_model_BM import DynamicStockModel as DSM
idx = pd.IndexSlice


# Generic interpolation function
def interpolate(original: pd.DataFrame, first_year=[], change: str='no'):
    """
    Generic linear interpolation function that interpolates between given years in the original data. With 2 additional functionalities:
    1) Function allows to indicate the first year of operation based on optional first_year argument (i.e. when was the time-series initiated with 0 as a value)
    2) Optionally, it allows to derive a trend based on the originally specified data and apply that trend into the future model years
    
    Assumptions: 
    Assumes: that the time dimension is the only dimension over which the interpolation is applied & that time is the (single-level) index of the DataFrame
    Assumes: that IF a first_year is given, it is a pandas dataframe with the same multi-index columns as the original data, and only one index (values are the first year, for which values will be set to 0) 
    Assumes: that if first_year is not given, and change='no' THEN values are assumed to remain constant at first/last known levels

    Parameters
    ----------
    original : pd.DataFrame
        Original data that requires interpolation over the time dimension, which is assumed to be the (single-level) index
    first_year : if given pd.DataFrame, optional (if not given default is an empty list)
        data describing the first year of operation. If given, it will assume values were 0 in the first year of operation and do the interpolation afterwards. The default is []: pd.DataFrame.
    change : str, optional
        argument to fix future values at the last known value (change=no) or to derive a trend based on the original data and apply it as an indexed growth after the last known year. The default is 'no'.

    Returns
    -------
    reindexed_filled : pd.DataFrame
        A pandas dataframe fully filled over the entire model time (index). Columns are the same as the original input

    """
    # determine the first & last available year in the original data
    start = original.first_valid_index()
    end   = original.last_valid_index()
    
    # reindexing from model start year to end year (i.e. the entire model period, adding NaN values for non-existing years)
    reindexed = original[:].reindex(list(range(FIRST_YEAR,END_YEAR+1)))
    
    # IF there is a first_year defined, add these as 0
    # TODO: Test if columns of first_year & original are the same
    if not isinstance(first_year, list):
        for item in list(original.columns):
            reindexed.loc[first_year[item],item] = 0
    else:
        pass
    
    # This function allows to derive a trend based on the original data and apply it into the future
    # change='no'  means the future values remains constant at the last known year
    # change='yes' means the future values are based on the trend in the years that were known in the original input
    if change=='no':
        # interpolate both ways, meaning that linear interpolation is performed between years with data & is kept constant at the first/last known data point before/after
        reindexed_filled = reindexed.interpolate(kind='linear', limit_direction='both')
    else:
        # if change = 'yes'
        # interpolate, but not after the last known year
        reindexed_filled = reindexed.interpolate(kind='linear', limit_direction='backward')
        
        # first determine the growth rate based on the historic series (start-to-end)
        growthrate = original.pct_change().sum(axis=0)/(end-start)
        # then apply the average annual growth rate to the last historic year
        for row in range(end+1, END_YEAR + 1):
            reindexed_filled.loc[row] = reindexed_filled.loc[row-1].values * (1+growthrate)

    return reindexed_filled

def tkms_to_nr_of_vehicles_fixed(tkms, mileage, load, loadfactor, unit_conversion=None):
    """
    This function translates ton kilometers (by year & by region) to nr of vehicles (same dimms) 
    using fixed indicators on mileage, load capacity and load factor

    Parameters:
    - tkms: DataFrame containing ton kilometers.
    - mileage: DataFrame or Series containing the mileage of vehicles.
    - load: The load capacity of the vehicles.
    - loadfactor: The load factor of the vehicles.
    - unit: The unit of tkms ('T' for Tera km, 'k' for km).
    """
    if unit_conversion == 'T':
        tkms = tkms * 1e12  # Convert Tera km to km
    elif unit_conversion == 'M':
        tkms = tkms * 1e6  # Convert Mega km to km
    elif unit_conversion == 'k':
        tkms = tkms * 1e3  # Convert kilo km to km
    elif unit_conversion == None:
       pass
    else:
       raise Exception(f"This unit conversion input '{unit_conversion}' is not supported.") 
       
    # if unit is None, assume tkms is already in km

    # then get the vehicle kilometers required to fulfill the transport demand
    vkms = tkms/(load*loadfactor)
    # then get the number of vehicles by dividing by the mileage
    nr_of_vehicles = vkms.div(mileage, axis=0)
    return nr_of_vehicles

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
           DSMforward = DSM(t = np.arange(0,len(stock[:,region]),1), s=stock[:,region],
                            lt = {'Type': 'FoldedNormal',
                                  'Mean': np.array(fact1),
                                  'StdDev': np.array(fact2_list)}) # folded normal distribution
        else:
           DSMforward = DSM(t = np.arange(0,len(stock[:,region]),1), s=stock[:,region],
                            lt = {'Type': 'Weibull',
                                  'Shape': np.array(fact1),
                                  'Scale': np.array(fact2_list)}) # Weibull distribution
        # run the DSM, to give 3 outputs: stock_by_cohort, outflow_by_cohort & inflow_per_year
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model_surplus()

        # store the regional results in the return dataframe
        # (full range of years (e.g. 1900 onwards), for later use in material calculations
        # & no moving average applied here)
        # sum the outflow by cohort to get the total outflow per year
        outflow_cohort[region,:,:] = out_oc
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

   inflow         = np.zeros((len(stock_share.columns.levels[0]), len(stock.iloc[0]), len(stock)))
   outflow_cohort = np.zeros((len(stock_share.columns.levels[0]), len(stock.iloc[0]), len(stock), len(stock)))
   stock_cohort   = np.zeros((len(stock_share.columns.levels[0]), len(stock.iloc[0]), len(stock), len(stock)))

   stock_by_vtype = pd.DataFrame(0, index=stock_share.index, columns=stock_share.columns)
   vtype_list     = list(stock_by_vtype.columns.unique('type'))
   
   # calculate the stocks of individual (vehicle) types (in nr of vehicles)
   for vtype in vtype_list:
      stock_by_vtype.loc[:,idx[vtype,:]] = stock.mul(stock_share.loc[:,idx[vtype,:]])

   vtype_count = list(range(0,len(stock_share.columns.levels[0])))
   vtype_dict  = dict(zip(vtype_list, vtype_count))

   # Then run the original DSM for each vehicle type & add it to the inflow, ouflow & stock containers, (for stock: index = time; columns = vtype, region)
   for vtype in vtype_list:
      if stock_share.loc[:, idx[vtype,:]].sum().sum() > 0.0001:
         dsm_inflow, dsm_outflow_coh, dsm_stock_coh = inflow_outflow_dynamic_np(
                 stock_by_vtype.loc[:,idx[vtype,:]].to_numpy(), fact1, fact2, distribution)

         inflow[vtype_dict[vtype],:,:]           = dsm_inflow.T
         outflow_cohort[vtype_dict[vtype],:,:,:] = dsm_outflow_coh
         stock_cohort[vtype_dict[vtype],:,:,:]   = dsm_stock_coh

      else:
         pass

   return inflow, outflow_cohort, stock_cohort

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
            outflow_mat[material,region,:] = np.multiply(
                    np.multiply(outflow_cohort[region,:,:].T, weight), composition_used).T.sum(axis=1)
            stock_mat[material,region,:]   = np.multiply(
                    np.multiply(stock_cohort[region,:,:], weight), composition_used).sum(axis=1)
      else:
         pass
     
   length_materials = len(composition.columns)
   length_time      = END_YEAR + 1 - (END_YEAR + 1 - len(inflow))

   index          = pd.MultiIndex.from_product(
                        [composition.columns, range(END_YEAR + 1 - len(inflow), END_YEAR + 1)],
                        names=['time', 'type'])
   pd_inflow_mat  = pd.DataFrame(
           inflow_mat.transpose(0,2,1).reshape((length_materials * length_time), REGIONS),
           index=index, columns=range(1,len(inflow[0]) + 1))
   pd_outflow_mat = pd.DataFrame(
           outflow_mat.transpose(0,2,1).reshape((length_materials * length_time), REGIONS),
           index=index, columns=range(1,len(inflow[0]) + 1))
   pd_stock_mat   = pd.DataFrame(
           stock_mat.transpose(0,2,1).reshape((length_materials * length_time), REGIONS),
           index=index, columns=range(1,len(inflow[0]) + 1))
      
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
    
    #composition.columns.names = ['type', 'material']
    
    # get two dictionaries to keep tarck of the order of materials & vtypes while using numpy
    vtype_list   = list(composition.columns.unique('type'))  # columns.unique() keeps the original order, which is important here (same for materials)
    vtype_count  = list(range(0,len(vtype_list))) 
    vtype_dict   = dict(zip(vtype_count, vtype_list))
    
    mater_list   = list(composition.columns.unique('material'))
    mater_count  = list(range(0,len(composition.columns.levels[1]))) 
    mater_dict   = dict(zip(mater_count, mater_list))
        
    for vtype in range(0, len(vtype_list)): 
       # before running, check if the vehicle type is at all relevant in the vehicle (save calculation time)
       weight_used = weight.loc[:,vtype_dict[vtype]].values
       if stock_cohort[vtype].sum() > 0.001:  
          for material in range(0, len(mater_list)): 
             composition_used = composition.loc[:,idx[vtype_dict[vtype],mater_dict[material]]].values
             # before running, check if the material is at all relevant in the vehicle (save calculation time)
             if composition_used.sum() > 0.001:          
                for region in range(0, len(inflow[0])):
                   inflow_mat[vtype, material, region, :] = (inflow[vtype, region, :] * weight_used) * composition_used
                   outflow_mat[vtype, material, region, :] = np.multiply(
                           np.multiply(outflow_cohort[vtype, region, :, :].T, weight_used),
                           composition_used).T.sum(axis=1)
                   stock_mat[vtype, material, region, :] = np.multiply(np.multiply(
                       stock_cohort[vtype, region, :, :], weight_used), composition_used).sum(axis=1)
    
             else:
                pass
       else:
          pass
    
    length_materials = len(composition.columns.levels[1])
    length_time      = END_YEAR + 1 - (END_YEAR + 1 - len(inflow[0][0]))
    
    #return as pandas dataframe, just once
    index          = pd.MultiIndex.from_product(
                        [mater_list, list(range((END_YEAR + 1 - len(inflow[0][0])), END_YEAR + 1))],
                        names=['material', 'time'])
    columns        = pd.MultiIndex.from_product(
                        [vtype_list, range(1,REGIONS+1)], names=['type', 'region'])
    pd_inflow_mat  = pd.DataFrame(
            inflow_mat.transpose(1,3,0,2).reshape((length_materials * length_time), (len(inflow) * REGIONS)),
            index=index, columns=columns)
    pd_outflow_mat = pd.DataFrame(
            outflow_mat.transpose(1,3,0,2).reshape((length_materials * length_time), (len(inflow) * REGIONS)),
            index=index, columns=columns)
    pd_stock_mat   = pd.DataFrame(
            stock_mat.transpose(1,3,0,2).reshape((length_materials * length_time), (len(inflow) * REGIONS)),
            index=index, columns=columns)
          
    return pd_inflow_mat, pd_outflow_mat, pd_stock_mat

def preprocess_to_xarray(preprocessing_results, unit_mapping):
    preprocessing_results_xarray = {}
    
    for key, df in preprocessing_results.items():
        ds = df.to_xarray()
        if key in unit_mapping:
            ds = ds.assign_attrs({name: unit for name, unit in unit_mapping[key]['columns'].items()})
            ds = ds.pint.quantify()
        preprocessing_results_xarray[key] = ds
    
    return preprocessing_results_xarray