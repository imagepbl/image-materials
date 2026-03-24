import numpy as np
import pandas as pd
import scipy

from stock_model.dynamic_stock_model import DynamicStockModel as DSM

idx = pd.IndexSlice

#these functions use the stock-driven DSM to return inflow and outflow based on stock.

def inflow_outflow_dynamic_np(stock, fact1, fact2, distribution):
    """
    In this function the stock-driven DSM is applied to return inflow & outflow for all regions
    IN: stock as a dataframe (original unit by year & region), lifetime (in years), stdev_mult (multiplier to derive the standard deviation from the mean)
    OUT: 3 dataframes of stock, inflow & outflow (original unit by year^2 & region)
    
    This DYNAMIC variant of the DSM calculates & returns the full unadjusted (cohort specific) matrix (YES, very memory intensive, but neccesary for dynamic assesmment, i.e. changing vehicle weights & compositions)
    """
    inflow         = np.empty((len(stock[0]), len(stock)))
    outflow_cohort = np.empty((len(stock[0]), len(stock), len(stock))) 
    stock_cohort   = np.empty((len(stock[0]), len(stock), len(stock))) 

    # define mean (fact1) & standard deviation (fact2) (applied accross all regions) (For Weibull fact1 = shape & fact2 = scale)
    fact1_list = [fact1 for i in range(0,len(stock))]                      # list with the fixed (=mean) lifetime, given for every timestep (e.g. 1900-2100), needed for the DSM as it allows to change lifetime for different cohort (even though we keep it constant)
    if distribution == 'FoldedNormal':
       fact2_list = [fact1_list[i] * fact2 for i in range(0,len(stock))]   # for the (Folded) normal distribution, the standard_deviation is calculated as the mean times the multiplier (i.e. the standard deviation as a fraction of the mean)
    else:
       fact2_list = [fact2 for i in range(0,len(stock))]

    for region in range(0,len(stock[0])):
        # define and run the DSM   
        if distribution == 'FoldedNormal':
           DSMforward = DSM(t = np.arange(0,len(stock[:,region]),1), s=stock[:,region], lt = {'Type': 'FoldedNormal', 'Mean': np.array(fact1_list), 'StdDev': np.array(fact2_list)})  # definition of the DSM based on a folded normal distribution
        else:
           DSMforward = DSM(t = np.arange(0,len(stock[:,region]),1), s=stock[:,region], lt = {'Type': 'Weibull', 'Shape': np.array(fact1_list), 'Scale': np.array(fact2_list)})       # definition of the DSM based on a Weibull distribution
           
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)                                                                 # run the DSM, to give 3 outputs: stock_by_cohort, outflow_by_cohort & inflow_per_year
        
        # store the regional results in the return dataframe (full range of years (e.g. 1900 onwards), for later use in material calculations  & no moving average applied here)
        outflow_cohort[region,:,:] = out_oc                                                                                                      # sum the outflow by cohort to get the total outflow per year
        stock_cohort[region,:,:]   = out_sc
        inflow[region,:]           = out_i
         
    return inflow.T, outflow_cohort, stock_cohort     


def inflow_outflow_typical_np(stock, fact1, fact2, distribution):
    
    """
    In this function the dynamic inflow & outflow calculations are applied to stock with sub-types. To return inflow & outflow for all regions & stock types
    IN: stock as a dataframe (stock by year, region and type), lifetime (in years), stdev_mult (multiplier to derive the standard deviation from the mean)
    OUT: 3 dataframes of stock, inflow & outflow (stock by year^2 & region & type)
    
    This TYPICAL variant of the DSM calculates & returns the full unadjusted (cohort specific) matrix (time*time) for every region AND stock type (YES, very memory intensive, but neccesary for dynamic assesmment, i.e. changing vehicle weights & compositions)
    """
    
    stypes              = stock.index.unique(0)
    time                = stock.index.levels[1]
    inflow              = np.zeros((len(stypes), len(stock.iloc[0]), len(time)))
    outflow_cohort      = np.zeros((len(stypes), len(stock.iloc[0]), len(time), len(time)))
    stock_cohort        = np.zeros((len(stypes), len(stock.iloc[0]), len(time), len(time)))
    
    stype_list  = list(stypes)
    stype_count = list(range(0,len(stypes))) 
    stype_dict   = dict(zip(stype_count, stype_list))
    
    # Then run the original DSM for each stock type & add it to the inflow, ouflow & stock containers, (for stock: index = type, time; columns = region, time)
    
    
    for stype in range(0,len(stypes)): 
        
        if len(stypes) == 1: # if there is only 1 type (e.g. in processing), just use the original dataset in the dynamic model, no need to slice it up first
            dsm_inflow, dsm_outflow_coh, dsm_stock_coh = inflow_outflow_dynamic_np(stock.droplevel(0).to_numpy(), float(fact1), fact2, distribution)
      
            inflow[stype,:,:]           = dsm_inflow.T
            outflow_cohort[stype,:,:,:] = dsm_outflow_coh
            stock_cohort[stype,:,:,:]   = dsm_stock_coh
            
        elif stock.loc[stype_dict[stype]].sum().sum() > 0.001: #if there is more that one type, slice it by type. 
            dsm_inflow, dsm_outflow_coh, dsm_stock_coh = inflow_outflow_dynamic_np(stock.loc[idx[stype_dict[stype],:],:].droplevel(0).to_numpy(), float(fact1[stype]), fact2, distribution)
          
            inflow[stype,:,:]           = dsm_inflow.T
            outflow_cohort[stype,:,:,:] = dsm_outflow_coh
            stock_cohort[stype,:,:,:]   = dsm_stock_coh
       
        else:
            pass
         
    return inflow, outflow_cohort, stock_cohort

def inflow_outflow_surplus(stock: pd.DataFrame, lifetime_mean: float, lifetime_stdv: float):
    """
    This function recreates a stock-driven dynamic stock model while allowing for surplus stock & natural decay of existing stock when required stock is temporary lowered
    Current implementation is only for a folded-normal districbution (thus input = mean lifetime & standard deviation as a fraction of the mean lifetime)
    IN:  Stock (index: time, columns: regions), mean lifetime, standard-deviation (i.r.t. the mean)
    OUT: Inflow (time*region), Stock per cohort (time*time*region), Outflow per cohort (time*time*region)
    """
    
    # define & assign key settings
    start         = stock.first_valid_index()[0]
    end           = stock.last_valid_index()[0]
    
    lifetime_mean = [lifetime_mean for i in range(start,end+1)]    # input is a single value for mean lifetime, this makes a list of the same value over the entire time period (needs to be changed if changing lifetimes are applied)
    lifetime_stdv = [lifetime_stdv for i in range(start,end+1)]    # input is a single value for the standard deviation (as a fraction of the mean lifetime), this makes a list of the same value over the entire time period (needs to be changed if changing lifetimes are applied
    time          = range(start, end+1)
    
    # create empty np arrays to be filled
    inflow               = np.zeros((len(time), len(stock.iloc[0])))
    outflow_cohorts      = np.zeros((len(time), len(time), len(stock.iloc[0])))
    stock_cohorts        = np.zeros((len(time), len(time), len(stock.iloc[0])))

    # Full implementation of stock-driven DSM model in numpy
    for region in range(0,len(stock.columns)):
    
        # define a stock-driven model
        for year in range(0,end-start+1):
            
            # calculate the remaining stock as the sum of all cohorts in a year
            remaining_stock = stock_cohorts[:,year,region].sum()
            
            # total inflow required (= required stock - remaining stock);    
            inflow_year = np.maximum(0, stock.loc[year+start, region+1] - remaining_stock)   # max 0 avoids negative inflow, but allows for idle stock surplus in case the size of the required stock is declining more rapidly than it's natural decay
                          
            # then calculate the survival rate (based on lifetimes specific to the year of inflow)
            survival = scipy.stats.foldnorm.sf(np.arange(0,(end-start+1)-year), lifetime_mean[year]/(lifetime_mean[year]*lifetime_stdv[year]), 0, scale=(lifetime_mean[year]*lifetime_stdv[year]))           
            
            # then apply the survival to the inflow in current cohort, both the inflow & the survival are entered into the stock_cohort dataframe in 1 step
            stock_cohorts_list = inflow_year.values  *  survival
                
            stock_cohorts[year, list(range(year,end-start+1)) ,region] = stock_cohorts_list
            inflow[year, region] = inflow_year
        
    # separate the outflow (by cohort) calculation (separate shift calculation for each region & tech is MUCH more efficient than including it in additional loop over years)
    # calculate the outflow by cohort based on the stock by cohort that was just calculated
    for region in range(0,len(stock.columns)):
        stock_shifted = pd.DataFrame(stock_cohorts[:,:,region]).T.shift(1,axis=0).to_numpy()
        outflow_cohorts[:,:,region] = stock_shifted - stock_cohorts[:,:,region].T
    
    # Finally strip the outflow of negative values (shift method creates negative values in the inflow year, these are replaced by 0 here)
    outflow_cohorts[outflow_cohorts < 0] = 0
    
    return inflow, outflow_cohorts.transpose(2,0,1), stock_cohorts.transpose(2,1,0) #transposing makes sure that 
def inflow_outflow_surplus_typical_np (stock : pd.DataFrame, lifetime, lifetime_std):
    stypes              = stock.index.unique(0)
    time                = stock.index.levels[1]
    inflow              = np.zeros((len(stypes), len(stock.iloc[0]), len(time)))
    outflow_cohort      = np.zeros((len(stypes), len(stock.iloc[0]), len(time), len(time)))
    stock_cohort        = np.zeros((len(stypes), len(stock.iloc[0]), len(time), len(time)))
    stype_list  = list(stypes)
    stype_count = list(range(0,len(stypes))) 
    stype_dict  = dict(zip(stype_count, stype_list))
    for stype in range(0,len(stypes)): 
        if len(stypes) == 1:
            dsm_inflow, dsm_outflow_coh, dsm_stock_coh = inflow_outflow_surplus(stock.droplevel(0), float(lifetime), lifetime_std)
            inflow[stype,:,:]           = dsm_inflow.T
            outflow_cohort[stype,:,:,:] = dsm_outflow_coh
            stock_cohort[stype,:,:,:]   = dsm_stock_coh
        elif stock.loc[stype_dict[stype]].sum().sum() > 0.001:
            dsm_inflow, dsm_outflow_coh, dsm_stock_coh = inflow_outflow_surplus(stock.loc[idx[stype_dict[stype],:],:].droplevel(0), float(lifetime[stype]), lifetime_std)
            inflow[stype,:,:]           = dsm_inflow.T
            outflow_cohort[stype,:,:,:] = dsm_outflow_coh
            stock_cohort[stype,:,:,:]   = dsm_stock_coh
        else:
            pass
    return inflow, outflow_cohort, stock_cohort
def inflow_outflow_surplus_int(stock, lifetime_mean, lifetime_stdv): # SPECIFIC FOR INTERMEDIATE RESULTS
    """
    This function recreates a stock-driven dynamic stock model while allowing for surplus stock & natural decay of existing stock when required stock is temporary lowered
    Current implementation is only for a folded-normal districbution (thus input = mean lifetime & standard deviation as a fraction of the mean lifetime)
    IN:  Stock (index: time, columns: regions), mean lifetime, standard-deviation (i.r.t. the mean)
    OUT: Inflow (time*region), Stock per cohort (time*time*region), Outflow per cohort (time*time*region)
    """
    start         = stock.first_valid_index()
    end           = stock.last_valid_index()
    lifetime_list = [lifetime_mean for i in range(start,end+1)]    # input is a single value for mean lifetime, this makes a list of the same value over the entire time period (needs to be changed if changing lifetimes are applied)
    lifetime_stdv = [lifetime_stdv for i in range(start,end+1)]    # input is a single value for the standard deviation (as a fraction of the mean lifetime), this makes a list of the same value over the entire time period (needs to be changed if changing lifetimes are applied
    time          = range(start, end+1)
    inflow               = np.zeros((len(time), len(stock.iloc[0])))
    outflow_cohorts      = np.zeros((len(time), len(time), len(stock.iloc[0])))
    stock_cohorts        = np.zeros((len(time), len(time), len(stock.iloc[0])))
    for region in range(0,len(stock.columns)):
        for year in range(0,end-start+1):
            remaining_stock = stock_cohorts[:,year,region].sum()
            inflow_year = max(0, stock.loc[year+start, region+1] - remaining_stock)   # max 0 avoids negative inflow, but allows for idle stock surplus in case the size of the required stock is declining more rapidly than it's natural decay
            survival = scipy.stats.foldnorm.sf(np.arange(0,(end-start+1)-year), lifetime_list[year]/(lifetime_list[year]*lifetime_stdv[year]), 0, scale=(lifetime_list[year]*lifetime_stdv[year]))           
            stock_cohorts_list = inflow_year  *  survival
            stock_cohorts[year, list(range(year,end-start+1)) ,region] = stock_cohorts_list
            inflow[year, region] = inflow_year
    for region in range(0,len(stock.columns)):
        stock_shifted = pd.DataFrame(stock_cohorts[:,:,region]).T.shift(1,axis=0).to_numpy()
        outflow_cohorts[:,:,region] = stock_shifted - stock_cohorts[:,:,region].T
    outflow_cohorts[outflow_cohorts < 0] = 0
    return inflow, outflow_cohorts.transpose(2,0,1), stock_cohorts.transpose(2,1,0)