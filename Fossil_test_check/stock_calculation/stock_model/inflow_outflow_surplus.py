def inflow_outflow_surplus_2(stock, lifetime_mean, lifetime_stdv):
    """
    This function recreates a stock-driven dynamic stock model while allowing for surplus stock & natural decay of existing stock when required stock is temporary lowered
    Current implementation is only for a folded-normal districbution (thus input = mean lifetime & standard deviation as a fraction of the mean lifetime)
    IN:  Stock (index: time, columns: regions), mean lifetime, standard-deviation (i.r.t. the mean)
    OUT: Inflow (time*region), Stock per cohort (time*time*region), Outflow per cohort (time*time*region)
    """
    
    # define & assign key settings
    start         = stock.first_valid_index()[0] #get the second value from the first valid multi index (as tuple)
    end           = stock.last_valid_index()[0]
    return(start)   
    # lifetime_mean = [lifetime_mean for i in range(start,end+1)]    # input is a single value for mean lifetime, this makes a list of the same value over the entire time period (needs to be changed if changing lifetimes are applied)
    # lifetime_stdv = [lifetime_stdv for i in range(start,end+1)]    # input is a single value for the standard deviation (as a fraction of the mean lifetime), this makes a list of the same value over the entire time period (needs to be changed if changing lifetimes are applied
    # time          = range(start, end+1)
    
    # return (lifetime_mean)