

#%% SQUARE METER Calculations (requires dynamic stock model to disaggregate building types) -----------------------------------------------------------

# share in new construction (from Fishman 2021)
housing_type_rur_new = housing_type_new2.loc[idx[:,:,'Rural'],:].droplevel(2)
housing_type_urb_new = housing_type_new2.loc[idx[:,:,'Urban'],:].droplevel(2)

# in the IRP 'baseline' (i.e. no Resource Efficiency assumptions) we maintain housing_type_rur_new to split stock
# in the Resource Efficiency scenario however, the indications of future type-split are based on new inflow (Fishman et al. 2021)
# in order to derive the share of building types in the stock we have to calculate total historic inflow first, 
# so lifetimes & the DSM are loaded here first

import dynamic_stock_model
from dynamic_stock_model import DynamicStockModel as DSM

if flag_Normal == 0:
    lifetimes_DB   = pd.read_csv('files_lifetimes\\' + scenario_select + '\\lifetimes.csv', index_col = [0,1,2,3])   # Weibull parameter database for residential buildings (shape & scale parameters given by region, area & building-type)
    lifetimes_comm = pd.read_csv('files_lifetimes\\' + scenario_select + '\\lifetimes_comm.csv', index_col = [0,1])  # Weibull parameter database for commercial buildings (shape & scale parameters given by region, area & building-type)
else:
    lifetimes_DB = pd.read_csv('files_lifetimes\lifetimes_normal.csv')  # Normal distribution database (Mean & StDev parameters given by region, area & building-type, though only defined by region for now)

# interpolate lifetime data
lifetimes_comm_shape = lifetimes_comm['Shape'].unstack().reindex(list(range(hist_year, end_year + 1))).interpolate(limit=300, limit_direction='both')
lifetimes_comm_scale = lifetimes_comm['Scale'].unstack().reindex(list(range(hist_year, end_year + 1))).interpolate(limit=300, limit_direction='both')

lifetimes_shape = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(hist_year, end_year + 1)), list(lifetimes_DB.index.levels[2]), list(lifetimes_DB.index.levels[3])]), columns=lifetimes_DB.index.levels[1])
lifetimes_scale = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(hist_year, end_year + 1)), list(lifetimes_DB.index.levels[2]), list(lifetimes_DB.index.levels[3])]), columns=lifetimes_DB.index.levels[1])
for building in list(lifetimes_DB.index.levels[2]):
    for area in list(lifetimes_DB.index.levels[3]):
        lifetimes_shape.loc[idx[:,building,area],:] = lifetimes_DB['Shape'].unstack(level=1).loc[:,building,area].reindex(list(range(hist_year, end_year + 1))).interpolate(method='linear', limit=300, limit_direction='both').values
        lifetimes_scale.loc[idx[:,building,area],:] = lifetimes_DB['Scale'].unstack(level=1).loc[:,building,area].reindex(list(range(hist_year, end_year + 1))).interpolate(method='linear', limit=300, limit_direction='both').values

# actual inflow calculations (Stock-driven DSM)
def inflow_outflow(shape, scale, stock):
    
   columns = pd.MultiIndex.from_product([list(range(1,27)), list(range(hist_year, stock.last_valid_index() + 1))], names=['regions', 'time'])
   out_oc_reg = pd.DataFrame(index=range(hist_year, stock.last_valid_index() + 1), columns=columns)
   out_sc_reg = pd.DataFrame(index=range(hist_year, stock.last_valid_index() + 1), columns=columns)
   out_i_reg  = pd.DataFrame(index=range(hist_year, stock.last_valid_index() + 1), columns=range(1,27))
    
    
   for region in range(1,regions+1):
      shape_list = shape[region]    
      scale_list = scale[region]
      
      if flag_Normal == 0:
         DSMforward = DSM(t = np.arange(0,len(stock),1), s=np.array(stock[region]), lt = {'Type': 'Weibull', 'Shape': np.array(shape_list), 'Scale': np.array(scale_list)})
      else:
         DSMforward = DSM(t = np.arange(0,len(stock),1), s=np.array(stock[region]), lt = {'Type': 'FoldedNormal', 'Mean': np.array(shape_list), 'StdDev': np.array(scale_list)}) # shape & scale list are actually Mean & StDev here
      
      out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)
      
      # list as dataframes
      out_oc[out_oc < 0] = 0  # in the rare occasion that negative outflow exists (as a consequence of negative inflow correct), purge values below zero
      out_oc_reg.loc[:,idx[region,:]] = pd.DataFrame(out_oc, index=list(range(hist_year, stock.last_valid_index() + 1)), columns=list(range(hist_year, stock.last_valid_index() + 1))).values
      out_sc_reg.loc[:,idx[region,:]] = pd.DataFrame(out_sc, index=list(range(hist_year, stock.last_valid_index() + 1)), columns=list(range(hist_year, stock.last_valid_index() + 1))).values
      out_i_reg[region]               = pd.Series(out_i, index=list(range(hist_year, stock.last_valid_index() + 1)))
      
   return out_oc_reg, out_i_reg, out_sc_reg

# stock calculations (inflow-driven DSM)
def inflow_driven(shape, scale, inflow):
    
   columns = pd.MultiIndex.from_product([list(range(1,27)), list(range(hist_year, end_year + 1))], names=['regions', 'time'])
   out_sc_reg = pd.DataFrame(index=range(hist_year, end_year + 1), columns=columns)
    
   for region in range(1,regions+1):
      shape_list = shape[region]    
      scale_list = scale[region]
      
      if flag_Normal == 0:
         DSMforward = DSM(t = np.arange(0,len(inflow),1), i=np.array(inflow[region]), lt = {'Type': 'Weibull', 'Shape': np.array(shape_list), 'Scale': np.array(scale_list)})
      else:
         DSMforward = DSM(t = np.arange(0,len(stock),1), i=np.array(inflow[region]), lt = {'Type': 'FoldedNormal', 'Mean': np.array(shape_list), 'StdDev': np.array(scale_list)}) # shape & scale list are actually Mean & StDev here    
      out_sc  = DSMforward.compute_s_c_inflow_driven()
      
      # list stock by cohort as dataframe
      out_sc_reg.loc[:,idx[region,:]] = pd.DataFrame(out_sc, index=list(range(hist_year, end_year + 1)), columns=list(range(hist_year, end_year + 1))).values
   
   # We're only interested in the total stock (for now), so we sum the stock by cohort each year
   out_s_reg = out_sc_reg.sum(axis=1, level=0)    
   
   return out_s_reg

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

# if the scenario variant is "Resource Efficient'
# first (1) calculate the inflow-type share over the historic period based on the known historic stock-share
# then  (2) extend the inflow-share with given values for 2050 and interpolate
# then  (3) re-calculate the stock using an inflow-driven approach and
# then  (4) derive the stock share and replace the original data
if scenario_variant == "CP_RE" or scenario_variant == "2D_RE":
    
    # (1) calculate the inflow-type share over the historic period based on the known historic stock-share
    m2_rur_det_oc, m2_rur_det_i, m2_rur_det_sc = inflow_outflow(lifetimes_shape.loc[idx[:,'Detached','Rural'],:].droplevel([1,2]),      lifetimes_scale.loc[idx[:,'Detached','Rural'],:].droplevel([1,2]),      m2_det_rur)
    m2_rur_sem_oc, m2_rur_sem_i, m2_rur_sem_sc = inflow_outflow(lifetimes_shape.loc[idx[:,'Semi-detached','Rural'],:].droplevel([1,2]), lifetimes_scale.loc[idx[:,'Semi-detached','Rural'],:].droplevel([1,2]), m2_sem_rur)
    m2_rur_app_oc, m2_rur_app_i, m2_rur_app_sc = inflow_outflow(lifetimes_shape.loc[idx[:,'Appartments','Rural'],:].droplevel([1,2]),   lifetimes_scale.loc[idx[:,'Appartments','Rural'],:].droplevel([1,2]),   m2_app_rur)
    m2_rur_hig_oc, m2_rur_hig_i, m2_rur_hig_sc = inflow_outflow(lifetimes_shape.loc[idx[:,'High-rise','Rural'],:].droplevel([1,2]),     lifetimes_scale.loc[idx[:,'High-rise','Rural'],:].droplevel([1,2]),     m2_hig_rur)
    
    m2_urb_det_oc, m2_urb_det_i, m2_urb_det_sc = inflow_outflow(lifetimes_shape.loc[idx[:,'Detached','Urban'],:].droplevel([1,2]),      lifetimes_scale.loc[idx[:,'Detached','Urban'],:].droplevel([1,2]),      m2_det_urb)
    m2_urb_sem_oc, m2_urb_sem_i, m2_urb_sem_sc = inflow_outflow(lifetimes_shape.loc[idx[:,'Semi-detached','Urban'],:].droplevel([1,2]), lifetimes_scale.loc[idx[:,'Semi-detached','Urban'],:].droplevel([1,2]), m2_sem_urb)
    m2_urb_app_oc, m2_urb_app_i, m2_urb_app_sc = inflow_outflow(lifetimes_shape.loc[idx[:,'Appartments','Urban'],:].droplevel([1,2]),   lifetimes_scale.loc[idx[:,'Appartments','Urban'],:].droplevel([1,2]),   m2_app_urb)
    m2_urb_hig_oc, m2_urb_hig_i, m2_urb_hig_sc = inflow_outflow(lifetimes_shape.loc[idx[:,'High-rise','Urban'],:].droplevel([1,2]),     lifetimes_scale.loc[idx[:,'High-rise','Urban'],:].droplevel([1,2]),     m2_hig_urb)
    
    total_inflow_rur = m2_rur_det_i + m2_rur_sem_i + m2_rur_app_i + m2_rur_hig_i
    total_inflow_urb = m2_urb_det_i + m2_urb_sem_i + m2_urb_app_i + m2_urb_hig_i
    
    inflow_share_rur_det = (m2_rur_det_i / total_inflow_rur).fillna(0).loc[:switch_year]
    inflow_share_rur_sem = (m2_rur_sem_i / total_inflow_rur).fillna(0).loc[:switch_year]
    inflow_share_rur_app = (m2_rur_app_i / total_inflow_rur).fillna(0).loc[:switch_year]
    inflow_share_rur_hig = (m2_rur_hig_i / total_inflow_rur).fillna(0).loc[:switch_year]
    
    inflow_share_urb_det = (m2_urb_det_i / total_inflow_urb).fillna(0).loc[:switch_year]
    inflow_share_urb_sem = (m2_urb_sem_i / total_inflow_urb).fillna(0).loc[:switch_year]
    inflow_share_urb_app = (m2_urb_app_i / total_inflow_urb).fillna(0).loc[:switch_year]
    inflow_share_urb_hig = (m2_urb_hig_i / total_inflow_urb).fillna(0).loc[:switch_year]
    
    # (2) extend the inflow-share with given values for 2050 and interpolate
    inflow_share_rur_det.loc[2050] = housing_type_rur_new['Detached'].unstack().loc[2050]
    inflow_share_rur_sem.loc[2050] = housing_type_rur_new['Semi-detached'].unstack().loc[2050]
    inflow_share_rur_app.loc[2050] = housing_type_rur_new['Appartment'].unstack().loc[2050]
    inflow_share_rur_hig.loc[2050] = housing_type_rur_new['High-rise'].unstack().loc[2050]
    
    inflow_share_urb_det.loc[2050] = housing_type_urb_new['Detached'].unstack().loc[2050]
    inflow_share_urb_sem.loc[2050] = housing_type_urb_new['Semi-detached'].unstack().loc[2050]
    inflow_share_urb_app.loc[2050] = housing_type_urb_new['Appartment'].unstack().loc[2050]
    inflow_share_urb_hig.loc[2050] = housing_type_urb_new['High-rise'].unstack().loc[2050]
    
    inflow_share_rur_det = inflow_share_rur_det.reindex(list(range(hist_year, end_year + 1))).interpolate(limit_direction='both')
    inflow_share_rur_sem = inflow_share_rur_sem.reindex(list(range(hist_year, end_year + 1))).interpolate(limit_direction='both')
    inflow_share_rur_app = inflow_share_rur_app.reindex(list(range(hist_year, end_year + 1))).interpolate(limit_direction='both')
    inflow_share_rur_hig = inflow_share_rur_hig.reindex(list(range(hist_year, end_year + 1))).interpolate(limit_direction='both')
    
    inflow_share_urb_det = inflow_share_urb_det.reindex(list(range(hist_year, end_year + 1))).interpolate(limit_direction='both')
    inflow_share_urb_sem = inflow_share_urb_sem.reindex(list(range(hist_year, end_year + 1))).interpolate(limit_direction='both')
    inflow_share_urb_app = inflow_share_urb_app.reindex(list(range(hist_year, end_year + 1))).interpolate(limit_direction='both')
    inflow_share_urb_hig = inflow_share_urb_hig.reindex(list(range(hist_year, end_year + 1))).interpolate(limit_direction='both')
    
    # (3) re-calculate the stock using an inflow-driven approach and 
    stock_rur_det = inflow_driven(lifetimes_shape.loc[idx[:,'Detached','Rural'],:].droplevel([1,2]),      lifetimes_scale.loc[idx[:,'Detached','Rural'],:].droplevel([1,2]),      total_inflow_rur * inflow_share_rur_det)
    stock_rur_sem = inflow_driven(lifetimes_shape.loc[idx[:,'Semi-detached','Rural'],:].droplevel([1,2]), lifetimes_scale.loc[idx[:,'Semi-detached','Rural'],:].droplevel([1,2]), total_inflow_rur * inflow_share_rur_sem)
    stock_rur_app = inflow_driven(lifetimes_shape.loc[idx[:,'Appartments','Rural'],:].droplevel([1,2]),   lifetimes_scale.loc[idx[:,'Appartments','Rural'],:].droplevel([1,2]),   total_inflow_rur * inflow_share_rur_app)
    stock_rur_hig = inflow_driven(lifetimes_shape.loc[idx[:,'High-rise','Rural'],:].droplevel([1,2]),     lifetimes_scale.loc[idx[:,'High-rise','Rural'],:].droplevel([1,2]),     total_inflow_rur * inflow_share_rur_hig)
    stock_rur_tot = stock_rur_det + stock_rur_sem + stock_rur_app + stock_rur_hig
    
    stock_urb_det = inflow_driven(lifetimes_shape.loc[idx[:,'Detached','Urban'],:].droplevel([1,2]),      lifetimes_scale.loc[idx[:,'Detached','Urban'],:].droplevel([1,2]),      total_inflow_urb * inflow_share_urb_det)
    stock_urb_sem = inflow_driven(lifetimes_shape.loc[idx[:,'Semi-detached','Urban'],:].droplevel([1,2]), lifetimes_scale.loc[idx[:,'Semi-detached','Urban'],:].droplevel([1,2]), total_inflow_urb * inflow_share_urb_sem)
    stock_urb_app = inflow_driven(lifetimes_shape.loc[idx[:,'Appartments','Urban'],:].droplevel([1,2]),   lifetimes_scale.loc[idx[:,'Appartments','Urban'],:].droplevel([1,2]),   total_inflow_urb * inflow_share_urb_app)
    stock_urb_hig = inflow_driven(lifetimes_shape.loc[idx[:,'High-rise','Urban'],:].droplevel([1,2]),     lifetimes_scale.loc[idx[:,'High-rise','Urban'],:].droplevel([1,2]),     total_inflow_urb * inflow_share_urb_hig)
    stock_urb_tot = stock_urb_det + stock_urb_sem + stock_urb_app + stock_urb_hig
    
    # (4) derive the stock share and replace the original data
    stock_share_rur_det = stock_rur_det / stock_rur_tot
    stock_share_rur_sem = stock_rur_sem / stock_rur_tot
    stock_share_rur_app = stock_rur_app / stock_rur_tot
    stock_share_rur_hig = stock_rur_hig / stock_rur_tot
    
    stock_share_urb_det = stock_urb_det / stock_urb_tot
    stock_share_urb_sem = stock_urb_sem / stock_urb_tot
    stock_share_urb_app = stock_urb_app / stock_urb_tot
    stock_share_urb_hig = stock_urb_hig / stock_urb_tot
    
    # checksum (should be all 1's)
    checksum_rur = stock_share_rur_det + stock_share_rur_sem + stock_share_rur_app + stock_share_rur_hig
    checksum_urb = stock_share_urb_det + stock_share_urb_sem + stock_share_urb_app + stock_share_urb_hig
    
    # replace 2020-2060 floorspace stock values by the newly calculated stock stock based on new shares (derived from changing annual inflow shares) 
    m2_det_rur.loc[switch_year+1:] = pd.DataFrame(stock_share_rur_det.loc[switch_year+1:].values * m2_rur.loc[switch_year+1:].values, columns=people_rur.columns, index=people_rur.loc[switch_year+1:].index)
    m2_sem_rur.loc[switch_year+1:] = pd.DataFrame(stock_share_rur_sem.loc[switch_year+1:].values * m2_rur.loc[switch_year+1:].values, columns=people_rur.columns, index=people_rur.loc[switch_year+1:].index)
    m2_app_rur.loc[switch_year+1:] = pd.DataFrame(stock_share_rur_app.loc[switch_year+1:].values * m2_rur.loc[switch_year+1:].values, columns=people_rur.columns, index=people_rur.loc[switch_year+1:].index)
    m2_hig_rur.loc[switch_year+1:] = pd.DataFrame(stock_share_rur_hig.loc[switch_year+1:].values * m2_rur.loc[switch_year+1:].values, columns=people_rur.columns, index=people_rur.loc[switch_year+1:].index)
    
    m2_det_urb.loc[switch_year+1:] = pd.DataFrame(stock_share_urb_det.loc[switch_year+1:].values * m2_urb.loc[switch_year+1:].values, columns=people_urb.columns, index=people_urb.loc[switch_year+1:].index)
    m2_sem_urb.loc[switch_year+1:] = pd.DataFrame(stock_share_urb_sem.loc[switch_year+1:].values * m2_urb.loc[switch_year+1:].values, columns=people_urb.columns, index=people_urb.loc[switch_year+1:].index)
    m2_app_urb.loc[switch_year+1:] = pd.DataFrame(stock_share_urb_app.loc[switch_year+1:].values * m2_urb.loc[switch_year+1:].values, columns=people_urb.columns, index=people_urb.loc[switch_year+1:].index)
    m2_hig_urb.loc[switch_year+1:] = pd.DataFrame(stock_share_urb_hig.loc[switch_year+1:].values * m2_urb.loc[switch_year+1:].values, columns=people_urb.columns, index=people_urb.loc[switch_year+1:].index)
    
# else, simply apply the original shares as stock-shares (already done above, so pass)
else:
    pass

# total RESIDENTIAL square meters by region
m2 = m2_det_rur + m2_sem_rur + m2_app_rur + m2_hig_rur + m2_det_urb + m2_sem_urb + m2_app_urb + m2_hig_urb

# Total m2 for COMMERCIAL Buildings
commercial_m2_office = pd.DataFrame(commercial_m2_cap_office_tail.values * pop_tail.values, columns=people_rur.columns, index=people_rur.index)
commercial_m2_retail = pd.DataFrame(commercial_m2_cap_retail_tail.values * pop_tail.values, columns=people_rur.columns, index=people_rur.index)
commercial_m2_hotels = pd.DataFrame(commercial_m2_cap_hotels_tail.values * pop_tail.values, columns=people_rur.columns, index=people_rur.index)
commercial_m2_govern = pd.DataFrame(commercial_m2_cap_govern_tail.values * pop_tail.values, columns=people_rur.columns, index=people_rur.index)

#%% Add the share of the people per housing type (in the stock) as an output indicator
# To do so we need to work our way back from the share of in the floorspace stock, because different building types use different per capita floorspace
# we use the same relative share of the people per building type to disaggregate the actual IMAGE population data

people_det_rur_relative = m2_det_rur.div(avg_m2_cap_rur2.loc['1',:])
people_sem_rur_relative = m2_sem_rur.div(avg_m2_cap_rur2.loc['2',:])
people_app_rur_relative = m2_app_rur.div(avg_m2_cap_rur2.loc['3',:])
people_hig_rur_relative = m2_hig_rur.div(avg_m2_cap_rur2.loc['4',:])
people_rur_relative_tot = people_det_rur_relative + people_sem_rur_relative + people_app_rur_relative + people_hig_rur_relative

people_det_urb_relative = m2_det_urb.div(avg_m2_cap_urb2.loc['1',:])
people_sem_urb_relative = m2_sem_urb.div(avg_m2_cap_urb2.loc['2',:])
people_app_urb_relative = m2_app_urb.div(avg_m2_cap_urb2.loc['3',:])
people_hig_urb_relative = m2_hig_urb.div(avg_m2_cap_urb2.loc['4',:])
people_urb_relative_tot = people_det_urb_relative + people_sem_urb_relative + people_app_urb_relative + people_hig_urb_relative

# calculate the total number of people (urban/rural) BY HOUSING TYPE (the sum of det,sem,app & hig equals the total population e.g. people_rur) in millions of people
people_det_rur = pd.DataFrame(people_det_rur_relative.div(people_rur_relative_tot).values * people_rur.values, columns=people_rur.columns, index=people_rur.index)
people_sem_rur = pd.DataFrame(people_sem_rur_relative.div(people_rur_relative_tot).values * people_rur.values, columns=people_rur.columns, index=people_rur.index)
people_app_rur = pd.DataFrame(people_app_rur_relative.div(people_rur_relative_tot).values * people_rur.values, columns=people_rur.columns, index=people_rur.index)
people_hig_rur = pd.DataFrame(people_hig_rur_relative.div(people_rur_relative_tot).values * people_rur.values, columns=people_rur.columns, index=people_rur.index)

people_det_urb = pd.DataFrame(people_det_urb_relative.div(people_urb_relative_tot).values * people_urb.values, columns=people_urb.columns, index=people_urb.index)
people_sem_urb = pd.DataFrame(people_sem_urb_relative.div(people_urb_relative_tot).values * people_urb.values, columns=people_urb.columns, index=people_urb.index)
people_app_urb = pd.DataFrame(people_app_urb_relative.div(people_urb_relative_tot).values * people_urb.values, columns=people_urb.columns, index=people_urb.index)
people_hig_urb = pd.DataFrame(people_hig_urb_relative.div(people_urb_relative_tot).values * people_urb.values, columns=people_urb.columns, index=people_urb.index)

people_det = people_det_rur + people_det_urb
people_sem = people_sem_rur + people_sem_urb
people_app = people_app_rur + people_app_urb
people_hig = people_hig_rur + people_hig_urb

# intermediate output of population per building type for IRP scenarios
year_select = list(range(start_year, end_year+1)) 
people_out = pd.concat([people_det.loc[year_select], people_sem.loc[year_select], people_app.loc[year_select], people_hig.loc[year_select]],
              keys=['det', 'sem', 'app', 'hig']).stack().unstack(level=1)
people_out.to_csv('output\\' + scenario_select + '\\people_per_building_type.csv') # in millions


#%% MATERIAL CALCULATIONS

#First: interpolate the dynamic material intensity data
materials_commercial_dynamic = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(hist_year, end_year + 1)), list(materials_commercial.index.levels[1])]), columns=materials_commercial.columns)
building_materials_dynamic   = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(hist_year, end_year + 1)), list(range(1,27)), list(range(1,5))]), columns=building_materials.columns)

# interpolate material intensity data from files (residential buildings)
for material in building_materials.columns:
   for building in list(building_materials.index.levels[2]):
      selection = building_materials.loc[idx[:,:,building],material].droplevel(2, axis=0).unstack()
      selection.loc[hist_year,:] = selection.loc[selection.first_valid_index(),:]
      selection.loc[end_year + 1,:] = selection.loc[selection.last_valid_index(),:]
      selection = selection.reindex(list(range(hist_year, end_year + 1))).interpolate()
      building_materials_dynamic.loc[idx[:,:,building], material] = selection.stack()

# interpolate material intensity data from files (commercial buildings)
for building in materials_commercial.columns:
   selection = materials_commercial.loc[idx[:,:], building].unstack()
   selection.loc[hist_year,:] = selection.loc[selection.first_valid_index(),:]
   selection.loc[end_year + 1,:] = selection.loc[selection.last_valid_index(),:]
   selection = selection.reindex(list(range(hist_year, end_year + 1))).interpolate()
   materials_commercial_dynamic.loc[idx[:,:], building] = selection.stack()   

# restructuring for the residential materials (kg/m2)
material_steel     = building_materials_dynamic.loc[idx[:,:,:],'Steel'].unstack(level=1)
material_cement    = pd.DataFrame(0, index=material_steel.index, columns=material_steel.columns)
material_concrete  = building_materials_dynamic.loc[idx[:,:,:],'Concrete'].unstack(level=1)
material_wood      = building_materials_dynamic.loc[idx[:,:,:],'Wood'].unstack(level=1)
material_copper    = building_materials_dynamic.loc[idx[:,:,:],'Copper'].unstack(level=1)
material_aluminium = building_materials_dynamic.loc[idx[:,:,:],'Aluminium'].unstack(level=1)
material_glass     = building_materials_dynamic.loc[idx[:,:,:],'Glass'].unstack(level=1)
material_brick     = building_materials_dynamic.loc[idx[:,:,:],'Brick'].unstack(level=1)

# restructuring for the commercial materials (kg/m2)
columns = pd.MultiIndex.from_product([list(range(1,27)), ['Offices','Retail+','Hotels+','Govt+'] ])
material_com_steel     = pd.concat([materials_commercial_dynamic.loc[idx[:,'Steel'],:].droplevel(1)] * 26, axis=1).set_axis(columns, axis=1, inplace=False)
material_com_cement    = pd.concat([materials_commercial_dynamic.loc[idx[:,'Cement'],:].droplevel(1)] * 26, axis=1).set_axis(columns, axis=1, inplace=False)
material_com_concrete  = pd.concat([materials_commercial_dynamic.loc[idx[:,'Concrete'],:].droplevel(1)] * 26, axis=1).set_axis(columns, axis=1, inplace=False)
material_com_wood      = pd.concat([materials_commercial_dynamic.loc[idx[:,'Wood'],:].droplevel(1)] * 26, axis=1).set_axis(columns, axis=1, inplace=False)
material_com_copper    = pd.concat([materials_commercial_dynamic.loc[idx[:,'Copper'],:].droplevel(1)] * 26, axis=1).set_axis(columns, axis=1, inplace=False)
material_com_aluminium = pd.concat([materials_commercial_dynamic.loc[idx[:,'Aluminium'],:].droplevel(1)] * 26, axis=1).set_axis(columns, axis=1, inplace=False)
material_com_glass     = pd.concat([materials_commercial_dynamic.loc[idx[:,'Glass'],:].droplevel(1)] * 26, axis=1).set_axis(columns, axis=1, inplace=False)
material_com_brick     = pd.concat([materials_commercial_dynamic.loc[idx[:,'Brick'],:].droplevel(1)] * 26, axis=1).set_axis(columns, axis=1, inplace=False)


#%% INFLOW & OUTFLOW


# call the actual stock model to derive inflow & outflow based on stock & lifetime
m2_det_rur_o, m2_det_rur_i, m2_det_rur_s = inflow_outflow(lifetimes_shape.loc[:,'Detached','Rural'],      lifetimes_scale.loc[:,'Detached','Rural'],      m2_det_rur)
m2_sem_rur_o, m2_sem_rur_i, m2_sem_rur_s = inflow_outflow(lifetimes_shape.loc[:,'Semi-detached','Rural'], lifetimes_scale.loc[:,'Semi-detached','Rural'], m2_sem_rur)
m2_app_rur_o, m2_app_rur_i, m2_app_rur_s = inflow_outflow(lifetimes_shape.loc[:,'Appartments','Rural'],   lifetimes_scale.loc[:,'Appartments','Rural'],   m2_app_rur)
m2_hig_rur_o, m2_hig_rur_i, m2_hig_rur_s = inflow_outflow(lifetimes_shape.loc[:,'High-rise','Rural'],     lifetimes_scale.loc[:,'High-rise','Rural'],     m2_hig_rur)

m2_det_urb_o, m2_det_urb_i, m2_det_urb_s = inflow_outflow(lifetimes_shape.loc[:,'Detached','Urban'],      lifetimes_scale.loc[:,'Detached','Urban'],      m2_det_urb)
m2_sem_urb_o, m2_sem_urb_i, m2_sem_urb_s = inflow_outflow(lifetimes_shape.loc[:,'Semi-detached','Urban'], lifetimes_scale.loc[:,'Semi-detached','Urban'], m2_sem_urb)
m2_app_urb_o, m2_app_urb_i, m2_app_urb_s = inflow_outflow(lifetimes_shape.loc[:,'Appartments','Urban'],   lifetimes_scale.loc[:,'Appartments','Urban'],   m2_app_urb)
m2_hig_urb_o, m2_hig_urb_i, m2_hig_urb_s = inflow_outflow(lifetimes_shape.loc[:,'High-rise','Urban'],     lifetimes_scale.loc[:,'High-rise','Urban'],     m2_hig_urb)

m2_office_o, m2_office_i, m2_office_s    = inflow_outflow(lifetimes_comm_shape, lifetimes_comm_scale, commercial_m2_office)
m2_retail_o, m2_retail_i, m2_retail_s    = inflow_outflow(lifetimes_comm_shape, lifetimes_comm_scale, commercial_m2_retail)
m2_hotels_o, m2_hotels_i, m2_hotels_s    = inflow_outflow(lifetimes_comm_shape, lifetimes_comm_scale, commercial_m2_hotels)
m2_govern_o, m2_govern_i, m2_govern_s    = inflow_outflow(lifetimes_comm_shape, lifetimes_comm_scale, commercial_m2_govern)

# total MILLIONS of square meters inflow & outflow
m2_res_o  = m2_det_rur_o.sum(axis=1, level=0) + m2_sem_rur_o.sum(axis=1, level=0)  + m2_app_rur_o.sum(axis=1, level=0)  + m2_hig_rur_o.sum(axis=1, level=0)  + m2_det_urb_o.sum(axis=1, level=0)  + m2_sem_urb_o.sum(axis=1, level=0)  + m2_app_urb_o.sum(axis=1, level=0)  + m2_hig_urb_o.sum(axis=1, level=0) 
m2_res_i  = m2_det_rur_i + m2_sem_rur_i + m2_app_rur_i + m2_hig_rur_i + m2_det_urb_i + m2_sem_urb_i + m2_app_urb_i + m2_hig_urb_i
m2_comm_o = m2_office_o.sum(axis=1, level=0)  + m2_retail_o.sum(axis=1, level=0)  + m2_hotels_o.sum(axis=1, level=0)  + m2_govern_o.sum(axis=1, level=0) 
m2_comm_i = m2_office_i + m2_retail_i + m2_hotels_i + m2_govern_i

#####################################################################################################################################
#%% Material stocks

# RURAL material stock (Millions of kgs = *1000 tons)
kg_det_rur_steel     = m2_det_rur_s.mul(material_steel.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_rur_cement    = m2_det_rur_s.mul(material_cement.loc[idx[:,1],:].droplevel(1).unstack(),    axis=1).sum(axis=1, level=0)
kg_det_rur_concrete  = m2_det_rur_s.mul(material_concrete.loc[idx[:,1],:].droplevel(1).unstack(),  axis=1).sum(axis=1, level=0)
kg_det_rur_wood      = m2_det_rur_s.mul(material_wood.loc[idx[:,1],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_det_rur_copper    = m2_det_rur_s.mul(material_copper.loc[idx[:,1],:].droplevel(1).unstack(),    axis=1).sum(axis=1, level=0)
kg_det_rur_aluminium = m2_det_rur_s.mul(material_aluminium.loc[idx[:,1],:].droplevel(1).unstack(), axis=1).sum(axis=1, level=0)
kg_det_rur_glass     = m2_det_rur_s.mul(material_glass.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_rur_brick     = m2_det_rur_s.mul(material_brick.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)

kg_sem_rur_steel     = m2_sem_rur_s.mul(material_steel.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_rur_cement    = m2_sem_rur_s.mul(material_cement.loc[idx[:,2],:].droplevel(1).unstack(),    axis=1).sum(axis=1, level=0)
kg_sem_rur_concrete  = m2_sem_rur_s.mul(material_concrete.loc[idx[:,2],:].droplevel(1).unstack(),  axis=1).sum(axis=1, level=0) 
kg_sem_rur_wood      = m2_sem_rur_s.mul(material_wood.loc[idx[:,2],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_sem_rur_copper    = m2_sem_rur_s.mul(material_copper.loc[idx[:,2],:].droplevel(1).unstack(),    axis=1).sum(axis=1, level=0)
kg_sem_rur_aluminium = m2_sem_rur_s.mul(material_aluminium.loc[idx[:,2],:].droplevel(1).unstack(), axis=1).sum(axis=1, level=0)    
kg_sem_rur_glass     = m2_sem_rur_s.mul(material_glass.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)   
kg_sem_rur_brick     = m2_sem_rur_s.mul(material_brick.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)  

kg_app_rur_steel     = m2_app_rur_s.mul(material_steel.loc[idx[:,3],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)
kg_app_rur_cement    = m2_app_rur_s.mul(material_cement.loc[idx[:,3],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_app_rur_concrete  = m2_app_rur_s.mul(material_concrete.loc[idx[:,3],:].droplevel(1).unstack(),    axis=1).sum(axis=1, level=0)
kg_app_rur_wood      = m2_app_rur_s.mul(material_wood.loc[idx[:,3],:].droplevel(1).unstack(),        axis=1).sum(axis=1, level=0)
kg_app_rur_copper    = m2_app_rur_s.mul(material_copper.loc[idx[:,3],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_app_rur_aluminium = m2_app_rur_s.mul(material_aluminium.loc[idx[:,3],:].droplevel(1).unstack(),   axis=1).sum(axis=1, level=0)   
kg_app_rur_glass     = m2_app_rur_s.mul(material_glass.loc[idx[:,3],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)       
kg_app_rur_brick     = m2_app_rur_s.mul(material_brick.loc[idx[:,3],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)       

kg_hig_rur_steel     = m2_hig_rur_s.mul(material_steel.loc[idx[:,4],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)
kg_hig_rur_cement    = m2_hig_rur_s.mul(material_cement.loc[idx[:,4],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_hig_rur_concrete  = m2_hig_rur_s.mul(material_concrete.loc[idx[:,4],:].droplevel(1).unstack(),    axis=1).sum(axis=1, level=0)
kg_hig_rur_wood      = m2_hig_rur_s.mul(material_wood.loc[idx[:,4],:].droplevel(1).unstack(),        axis=1).sum(axis=1, level=0)
kg_hig_rur_copper    = m2_hig_rur_s.mul(material_copper.loc[idx[:,4],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_hig_rur_aluminium = m2_hig_rur_s.mul(material_aluminium.loc[idx[:,4],:].droplevel(1).unstack(),   axis=1).sum(axis=1, level=0)   
kg_hig_rur_glass     = m2_hig_rur_s.mul(material_glass.loc[idx[:,4],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)       
kg_hig_rur_brick     = m2_hig_rur_s.mul(material_brick.loc[idx[:,4],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)       

# URBAN material stock (millions of kgs)
kg_det_urb_steel     = m2_det_urb_s.mul(material_steel.loc[idx[:,1],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)
kg_det_urb_cement    = m2_det_urb_s.mul(material_cement.loc[idx[:,1],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_det_urb_concrete  = m2_det_urb_s.mul(material_concrete.loc[idx[:,1],:].droplevel(1).unstack(),    axis=1).sum(axis=1, level=0)
kg_det_urb_wood      = m2_det_urb_s.mul(material_wood.loc[idx[:,1],:].droplevel(1).unstack(),        axis=1).sum(axis=1, level=0)
kg_det_urb_copper    = m2_det_urb_s.mul(material_copper.loc[idx[:,1],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_det_urb_aluminium = m2_det_urb_s.mul(material_aluminium.loc[idx[:,1],:].droplevel(1).unstack(),   axis=1).sum(axis=1, level=0)
kg_det_urb_glass     = m2_det_urb_s.mul(material_glass.loc[idx[:,1],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)
kg_det_urb_brick     = m2_det_urb_s.mul(material_brick.loc[idx[:,1],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)

kg_sem_urb_steel     = m2_sem_urb_s.mul(material_steel.loc[idx[:,2],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)
kg_sem_urb_cement    = m2_sem_urb_s.mul(material_cement.loc[idx[:,2],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_sem_urb_concrete  = m2_sem_urb_s.mul(material_concrete.loc[idx[:,2],:].droplevel(1).unstack(),    axis=1).sum(axis=1, level=0)
kg_sem_urb_wood      = m2_sem_urb_s.mul(material_wood.loc[idx[:,2],:].droplevel(1).unstack(),        axis=1).sum(axis=1, level=0)
kg_sem_urb_copper    = m2_sem_urb_s.mul(material_copper.loc[idx[:,2],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_sem_urb_aluminium = m2_sem_urb_s.mul(material_aluminium.loc[idx[:,2],:].droplevel(1).unstack(),   axis=1).sum(axis=1, level=0) 
kg_sem_urb_glass     = m2_sem_urb_s.mul(material_glass.loc[idx[:,2],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0) 
kg_sem_urb_brick     = m2_sem_urb_s.mul(material_brick.loc[idx[:,2],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)   

kg_app_urb_steel     = m2_app_urb_s.mul(material_steel.loc[idx[:,3],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)
kg_app_urb_cement    = m2_app_urb_s.mul(material_cement.loc[idx[:,3],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_app_urb_concrete  = m2_app_urb_s.mul(material_concrete.loc[idx[:,3],:].droplevel(1).unstack(),    axis=1).sum(axis=1, level=0)
kg_app_urb_wood      = m2_app_urb_s.mul(material_wood.loc[idx[:,3],:].droplevel(1).unstack(),        axis=1).sum(axis=1, level=0)
kg_app_urb_copper    = m2_app_urb_s.mul(material_copper.loc[idx[:,3],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_app_urb_aluminium = m2_app_urb_s.mul(material_aluminium.loc[idx[:,3],:].droplevel(1).unstack(),   axis=1).sum(axis=1, level=0)
kg_app_urb_glass     = m2_app_urb_s.mul(material_glass.loc[idx[:,3],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)  
kg_app_urb_brick     = m2_app_urb_s.mul(material_brick.loc[idx[:,3],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)        

kg_hig_urb_steel     = m2_hig_urb_s.mul(material_steel.loc[idx[:,4],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)
kg_hig_urb_cement    = m2_hig_urb_s.mul(material_cement.loc[idx[:,4],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_hig_urb_concrete  = m2_hig_urb_s.mul(material_concrete.loc[idx[:,4],:].droplevel(1).unstack(),    axis=1).sum(axis=1, level=0)
kg_hig_urb_wood      = m2_hig_urb_s.mul(material_wood.loc[idx[:,4],:].droplevel(1).unstack(),        axis=1).sum(axis=1, level=0)
kg_hig_urb_copper    = m2_hig_urb_s.mul(material_copper.loc[idx[:,4],:].droplevel(1).unstack(),      axis=1).sum(axis=1, level=0)
kg_hig_urb_aluminium = m2_hig_urb_s.mul(material_aluminium.loc[idx[:,4],:].droplevel(1).unstack(),   axis=1).sum(axis=1, level=0) 
kg_hig_urb_glass     = m2_hig_urb_s.mul(material_glass.loc[idx[:,4],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)    
kg_hig_urb_brick     = m2_hig_urb_s.mul(material_brick.loc[idx[:,4],:].droplevel(1).unstack(),       axis=1).sum(axis=1, level=0)    

# Commercial Building materials (in Million kg)
kg_office_steel     = m2_office_s.mul(material_com_steel.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_office_cement    = m2_office_s.mul(material_com_cement.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),    axis=1).sum(axis=1, level=0)
kg_office_concrete  = m2_office_s.mul(material_com_concrete.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),  axis=1).sum(axis=1, level=0)
kg_office_wood      = m2_office_s.mul(material_com_wood.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),      axis=1).sum(axis=1, level=0)
kg_office_copper    = m2_office_s.mul(material_com_copper.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),    axis=1).sum(axis=1, level=0)
kg_office_aluminium = m2_office_s.mul(material_com_aluminium.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(), axis=1).sum(axis=1, level=0)
kg_office_glass     = m2_office_s.mul(material_com_glass.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_office_brick     = m2_office_s.mul(material_com_brick.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)

kg_retail_steel     = m2_retail_s.mul(material_com_steel.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_retail_cement    = m2_retail_s.mul(material_com_cement.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),    axis=1).sum(axis=1, level=0)
kg_retail_concrete  = m2_retail_s.mul(material_com_concrete.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),  axis=1).sum(axis=1, level=0)
kg_retail_wood      = m2_retail_s.mul(material_com_wood.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),      axis=1).sum(axis=1, level=0)
kg_retail_copper    = m2_retail_s.mul(material_com_copper.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),    axis=1).sum(axis=1, level=0)
kg_retail_aluminium = m2_retail_s.mul(material_com_aluminium.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(), axis=1).sum(axis=1, level=0)
kg_retail_glass     = m2_retail_s.mul(material_com_glass.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_retail_brick     = m2_retail_s.mul(material_com_brick.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)

kg_hotels_steel     = m2_hotels_s.mul(material_com_steel.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hotels_cement    = m2_hotels_s.mul(material_com_cement.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),    axis=1).sum(axis=1, level=0)
kg_hotels_concrete  = m2_hotels_s.mul(material_com_concrete.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),  axis=1).sum(axis=1, level=0)
kg_hotels_wood      = m2_hotels_s.mul(material_com_wood.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),      axis=1).sum(axis=1, level=0)
kg_hotels_copper    = m2_hotels_s.mul(material_com_copper.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),    axis=1).sum(axis=1, level=0)
kg_hotels_aluminium = m2_hotels_s.mul(material_com_aluminium.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(), axis=1).sum(axis=1, level=0)
kg_hotels_glass     = m2_hotels_s.mul(material_com_glass.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hotels_brick     = m2_hotels_s.mul(material_com_brick.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)

kg_govern_steel     = m2_govern_s.mul(material_com_steel.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),      axis=1).sum(axis=1, level=0)
kg_govern_cement    = m2_govern_s.mul(material_com_cement.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_govern_concrete  = m2_govern_s.mul(material_com_concrete.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),   axis=1).sum(axis=1, level=0)
kg_govern_wood      = m2_govern_s.mul(material_com_wood.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),       axis=1).sum(axis=1, level=0)
kg_govern_copper    = m2_govern_s.mul(material_com_copper.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_govern_aluminium = m2_govern_s.mul(material_com_aluminium.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),  axis=1).sum(axis=1, level=0)
kg_govern_glass     = m2_govern_s.mul(material_com_glass.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),      axis=1).sum(axis=1, level=0)
kg_govern_brick     = m2_govern_s.mul(material_com_brick.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),      axis=1).sum(axis=1, level=0)

# Summing commercial material stock (Million kg)
kg_steel_comm       = kg_office_steel + kg_retail_steel + kg_hotels_steel + kg_govern_steel
kg_cement_comm      = kg_office_cement + kg_retail_cement + kg_hotels_cement + kg_govern_cement
kg_concrete_comm    = kg_office_concrete + kg_retail_concrete + kg_hotels_concrete + kg_govern_concrete
kg_wood_comm        = kg_office_wood + kg_retail_wood + kg_hotels_wood + kg_govern_wood
kg_copper_comm      = kg_office_copper + kg_retail_copper + kg_hotels_copper + kg_govern_copper
kg_aluminium_comm   = kg_office_aluminium + kg_retail_aluminium + kg_hotels_aluminium + kg_govern_aluminium
kg_glass_comm       = kg_office_glass + kg_retail_glass + kg_hotels_glass + kg_govern_glass

# Summing across RESIDENTIAL building types (millions of kg, in stock)
kg_steel_urb = kg_hig_urb_steel + kg_app_urb_steel + kg_sem_urb_steel + kg_det_urb_steel 
kg_steel_rur = kg_hig_rur_steel + kg_app_rur_steel + kg_sem_rur_steel + kg_det_rur_steel 

kg_cement_urb = kg_hig_urb_cement + kg_app_urb_cement + kg_sem_urb_cement + kg_det_urb_cement 
kg_cement_rur = kg_hig_rur_cement + kg_app_rur_cement + kg_sem_rur_cement + kg_det_rur_cement

kg_concrete_urb = kg_hig_urb_concrete + kg_app_urb_concrete + kg_sem_urb_concrete + kg_det_urb_concrete 
kg_concrete_rur = kg_hig_rur_concrete + kg_app_rur_concrete + kg_sem_rur_concrete + kg_det_rur_concrete

kg_wood_urb = kg_hig_urb_wood + kg_app_urb_wood + kg_sem_urb_wood + kg_det_urb_wood 
kg_wood_rur = kg_hig_rur_wood + kg_app_rur_wood + kg_sem_rur_wood + kg_det_rur_wood

kg_copper_urb = kg_hig_urb_copper + kg_app_urb_copper + kg_sem_urb_copper + kg_det_urb_copper 
kg_copper_rur = kg_hig_rur_copper + kg_app_rur_copper + kg_sem_rur_copper + kg_det_rur_copper

kg_aluminium_urb = kg_hig_urb_aluminium + kg_app_urb_aluminium + kg_sem_urb_aluminium + kg_det_urb_aluminium 
kg_aluminium_rur = kg_hig_rur_aluminium + kg_app_rur_aluminium + kg_sem_rur_aluminium + kg_det_rur_aluminium

kg_glass_urb = kg_hig_urb_glass + kg_app_urb_glass + kg_sem_urb_glass + kg_det_urb_glass 
kg_glass_rur = kg_hig_rur_glass + kg_app_rur_glass + kg_sem_rur_glass + kg_det_rur_glass

# Sums for total building material use (in-stock, millions of kg)
kg_steel    = kg_steel_urb + kg_steel_rur + kg_steel_comm
kg_cement   = kg_cement_urb + kg_cement_rur + kg_cement_comm
kg_concrete = kg_concrete_urb + kg_concrete_rur + kg_concrete_comm
kg_wood     = kg_wood_urb + kg_wood_rur + kg_wood_comm
kg_copper   = kg_copper_urb + kg_copper_rur + kg_copper_comm
kg_aluminium = kg_aluminium_urb + kg_aluminium_rur + kg_aluminium_comm
kg_glass   = kg_glass_urb + kg_glass_rur + kg_glass_comm

#%% GRAPHS
# Preparing variables for plots 

kg_steel_world = kg_steel.sum(axis=1)
kg_steel_world_index = kg_steel_world * 100/kg_steel_world[1971]

kg_steel_index = pd.DataFrame(index=kg_steel.index, columns=kg_steel.columns)
for region in range(1,27):
    kg_steel_index[region] = kg_steel[region] * 100/kg_steel[region][1971]

kg_wood_world = kg_wood.sum(axis=1)
kg_wood_world_index = kg_wood_world * 100/kg_wood_world[1971]

kg_wood_index = pd.DataFrame(index=kg_wood.index, columns=kg_wood.columns)
for region in range(1,27):
    kg_wood_index[region] = kg_wood[region] * 100/kg_wood[region][1971]

# ----------- GRAPH (1) Wood vs Steel in Japan -------------

# Setting up a comparison between wood & steel demand (stock, indexed) for Japan, plotvar is added to bundle the two variables (though not used here)
plotvar = pd.DataFrame(index=kg_copper_urb.index, columns=[0,1])
plotvar[0] = kg_steel_index[23].tail(100)     # 23 = Japan
plotvar[1] = kg_wood_index[23].tail(100)      # 23 = Japan

# Drawing the plot comparing wood & steel demand (stock, indexed) for Japan
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
plt.plot(plotvar[0], color='blue', linewidth=3.3, label="Steel")
plt.plot(plotvar[1], color='brown', linewidth=3.3, label="Wood")
plt.tick_params(axis='both', labelsize=20)
fig.suptitle('Wood demand vs Steel demand in Japan (indexed at 1971)', fontsize=28)
plt.xlabel('years', fontsize=25)
plt.ylabel('index', fontsize=25)
plt.legend(loc='upper left', prop={'size': 25}, borderaxespad=0., frameon=False)
fig.savefig('output\\' + scenario_select + '\\wood_vs_steel_index_Japan.png')

# ---------- GRAPH (2) Copper global stock, total, urban & rural ------------

# Setting up a comparison between urban & rural copper demand (stock, global) 
plotvar = pd.DataFrame(index=kg_copper_urb.index, columns=[0,1])
plotvar[0] = kg_copper.sum(axis=1)      # = World total
plotvar[1] = kg_copper_urb.sum(axis=1)  # = World total urban
plotvar[2] = kg_copper_rur.sum(axis=1)  # = World total rural
plotvar[3] = kg_copper_comm.sum(axis=1) # = World total commercial

# Drawing the plot comparing wood & steel demand (stock, indexed) for Japan
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
plt.plot(plotvar[0].tail(100), color='orange', linewidth=3.3, label="Total copper")
plt.plot(plotvar[1].tail(100), '--', color='orange', linewidth=3.3, label="Urban copper")
plt.plot(plotvar[2].tail(100), '-.', color='red', linewidth=3.3, label="Rural copper")
plt.plot(plotvar[3].tail(100), '.', color='red', linewidth=3.3, label="Commercial copper")
plt.tick_params(axis='both', labelsize=20)
fig.suptitle('Global building copper stock (urban/rural/commercial)', fontsize=28)
plt.xlabel('years', fontsize=25)
plt.ylabel('copper stock (in 1000 tonnes)', fontsize=25)
plt.legend(loc='upper left', prop={'size': 25}, borderaxespad=0., frameon=False)
fig.savefig('output\\' + scenario_select + '\\Global_copper_urb_rur.png')

# ---------- GRAPH (3) Division across building types (steel stock, global total) ----------------------

# Setting up 
plotvar = pd.DataFrame(index=kg_hig_urb_steel.index, columns=[0,1,2,3])
plotvar[0] = (kg_hig_urb_steel.sum(axis=1) + kg_hig_rur_steel.sum(axis=1))/1000    # = high-rise
plotvar[1] = (kg_app_urb_steel.sum(axis=1) + kg_app_rur_steel.sum(axis=1))/1000    # = appartments
plotvar[2] = (kg_sem_urb_steel.sum(axis=1) + kg_sem_rur_steel.sum(axis=1))/1000    # = semi-detached
plotvar[3] = (kg_det_urb_steel.sum(axis=1) + kg_det_rur_steel.sum(axis=1))/1000    # = detached

# Drawing the plot comparing steel demand, by building type (stock, global)
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
plt.plot(plotvar[0].tail(100), color='red', linewidth=3.3, label="high-rise")
plt.plot(plotvar[1].tail(100), color='orange', linewidth=3.3, label="appartments")
plt.plot(plotvar[2].tail(100), color='blue', linewidth=3.3, label="semi-detached")
plt.plot(plotvar[3].tail(100), color='green', linewidth=3.3, label="detached")
plt.tick_params(axis='both', labelsize=20)
fig.suptitle('Global steel stock by building type', fontsize=28)
plt.xlabel('years', fontsize=25)
plt.ylabel('steel stock (in Mega-tonnes)', fontsize=25)
plt.legend(loc='upper left', prop={'size': 25}, borderaxespad=0., frameon=False)
fig.savefig('output\\' + scenario_select + '\\building_type_steel_global.png')

# ---------- GRAPH (4) Total steel stock by grouped region ----------------------

# North-America, Latin America, Africa, Europe, Russia etc., Middle-East, India, China, Other Asia, Oceania 
col_list = [[1,2,3],[4,5,6],[7,8,9,10,26],[11,12],[14,15,16],[13,17],[18],[20],[19,21,22,23,25],[24]]

# Setting up
plotvar = pd.DataFrame(index=kg_copper_urb.index, columns=[0,1])
plotvar[0] = kg_steel[col_list[0]].sum(axis=1)          # = NA
plotvar[1] = kg_steel[col_list[1]].sum(axis=1)          # = LA
plotvar[2] = kg_steel[col_list[2]].sum(axis=1)          # = AF
plotvar[3] = kg_steel[col_list[3]].sum(axis=1)          # = EU
plotvar[4] = kg_steel[col_list[4]].sum(axis=1)          # = RU
plotvar[5] = kg_steel[col_list[5]].sum(axis=1)          # = ME
plotvar[6] = kg_steel[col_list[6]].sum(axis=1)          # = IN
plotvar[7] = kg_steel[col_list[7]].sum(axis=1)          # = CH
plotvar[8] = kg_steel[col_list[8]].sum(axis=1)          # = AS
plotvar[9] = kg_steel[col_list[9]].sum(axis=1)          # = AU

# 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'
# Drawing the plot comparing Total steel stock in grouped regions
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
plt.plot(plotvar[0].tail(100)/1000000, color='b', linewidth=3.3, label="North-America")
plt.plot(plotvar[1].tail(100)/1000000, color='g', linewidth=3.3, label="Latin America")
plt.plot(plotvar[2].tail(100)/1000000, color='r', linewidth=3.3, label="Africa")
plt.plot(plotvar[3].tail(100)/1000000, color='c', linewidth=3.3, label="Europe")
plt.plot(plotvar[4].tail(100)/1000000, color='m', linewidth=3.3, label="Russia +")
plt.plot(plotvar[5].tail(100)/1000000, color='y', linewidth=3.3, label="Middle-East")
plt.plot(plotvar[6].tail(100)/1000000, color='k', linewidth=3.3, label="India")
plt.plot(plotvar[7].tail(100)/1000000, color='#82cafc', linewidth=3.3, label="China")    #https://xkcd.com/color/rgb/
plt.plot(plotvar[8].tail(100)/1000000, color='#cb0162', linewidth=3.3, label="Rest of Asia")
plt.plot(plotvar[9].tail(100)/1000000, color='#a9561e', linewidth=3.3, label="Oceania")
plt.tick_params(axis='both', labelsize=20)
fig.suptitle('Total steel stock', fontsize=28)
plt.xlabel('years', fontsize=25)
plt.ylabel('steel stock (in Gt)', fontsize=25)
plt.legend(loc='upper left', prop={'size': 25}, borderaxespad=0., frameon=False)
fig.savefig('output\\' + scenario_select + '\\Total_steel_grouped.png')

# ------------- Graph (5) --------------------------------------------------------------------

# show steel (in stock) in urban high-rise buildings
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
plt.plot(kg_hig_urb_steel, linewidth=3.3)
plt.tick_params(axis='both', labelsize=20)
fig.suptitle('Steel in urban high-rise buildings', fontsize=28)
plt.xlabel('years', fontsize=25)
plt.ylabel('*1000 tons of steel', fontsize=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.savefig('output\\' + scenario_select + '\\kg_hig_urb_steel.png')


#%% Material inflow & outflow & stock

# RURAL material inlow (Millions of kgs = *1000 tons)
kg_det_rur_steel_i     = m2_det_rur_i.mul(material_steel.loc[idx[:,1],:].droplevel(1))
kg_det_rur_cement_i    = m2_det_rur_i.mul(material_cement.loc[idx[:,1],:].droplevel(1))
kg_det_rur_concrete_i  = m2_det_rur_i.mul(material_concrete.loc[idx[:,1],:].droplevel(1))
kg_det_rur_wood_i      = m2_det_rur_i.mul(material_wood.loc[idx[:,1],:].droplevel(1))
kg_det_rur_copper_i    = m2_det_rur_i.mul(material_copper.loc[idx[:,1],:].droplevel(1))
kg_det_rur_aluminium_i = m2_det_rur_i.mul(material_aluminium.loc[idx[:,1],:].droplevel(1))
kg_det_rur_glass_i     = m2_det_rur_i.mul(material_glass.loc[idx[:,1],:].droplevel(1))
kg_det_rur_brick_i     = m2_det_rur_i.mul(material_brick.loc[idx[:,1],:].droplevel(1))

kg_sem_rur_steel_i     = m2_sem_rur_i.mul(material_steel.loc[idx[:,2],:].droplevel(1))
kg_sem_rur_cement_i    = m2_sem_rur_i.mul(material_cement.loc[idx[:,2],:].droplevel(1))
kg_sem_rur_concrete_i  = m2_sem_rur_i.mul(material_concrete.loc[idx[:,2],:].droplevel(1))
kg_sem_rur_wood_i      = m2_sem_rur_i.mul(material_wood.loc[idx[:,2],:].droplevel(1))
kg_sem_rur_copper_i    = m2_sem_rur_i.mul(material_copper.loc[idx[:,2],:].droplevel(1))
kg_sem_rur_aluminium_i = m2_sem_rur_i.mul(material_aluminium.loc[idx[:,2],:].droplevel(1)) 
kg_sem_rur_glass_i     = m2_sem_rur_i.mul(material_glass.loc[idx[:,2],:].droplevel(1))     
kg_sem_rur_brick_i     = m2_sem_rur_i.mul(material_brick.loc[idx[:,2],:].droplevel(1))     

kg_app_rur_steel_i     = m2_app_rur_i.mul(material_steel.loc[idx[:,3],:].droplevel(1))
kg_app_rur_cement_i    = m2_app_rur_i.mul(material_cement.loc[idx[:,3],:].droplevel(1))
kg_app_rur_concrete_i  = m2_app_rur_i.mul(material_concrete.loc[idx[:,3],:].droplevel(1))
kg_app_rur_wood_i      = m2_app_rur_i.mul(material_wood.loc[idx[:,3],:].droplevel(1))
kg_app_rur_copper_i    = m2_app_rur_i.mul(material_copper.loc[idx[:,3],:].droplevel(1))
kg_app_rur_aluminium_i = m2_app_rur_i.mul(material_aluminium.loc[idx[:,3],:].droplevel(1))  
kg_app_rur_glass_i     = m2_app_rur_i.mul(material_glass.loc[idx[:,3],:].droplevel(1))     
kg_app_rur_brick_i     = m2_app_rur_i.mul(material_brick.loc[idx[:,3],:].droplevel(1))     

kg_hig_rur_steel_i     = m2_hig_rur_i.mul(material_steel.loc[idx[:,4],:].droplevel(1))
kg_hig_rur_cement_i    = m2_hig_rur_i.mul(material_cement.loc[idx[:,4],:].droplevel(1))
kg_hig_rur_concrete_i  = m2_hig_rur_i.mul(material_concrete.loc[idx[:,4],:].droplevel(1))
kg_hig_rur_wood_i      = m2_hig_rur_i.mul(material_wood.loc[idx[:,4],:].droplevel(1))
kg_hig_rur_copper_i    = m2_hig_rur_i.mul(material_copper.loc[idx[:,4],:].droplevel(1))
kg_hig_rur_aluminium_i = m2_hig_rur_i.mul(material_aluminium.loc[idx[:,4],:].droplevel(1))  
kg_hig_rur_glass_i     = m2_hig_rur_i.mul(material_glass.loc[idx[:,4],:].droplevel(1))        
kg_hig_rur_brick_i     = m2_hig_rur_i.mul(material_brick.loc[idx[:,4],:].droplevel(1))        

# URBAN material inflow (millions of kgs)
kg_det_urb_steel_i     = m2_det_urb_i.mul(material_steel.loc[idx[:,1],:].droplevel(1))
kg_det_urb_cement_i    = m2_det_urb_i.mul(material_cement.loc[idx[:,1],:].droplevel(1))
kg_det_urb_concrete_i  = m2_det_urb_i.mul(material_concrete.loc[idx[:,1],:].droplevel(1))
kg_det_urb_wood_i      = m2_det_urb_i.mul(material_wood.loc[idx[:,1],:].droplevel(1))
kg_det_urb_copper_i    = m2_det_urb_i.mul(material_copper.loc[idx[:,1],:].droplevel(1))
kg_det_urb_aluminium_i = m2_det_urb_i.mul(material_aluminium.loc[idx[:,1],:].droplevel(1))
kg_det_urb_glass_i     = m2_det_urb_i.mul(material_glass.loc[idx[:,1],:].droplevel(1))
kg_det_urb_brick_i     = m2_det_urb_i.mul(material_brick.loc[idx[:,1],:].droplevel(1))

kg_sem_urb_steel_i     = m2_sem_urb_i.mul(material_steel.loc[idx[:,2],:].droplevel(1))
kg_sem_urb_cement_i    = m2_sem_urb_i.mul(material_cement.loc[idx[:,2],:].droplevel(1))
kg_sem_urb_concrete_i  = m2_sem_urb_i.mul(material_concrete.loc[idx[:,2],:].droplevel(1))
kg_sem_urb_wood_i      = m2_sem_urb_i.mul(material_wood.loc[idx[:,2],:].droplevel(1))
kg_sem_urb_copper_i    = m2_sem_urb_i.mul(material_copper.loc[idx[:,2],:].droplevel(1))
kg_sem_urb_aluminium_i = m2_sem_urb_i.mul(material_aluminium.loc[idx[:,2],:].droplevel(1))  
kg_sem_urb_glass_i     = m2_sem_urb_i.mul(material_glass.loc[idx[:,2],:].droplevel(1))    
kg_sem_urb_brick_i     = m2_sem_urb_i.mul(material_brick.loc[idx[:,2],:].droplevel(1))    

kg_app_urb_steel_i     = m2_app_urb_i.mul(material_steel.loc[idx[:,3],:].droplevel(1))
kg_app_urb_cement_i    = m2_app_urb_i.mul(material_cement.loc[idx[:,3],:].droplevel(1))
kg_app_urb_concrete_i  = m2_app_urb_i.mul(material_concrete.loc[idx[:,3],:].droplevel(1))
kg_app_urb_wood_i      = m2_app_urb_i.mul(material_wood.loc[idx[:,3],:].droplevel(1))
kg_app_urb_copper_i    = m2_app_urb_i.mul(material_copper.loc[idx[:,3],:].droplevel(1))
kg_app_urb_aluminium_i = m2_app_urb_i.mul(material_aluminium.loc[idx[:,3],:].droplevel(1))
kg_app_urb_glass_i     = m2_app_urb_i.mul(material_glass.loc[idx[:,3],:].droplevel(1))
kg_app_urb_brick_i     = m2_app_urb_i.mul(material_brick.loc[idx[:,3],:].droplevel(1))

kg_hig_urb_steel_i     = m2_hig_urb_i.mul(material_steel.loc[idx[:,4],:].droplevel(1))
kg_hig_urb_cement_i    = m2_hig_urb_i.mul(material_cement.loc[idx[:,4],:].droplevel(1))
kg_hig_urb_concrete_i  = m2_hig_urb_i.mul(material_concrete.loc[idx[:,4],:].droplevel(1))
kg_hig_urb_wood_i      = m2_hig_urb_i.mul(material_wood.loc[idx[:,4],:].droplevel(1))
kg_hig_urb_copper_i    = m2_hig_urb_i.mul(material_copper.loc[idx[:,4],:].droplevel(1))
kg_hig_urb_aluminium_i = m2_hig_urb_i.mul(material_aluminium.loc[idx[:,4],:].droplevel(1))
kg_hig_urb_glass_i     = m2_hig_urb_i.mul(material_glass.loc[idx[:,4],:].droplevel(1))
kg_hig_urb_brick_i     = m2_hig_urb_i.mul(material_brick.loc[idx[:,4],:].droplevel(1))

# RURAL material OUTflow (Millions of kgs = *1000 tons)
kg_det_rur_steel_o     = m2_det_rur_o.mul(material_steel.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_rur_cement_o    = m2_det_rur_o.mul(material_cement.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_rur_concrete_o  = m2_det_rur_o.mul(material_concrete.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_rur_wood_o      = m2_det_rur_o.mul(material_wood.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_rur_copper_o    = m2_det_rur_o.mul(material_copper.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_rur_aluminium_o = m2_det_rur_o.mul(material_aluminium.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_rur_glass_o     = m2_det_rur_o.mul(material_glass.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_rur_brick_o     = m2_det_rur_o.mul(material_brick.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)

kg_sem_rur_steel_o     = m2_sem_rur_o.mul(material_steel.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_rur_cement_o    = m2_sem_rur_o.mul(material_cement.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_rur_concrete_o  = m2_sem_rur_o.mul(material_concrete.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_rur_wood_o      = m2_sem_rur_o.mul(material_wood.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_rur_copper_o    = m2_sem_rur_o.mul(material_copper.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_rur_aluminium_o = m2_sem_rur_o.mul(material_aluminium.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)   
kg_sem_rur_glass_o     = m2_sem_rur_o.mul(material_glass.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)  
kg_sem_rur_brick_o     = m2_sem_rur_o.mul(material_brick.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)  

kg_app_rur_steel_o     = m2_app_rur_o.mul(material_steel.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_rur_cement_o    = m2_app_rur_o.mul(material_cement.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_rur_concrete_o  = m2_app_rur_o.mul(material_concrete.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_rur_wood_o      = m2_app_rur_o.mul(material_wood.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_rur_copper_o    = m2_app_rur_o.mul(material_copper.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_rur_aluminium_o = m2_app_rur_o.mul(material_aluminium.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)   
kg_app_rur_glass_o     = m2_app_rur_o.mul(material_glass.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_rur_brick_o     = m2_app_rur_o.mul(material_brick.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)

kg_hig_rur_steel_o     = m2_hig_rur_o.mul(material_steel.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_rur_cement_o    = m2_hig_rur_o.mul(material_cement.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_rur_concrete_o  = m2_hig_rur_o.mul(material_concrete.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_rur_wood_o      = m2_hig_rur_o.mul(material_wood.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_rur_copper_o    = m2_hig_rur_o.mul(material_copper.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_rur_aluminium_o = m2_hig_rur_o.mul(material_aluminium.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_rur_glass_o     = m2_hig_rur_o.mul(material_glass.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_rur_brick_o     = m2_hig_rur_o.mul(material_brick.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)

# URBAN material OUTflow (millions of kgs)
kg_det_urb_steel_o     = m2_det_urb_o.mul(material_steel.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_urb_cement_o    = m2_det_urb_o.mul(material_cement.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_urb_concrete_o  = m2_det_urb_o.mul(material_concrete.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_urb_wood_o      = m2_det_urb_o.mul(material_wood.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_urb_copper_o    = m2_det_urb_o.mul(material_copper.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_urb_aluminium_o = m2_det_urb_o.mul(material_aluminium.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_urb_glass_o     = m2_det_urb_o.mul(material_glass.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_det_urb_brick_o     = m2_det_urb_o.mul(material_brick.loc[idx[:,1],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)

kg_sem_urb_steel_o     = m2_sem_urb_o.mul(material_steel.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_urb_cement_o    = m2_sem_urb_o.mul(material_cement.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_urb_concrete_o  = m2_sem_urb_o.mul(material_concrete.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_urb_wood_o      = m2_sem_urb_o.mul(material_wood.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_urb_copper_o    = m2_sem_urb_o.mul(material_copper.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_sem_urb_aluminium_o = m2_sem_urb_o.mul(material_aluminium.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)  
kg_sem_urb_glass_o     = m2_sem_urb_o.mul(material_glass.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)        
kg_sem_urb_brick_o     = m2_sem_urb_o.mul(material_brick.loc[idx[:,2],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)        

kg_app_urb_steel_o     = m2_app_urb_o.mul(material_steel.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_urb_cement_o    = m2_app_urb_o.mul(material_cement.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_urb_concrete_o  = m2_app_urb_o.mul(material_concrete.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_urb_wood_o      = m2_app_urb_o.mul(material_wood.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_urb_copper_o    = m2_app_urb_o.mul(material_copper.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_app_urb_aluminium_o = m2_app_urb_o.mul(material_aluminium.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)  
kg_app_urb_glass_o     = m2_app_urb_o.mul(material_glass.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)   
kg_app_urb_brick_o     = m2_app_urb_o.mul(material_brick.loc[idx[:,3],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)   

kg_hig_urb_steel_o     = m2_hig_urb_o.mul(material_steel.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_urb_cement_o    = m2_hig_urb_o.mul(material_cement.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_urb_concrete_o  = m2_hig_urb_o.mul(material_concrete.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_urb_wood_o      = m2_hig_urb_o.mul(material_wood.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_urb_copper_o    = m2_hig_urb_o.mul(material_copper.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hig_urb_aluminium_o = m2_hig_urb_o.mul(material_aluminium.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)  
kg_hig_urb_glass_o     = m2_hig_urb_o.mul(material_glass.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)   
kg_hig_urb_brick_o     = m2_hig_urb_o.mul(material_brick.loc[idx[:,4],:].droplevel(1).unstack(),     axis=1).sum(axis=1, level=0)   

# Commercial Building materials INFLOW (in Million kg)
kg_office_steel_i     = m2_office_i.mul(material_com_steel.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1))
kg_office_cement_i    = m2_office_i.mul(material_com_cement.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1))
kg_office_concrete_i  = m2_office_i.mul(material_com_concrete.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1))
kg_office_wood_i      = m2_office_i.mul(material_com_wood.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1))
kg_office_copper_i    = m2_office_i.mul(material_com_copper.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1))
kg_office_aluminium_i = m2_office_i.mul(material_com_aluminium.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1))
kg_office_glass_i     = m2_office_i.mul(material_com_glass.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1))
kg_office_brick_i     = m2_office_i.mul(material_com_brick.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1))

kg_retail_steel_i     = m2_retail_i.mul(material_com_steel.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1))
kg_retail_cement_i    = m2_retail_i.mul(material_com_cement.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1))
kg_retail_concrete_i  = m2_retail_i.mul(material_com_concrete.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1))
kg_retail_wood_i      = m2_retail_i.mul(material_com_wood.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1))
kg_retail_copper_i    = m2_retail_i.mul(material_com_copper.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1))
kg_retail_aluminium_i = m2_retail_i.mul(material_com_aluminium.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1))
kg_retail_glass_i     = m2_retail_i.mul(material_com_glass.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1))
kg_retail_brick_i     = m2_retail_i.mul(material_com_brick.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1))

kg_hotels_steel_i     = m2_hotels_i.mul(material_com_steel.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1))
kg_hotels_cement_i    = m2_hotels_i.mul(material_com_cement.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1))
kg_hotels_concrete_i  = m2_hotels_i.mul(material_com_concrete.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1))
kg_hotels_wood_i      = m2_hotels_i.mul(material_com_wood.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1))
kg_hotels_copper_i    = m2_hotels_i.mul(material_com_copper.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1))
kg_hotels_aluminium_i = m2_hotels_i.mul(material_com_aluminium.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1))
kg_hotels_glass_i     = m2_hotels_i.mul(material_com_glass.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1))
kg_hotels_brick_i     = m2_hotels_i.mul(material_com_brick.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1))

kg_govern_steel_i     = m2_govern_i.mul(material_com_steel.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1))
kg_govern_cement_i    = m2_govern_i.mul(material_com_cement.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1))
kg_govern_concrete_i  = m2_govern_i.mul(material_com_concrete.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1))
kg_govern_wood_i      = m2_govern_i.mul(material_com_wood.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1))
kg_govern_copper_i    = m2_govern_i.mul(material_com_copper.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1))
kg_govern_aluminium_i = m2_govern_i.mul(material_com_aluminium.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1))
kg_govern_glass_i     = m2_govern_i.mul(material_com_glass.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1))
kg_govern_brick_i     = m2_govern_i.mul(material_com_brick.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1))

# Commercial Building materials OUTFLOW (in Million kg)
kg_office_steel_o     = m2_office_o.mul(material_com_steel.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_office_cement_o    = m2_office_o.mul(material_com_cement.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_office_concrete_o  = m2_office_o.mul(material_com_concrete.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_office_wood_o      = m2_office_o.mul(material_com_wood.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_office_copper_o    = m2_office_o.mul(material_com_copper.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_office_aluminium_o = m2_office_o.mul(material_com_aluminium.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_office_glass_o     = m2_office_o.mul(material_com_glass.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_office_brick_o     = m2_office_o.mul(material_com_brick.loc[:,idx[:,'Offices']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)

kg_retail_steel_o     = m2_retail_o.mul(material_com_steel.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_retail_cement_o    = m2_retail_o.mul(material_com_cement.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_retail_concrete_o  = m2_retail_o.mul(material_com_concrete.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_retail_wood_o      = m2_retail_o.mul(material_com_wood.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_retail_copper_o    = m2_retail_o.mul(material_com_copper.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_retail_aluminium_o = m2_retail_o.mul(material_com_aluminium.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_retail_glass_o     = m2_retail_o.mul(material_com_glass.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_retail_brick_o     = m2_retail_o.mul(material_com_brick.loc[:,idx[:,'Retail+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)

kg_hotels_steel_o     = m2_hotels_o.mul(material_com_steel.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hotels_cement_o    = m2_hotels_o.mul(material_com_cement.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hotels_concrete_o  = m2_hotels_o.mul(material_com_concrete.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hotels_wood_o      = m2_hotels_o.mul(material_com_wood.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hotels_copper_o    = m2_hotels_o.mul(material_com_copper.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hotels_aluminium_o = m2_hotels_o.mul(material_com_aluminium.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hotels_glass_o     = m2_hotels_o.mul(material_com_glass.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_hotels_brick_o     = m2_hotels_o.mul(material_com_brick.loc[:,idx[:,'Hotels+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)

kg_govern_steel_o     = m2_govern_o.mul(material_com_steel.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_govern_cement_o    = m2_govern_o.mul(material_com_cement.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_govern_concrete_o  = m2_govern_o.mul(material_com_concrete.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_govern_wood_o      = m2_govern_o.mul(material_com_wood.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_govern_copper_o    = m2_govern_o.mul(material_com_copper.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_govern_aluminium_o = m2_govern_o.mul(material_com_aluminium.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_govern_glass_o     = m2_govern_o.mul(material_com_glass.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)
kg_govern_brick_o     = m2_govern_o.mul(material_com_brick.loc[:,idx[:,'Govt+']].droplevel(axis=1, level=1).unstack(),     axis=1).sum(axis=1, level=0)


#%% CSV output (material stock & m2 stock)

length = 3
tag = ['stock', 'inflow', 'outflow']
  
# first, define a function to transpose + combine all variables & add columns to identify material, area & appartment type. Only for csv output
def preprocess(stock, inflow, outflow, area, building, material):
   output_combined = [[]] * length
   output_combined[0] = stock.transpose()
   output_combined[1] = inflow.transpose()
   output_combined[2] = outflow.transpose()
   for item in range(0,length):
      output_combined[item].insert(0,'material', [material] * 26)
      output_combined[item].insert(0,'area', [area] * 26)
      output_combined[item].insert(0,'type', [building] * 26)
      output_combined[item].insert(0,'flow', [tag[item]] * 26)
   return output_combined

# RURAL
kg_det_rur_steel_out     = preprocess(kg_det_rur_steel,     kg_det_rur_steel_i,     kg_det_rur_steel_o,     'rural','detached', 'steel')
kg_det_rur_cement_out    = preprocess(kg_det_rur_cement,    kg_det_rur_cement_i,    kg_det_rur_cement_o,    'rural','detached', 'cement')
kg_det_rur_concrete_out  = preprocess(kg_det_rur_concrete,  kg_det_rur_concrete_i,  kg_det_rur_concrete_o,  'rural','detached', 'concrete') 
kg_det_rur_wood_out      = preprocess(kg_det_rur_wood,      kg_det_rur_wood_i,      kg_det_rur_wood_o,      'rural','detached', 'wood')  
kg_det_rur_copper_out    = preprocess(kg_det_rur_copper,    kg_det_rur_copper_i,    kg_det_rur_copper_o,    'rural','detached', 'copper') 
kg_det_rur_aluminium_out = preprocess(kg_det_rur_aluminium, kg_det_rur_aluminium_i, kg_det_rur_aluminium_o, 'rural','detached', 'aluminium') 
kg_det_rur_glass_out     = preprocess(kg_det_rur_glass,     kg_det_rur_glass_i,     kg_det_rur_glass_o,     'rural','detached', 'glass')
kg_det_rur_brick_out     = preprocess(kg_det_rur_brick,     kg_det_rur_brick_i,     kg_det_rur_brick_o,     'rural','detached', 'brick')

kg_sem_rur_steel_out     = preprocess(kg_sem_rur_steel,     kg_sem_rur_steel_i,     kg_sem_rur_steel_o,     'rural','semi-detached', 'steel')
kg_sem_rur_cement_out    = preprocess(kg_sem_rur_cement,    kg_sem_rur_cement_i,    kg_sem_rur_cement_o,    'rural','semi-detached', 'cement')
kg_sem_rur_concrete_out  = preprocess(kg_sem_rur_concrete,  kg_sem_rur_concrete_i,  kg_sem_rur_concrete_o,  'rural','semi-detached', 'concrete') 
kg_sem_rur_wood_out      = preprocess(kg_sem_rur_wood,      kg_sem_rur_wood_i,      kg_sem_rur_wood_o,      'rural','semi-detached', 'wood')  
kg_sem_rur_copper_out    = preprocess(kg_sem_rur_copper,    kg_sem_rur_copper_i,    kg_sem_rur_copper_o,    'rural','semi-detached', 'copper') 
kg_sem_rur_aluminium_out = preprocess(kg_sem_rur_aluminium, kg_sem_rur_aluminium_i, kg_sem_rur_aluminium_o, 'rural','semi-detached', 'aluminium') 
kg_sem_rur_glass_out     = preprocess(kg_sem_rur_glass,     kg_sem_rur_glass_i,     kg_sem_rur_glass_o,     'rural','semi-detached', 'glass')
kg_sem_rur_brick_out     = preprocess(kg_sem_rur_brick,     kg_sem_rur_brick_i,     kg_sem_rur_brick_o,     'rural','semi-detached', 'brick')

kg_app_rur_steel_out     = preprocess(kg_app_rur_steel,     kg_app_rur_steel_i,     kg_app_rur_steel_o,     'rural','appartments', 'steel')
kg_app_rur_cement_out    = preprocess(kg_app_rur_cement,    kg_app_rur_cement_i,    kg_app_rur_cement_o,    'rural','appartments', 'cement')
kg_app_rur_concrete_out  = preprocess(kg_app_rur_concrete,  kg_app_rur_concrete_i,  kg_app_rur_concrete_o,  'rural','appartments', 'concrete') 
kg_app_rur_wood_out      = preprocess(kg_app_rur_wood,      kg_app_rur_wood_i,      kg_app_rur_wood_o,      'rural','appartments', 'wood')  
kg_app_rur_copper_out    = preprocess(kg_app_rur_copper,    kg_app_rur_copper_i,    kg_app_rur_copper_o,    'rural','appartments', 'copper') 
kg_app_rur_aluminium_out = preprocess(kg_app_rur_aluminium, kg_app_rur_aluminium_i, kg_app_rur_aluminium_o, 'rural','appartments', 'aluminium') 
kg_app_rur_glass_out     = preprocess(kg_app_rur_glass,     kg_app_rur_glass_i,     kg_app_rur_glass_o,     'rural','appartments', 'glass')
kg_app_rur_brick_out     = preprocess(kg_app_rur_brick,     kg_app_rur_brick_i,     kg_app_rur_brick_o,     'rural','appartments', 'brick')

kg_hig_rur_steel_out     = preprocess(kg_hig_rur_steel,     kg_hig_rur_steel_i,     kg_hig_rur_steel_o,     'rural','high-rise', 'steel')
kg_hig_rur_cement_out    = preprocess(kg_hig_rur_cement,    kg_hig_rur_cement_i,    kg_hig_rur_cement_o,    'rural','high-rise', 'cement')
kg_hig_rur_concrete_out  = preprocess(kg_hig_rur_concrete,  kg_hig_rur_concrete_i,  kg_hig_rur_concrete_o,  'rural','high-rise', 'concrete') 
kg_hig_rur_wood_out      = preprocess(kg_hig_rur_wood,      kg_hig_rur_wood_i,      kg_hig_rur_wood_o,      'rural','high-rise', 'wood')  
kg_hig_rur_copper_out    = preprocess(kg_hig_rur_copper,    kg_hig_rur_copper_i,    kg_hig_rur_copper_o,    'rural','high-rise', 'copper') 
kg_hig_rur_aluminium_out = preprocess(kg_hig_rur_aluminium, kg_hig_rur_aluminium_i, kg_hig_rur_aluminium_o, 'rural','high-rise', 'aluminium') 
kg_hig_rur_glass_out     = preprocess(kg_hig_rur_glass,     kg_hig_rur_glass_i,     kg_hig_rur_glass_o,     'rural','high-rise', 'glass')
kg_hig_rur_brick_out     = preprocess(kg_hig_rur_brick,     kg_hig_rur_brick_i,     kg_hig_rur_brick_o,     'rural','high-rise', 'brick')

# URBAN 
kg_det_urb_steel_out     = preprocess(kg_det_urb_steel,     kg_det_urb_steel_i,     kg_det_urb_steel_o,     'urban','detached', 'steel')
kg_det_urb_cement_out    = preprocess(kg_det_urb_cement,    kg_det_urb_cement_i,    kg_det_urb_cement_o,    'urban','detached', 'cement')
kg_det_urb_concrete_out  = preprocess(kg_det_urb_concrete,  kg_det_urb_concrete_i,  kg_det_urb_concrete_o,  'urban','detached', 'concrete') 
kg_det_urb_wood_out      = preprocess(kg_det_urb_wood,      kg_det_urb_wood_i,      kg_det_urb_wood_o,      'urban','detached', 'wood')  
kg_det_urb_copper_out    = preprocess(kg_det_urb_copper,    kg_det_urb_copper_i,    kg_det_urb_copper_o,    'urban','detached', 'copper') 
kg_det_urb_aluminium_out = preprocess(kg_det_urb_aluminium, kg_det_urb_aluminium_i, kg_det_urb_aluminium_o, 'urban','detached', 'aluminium') 
kg_det_urb_glass_out     = preprocess(kg_det_urb_glass,     kg_det_urb_glass_i,     kg_det_urb_glass_o,     'urban','detached', 'glass')
kg_det_urb_brick_out     = preprocess(kg_det_urb_brick,     kg_det_urb_brick_i,     kg_det_urb_brick_o,     'urban','detached', 'brick')

kg_sem_urb_steel_out     = preprocess(kg_sem_urb_steel,     kg_sem_urb_steel_i,     kg_sem_urb_steel_o,     'urban','semi-detached', 'steel')
kg_sem_urb_cement_out    = preprocess(kg_sem_urb_cement,    kg_sem_urb_cement_i,    kg_sem_urb_cement_o,    'urban','semi-detached', 'cement')
kg_sem_urb_concrete_out  = preprocess(kg_sem_urb_concrete,  kg_sem_urb_concrete_i,  kg_sem_urb_concrete_o,  'urban','semi-detached', 'concrete') 
kg_sem_urb_wood_out      = preprocess(kg_sem_urb_wood,      kg_sem_urb_wood_i,      kg_sem_urb_wood_o,      'urban','semi-detached', 'wood')  
kg_sem_urb_copper_out    = preprocess(kg_sem_urb_copper,    kg_sem_urb_copper_i,    kg_sem_urb_copper_o,    'urban','semi-detached', 'copper') 
kg_sem_urb_aluminium_out = preprocess(kg_sem_urb_aluminium, kg_sem_urb_aluminium_i, kg_sem_urb_aluminium_o, 'urban','semi-detached', 'aluminium') 
kg_sem_urb_glass_out     = preprocess(kg_sem_urb_glass,     kg_sem_urb_glass_i,     kg_sem_urb_glass_o,     'urban','semi-detached', 'glass')
kg_sem_urb_brick_out     = preprocess(kg_sem_urb_brick,     kg_sem_urb_brick_i,     kg_sem_urb_brick_o,     'urban','semi-detached', 'brick')

kg_app_urb_steel_out     = preprocess(kg_app_urb_steel,     kg_app_urb_steel_i,     kg_app_urb_steel_o,     'urban','appartments', 'steel')
kg_app_urb_cement_out    = preprocess(kg_app_urb_cement,    kg_app_urb_cement_i,    kg_app_urb_cement_o,    'urban','appartments', 'cement')
kg_app_urb_concrete_out  = preprocess(kg_app_urb_concrete,  kg_app_urb_concrete_i,  kg_app_urb_concrete_o,  'urban','appartments', 'concrete') 
kg_app_urb_wood_out      = preprocess(kg_app_urb_wood,      kg_app_urb_wood_i,      kg_app_urb_wood_o,      'urban','appartments', 'wood')  
kg_app_urb_copper_out    = preprocess(kg_app_urb_copper,    kg_app_urb_copper_i,    kg_app_urb_copper_o,    'urban','appartments', 'copper') 
kg_app_urb_aluminium_out = preprocess(kg_app_urb_aluminium, kg_app_urb_aluminium_i, kg_app_urb_aluminium_o, 'urban','appartments', 'aluminium') 
kg_app_urb_glass_out     = preprocess(kg_app_urb_glass,     kg_app_urb_glass_i,     kg_app_urb_glass_o,     'urban','appartments', 'glass')
kg_app_urb_brick_out     = preprocess(kg_app_urb_brick,     kg_app_urb_brick_i,     kg_app_urb_brick_o,     'urban','appartments', 'brick')

kg_hig_urb_steel_out     = preprocess(kg_hig_urb_steel,     kg_hig_urb_steel_i,     kg_hig_urb_steel_o,     'urban','high-rise', 'steel')
kg_hig_urb_cement_out    = preprocess(kg_hig_urb_cement,    kg_hig_urb_cement_i,    kg_hig_urb_cement_o,    'urban','high-rise', 'cement')
kg_hig_urb_concrete_out  = preprocess(kg_hig_urb_concrete,  kg_hig_urb_concrete_i,  kg_hig_urb_concrete_o,  'urban','high-rise', 'concrete') 
kg_hig_urb_wood_out      = preprocess(kg_hig_urb_wood,      kg_hig_urb_wood_i,      kg_hig_urb_wood_o,      'urban','high-rise', 'wood')  
kg_hig_urb_copper_out    = preprocess(kg_hig_urb_copper,    kg_hig_urb_copper_i,    kg_hig_urb_copper_o,    'urban','high-rise', 'copper') 
kg_hig_urb_aluminium_out = preprocess(kg_hig_urb_aluminium, kg_hig_urb_aluminium_i, kg_hig_urb_aluminium_o, 'urban','high-rise', 'aluminium') 
kg_hig_urb_glass_out     = preprocess(kg_hig_urb_glass,     kg_hig_urb_glass_i,     kg_hig_urb_glass_o,     'urban','high-rise', 'glass')
kg_hig_urb_brick_out     = preprocess(kg_hig_urb_brick,     kg_hig_urb_brick_i,     kg_hig_urb_brick_o,     'urban','high-rise', 'brick')

# COMMERCIAL ------------------------------------------------------------------

# offices
kg_office_steel_out     = preprocess(kg_office_steel,     kg_office_steel_i,     kg_office_steel_o,     'commercial','office', 'steel')
kg_office_cement_out    = preprocess(kg_office_cement,    kg_office_cement_i,    kg_office_cement_o,    'commercial','office', 'cement')
kg_office_concrete_out  = preprocess(kg_office_concrete,  kg_office_concrete_i,  kg_office_concrete_o,  'commercial','office', 'concrete')
kg_office_wood_out      = preprocess(kg_office_wood,      kg_office_wood_i,      kg_office_wood_o,      'commercial','office', 'wood')
kg_office_copper_out    = preprocess(kg_office_copper,    kg_office_copper_i,    kg_office_copper_o,    'commercial','office', 'copper')
kg_office_aluminium_out = preprocess(kg_office_aluminium, kg_office_aluminium_i, kg_office_aluminium_o, 'commercial','office', 'aluminium')
kg_office_glass_out     = preprocess(kg_office_glass,     kg_office_glass_i,     kg_office_glass_o,     'commercial','office', 'glass')
kg_office_brick_out     = preprocess(kg_office_brick,     kg_office_brick_i,     kg_office_brick_o,     'commercial','office', 'brick')

# shops & retail
kg_retail_steel_out     = preprocess(kg_retail_steel,     kg_retail_steel_i,     kg_retail_steel_o,     'commercial','retail', 'steel')
kg_retail_cement_out    = preprocess(kg_retail_cement,    kg_retail_cement_i,    kg_retail_cement_o,    'commercial','retail', 'cement')
kg_retail_concrete_out  = preprocess(kg_retail_concrete,  kg_retail_concrete_i,  kg_retail_concrete_o,  'commercial','retail', 'concrete')
kg_retail_wood_out      = preprocess(kg_retail_wood,      kg_retail_wood_i,      kg_retail_wood_o,      'commercial','retail', 'wood')
kg_retail_copper_out    = preprocess(kg_retail_copper,    kg_retail_copper_i,    kg_retail_copper_o,    'commercial','retail', 'copper')
kg_retail_aluminium_out = preprocess(kg_retail_aluminium, kg_retail_aluminium_i, kg_retail_aluminium_o, 'commercial','retail', 'aluminium')
kg_retail_glass_out     = preprocess(kg_retail_glass,     kg_retail_glass_i,     kg_retail_glass_o,     'commercial','retail', 'glass')
kg_retail_brick_out     = preprocess(kg_retail_brick,     kg_retail_brick_i,     kg_retail_brick_o,     'commercial','retail', 'brick')

# hotels & restaurants
kg_hotels_steel_out     = preprocess(kg_hotels_steel,     kg_hotels_steel_i,     kg_hotels_steel_o,     'commercial','hotels', 'steel')
kg_hotels_cement_out    = preprocess(kg_hotels_cement,    kg_hotels_cement_i,    kg_hotels_cement_o,    'commercial','hotels', 'cement')
kg_hotels_concrete_out  = preprocess(kg_hotels_concrete,  kg_hotels_concrete_i,  kg_hotels_concrete_o,  'commercial','hotels', 'concrete')
kg_hotels_wood_out      = preprocess(kg_hotels_wood,      kg_hotels_wood_i,      kg_hotels_wood_o,      'commercial','hotels', 'wood')
kg_hotels_copper_out    = preprocess(kg_hotels_copper,    kg_hotels_copper_i,    kg_hotels_copper_o,    'commercial','hotels', 'copper')
kg_hotels_aluminium_out = preprocess(kg_hotels_aluminium, kg_hotels_aluminium_i, kg_hotels_aluminium_o, 'commercial','hotels', 'aluminium')
kg_hotels_glass_out     = preprocess(kg_hotels_glass,     kg_hotels_glass_i,     kg_hotels_glass_o,     'commercial','hotels', 'glass')
kg_hotels_brick_out     = preprocess(kg_hotels_brick,     kg_hotels_brick_i,     kg_hotels_brick_o,     'commercial','hotels', 'brick')

# government (schools, government, public transport, hospitals, other)
kg_govern_steel_out     = preprocess(kg_govern_steel,     kg_govern_steel_i,     kg_govern_steel_o,     'commercial','govern', 'steel')
kg_govern_cement_out    = preprocess(kg_govern_cement,    kg_govern_cement_i,    kg_govern_cement_o,    'commercial','govern', 'cement')
kg_govern_concrete_out  = preprocess(kg_govern_concrete,  kg_govern_concrete_i,  kg_govern_concrete_o,  'commercial','govern', 'concrete')
kg_govern_wood_out      = preprocess(kg_govern_wood,      kg_govern_wood_i,      kg_govern_wood_o,      'commercial','govern', 'wood')
kg_govern_copper_out    = preprocess(kg_govern_copper,    kg_govern_copper_i,    kg_govern_copper_o,    'commercial','govern', 'copper')
kg_govern_aluminium_out = preprocess(kg_govern_aluminium, kg_govern_aluminium_i, kg_govern_aluminium_o, 'commercial','govern', 'aluminium')
kg_govern_glass_out     = preprocess(kg_govern_glass,     kg_govern_glass_i,     kg_govern_glass_o,     'commercial','govern', 'glass')
kg_govern_brick_out     = preprocess(kg_govern_brick,     kg_govern_brick_i,     kg_govern_brick_o,     'commercial','govern', 'brick')


# stack into 1 dataframe
frames =    [kg_det_rur_steel_out[0], kg_det_rur_cement_out[0], kg_det_rur_concrete_out[0], kg_det_rur_wood_out[0], kg_det_rur_copper_out[0], kg_det_rur_aluminium_out[0], kg_det_rur_glass_out[0], kg_det_rur_brick_out[0],    
             kg_sem_rur_steel_out[0], kg_sem_rur_cement_out[0], kg_sem_rur_concrete_out[0], kg_sem_rur_wood_out[0], kg_sem_rur_copper_out[0], kg_sem_rur_aluminium_out[0], kg_sem_rur_glass_out[0], kg_sem_rur_brick_out[0],   
             kg_app_rur_steel_out[0], kg_app_rur_cement_out[0], kg_app_rur_concrete_out[0], kg_app_rur_wood_out[0], kg_app_rur_copper_out[0], kg_app_rur_aluminium_out[0], kg_app_rur_glass_out[0], kg_app_rur_brick_out[0],   
             kg_hig_rur_steel_out[0], kg_hig_rur_cement_out[0], kg_hig_rur_concrete_out[0], kg_hig_rur_wood_out[0], kg_hig_rur_copper_out[0], kg_hig_rur_aluminium_out[0], kg_hig_rur_glass_out[0], kg_hig_rur_brick_out[0],   
             kg_det_urb_steel_out[0], kg_det_urb_cement_out[0], kg_det_urb_concrete_out[0], kg_det_urb_wood_out[0], kg_det_urb_copper_out[0], kg_det_urb_aluminium_out[0], kg_det_urb_glass_out[0], kg_det_urb_brick_out[0],    
             kg_sem_urb_steel_out[0], kg_sem_urb_cement_out[0], kg_sem_urb_concrete_out[0], kg_sem_urb_wood_out[0], kg_sem_urb_copper_out[0], kg_sem_urb_aluminium_out[0], kg_sem_urb_glass_out[0], kg_sem_urb_brick_out[0],     
             kg_app_urb_steel_out[0], kg_app_urb_cement_out[0], kg_app_urb_concrete_out[0], kg_app_urb_wood_out[0], kg_app_urb_copper_out[0], kg_app_urb_aluminium_out[0], kg_app_urb_glass_out[0], kg_app_urb_brick_out[0],    
             kg_hig_urb_steel_out[0], kg_hig_urb_cement_out[0], kg_hig_urb_concrete_out[0], kg_hig_urb_wood_out[0], kg_hig_urb_copper_out[0], kg_hig_urb_aluminium_out[0], kg_hig_urb_glass_out[0], kg_hig_urb_brick_out[0],
             kg_office_steel_out[0],  kg_office_cement_out[0],  kg_office_concrete_out[0],  kg_office_wood_out[0],  kg_office_copper_out[0],  kg_office_aluminium_out[0],  kg_office_glass_out[0],  kg_office_brick_out[0],
             kg_retail_steel_out[0],  kg_retail_cement_out[0],  kg_retail_concrete_out[0],  kg_retail_wood_out[0],  kg_retail_copper_out[0],  kg_retail_aluminium_out[0],  kg_retail_glass_out[0],  kg_retail_brick_out[0],
             kg_hotels_steel_out[0],  kg_hotels_cement_out[0],  kg_hotels_concrete_out[0],  kg_hotels_wood_out[0],  kg_hotels_copper_out[0],  kg_hotels_aluminium_out[0],  kg_hotels_glass_out[0],  kg_hotels_brick_out[0],
             kg_govern_steel_out[0],  kg_govern_cement_out[0],  kg_govern_concrete_out[0],  kg_govern_wood_out[0],  kg_govern_copper_out[0],  kg_govern_aluminium_out[0],  kg_govern_glass_out[0],  kg_govern_brick_out[0],
                                                           
             kg_det_rur_steel_out[1], kg_det_rur_cement_out[1], kg_det_rur_concrete_out[1], kg_det_rur_wood_out[1], kg_det_rur_copper_out[1], kg_det_rur_aluminium_out[1], kg_det_rur_glass_out[1], kg_det_rur_brick_out[1],   
             kg_sem_rur_steel_out[1], kg_sem_rur_cement_out[1], kg_sem_rur_concrete_out[1], kg_sem_rur_wood_out[1], kg_sem_rur_copper_out[1], kg_sem_rur_aluminium_out[1], kg_sem_rur_glass_out[1], kg_sem_rur_brick_out[1],   
             kg_app_rur_steel_out[1], kg_app_rur_cement_out[1], kg_app_rur_concrete_out[1], kg_app_rur_wood_out[1], kg_app_rur_copper_out[1], kg_app_rur_aluminium_out[1], kg_app_rur_glass_out[1], kg_app_rur_brick_out[1],     
             kg_hig_rur_steel_out[1], kg_hig_rur_cement_out[1], kg_hig_rur_concrete_out[1], kg_hig_rur_wood_out[1], kg_hig_rur_copper_out[1], kg_hig_rur_aluminium_out[1], kg_hig_rur_glass_out[1], kg_hig_rur_brick_out[1],    
             kg_det_urb_steel_out[1], kg_det_urb_cement_out[1], kg_det_urb_concrete_out[1], kg_det_urb_wood_out[1], kg_det_urb_copper_out[1], kg_det_urb_aluminium_out[1], kg_det_urb_glass_out[1], kg_det_urb_brick_out[1],    
             kg_sem_urb_steel_out[1], kg_sem_urb_cement_out[1], kg_sem_urb_concrete_out[1], kg_sem_urb_wood_out[1], kg_sem_urb_copper_out[1], kg_sem_urb_aluminium_out[1], kg_sem_urb_glass_out[1], kg_sem_urb_brick_out[1],    
             kg_app_urb_steel_out[1], kg_app_urb_cement_out[1], kg_app_urb_concrete_out[1], kg_app_urb_wood_out[1], kg_app_urb_copper_out[1], kg_app_urb_aluminium_out[1], kg_app_urb_glass_out[1], kg_app_urb_brick_out[1],      
             kg_hig_urb_steel_out[1], kg_hig_urb_cement_out[1], kg_hig_urb_concrete_out[1], kg_hig_urb_wood_out[1], kg_hig_urb_copper_out[1], kg_hig_urb_aluminium_out[1], kg_hig_urb_glass_out[1], kg_hig_urb_brick_out[1],
             kg_office_steel_out[1],  kg_office_cement_out[1],  kg_office_concrete_out[1],  kg_office_wood_out[1],  kg_office_copper_out[1],  kg_office_aluminium_out[1],  kg_office_glass_out[1],  kg_office_brick_out[1],
             kg_retail_steel_out[1],  kg_retail_cement_out[1],  kg_retail_concrete_out[1],  kg_retail_wood_out[1],  kg_retail_copper_out[1],  kg_retail_aluminium_out[1],  kg_retail_glass_out[1],  kg_retail_brick_out[1],
             kg_hotels_steel_out[1],  kg_hotels_cement_out[1],  kg_hotels_concrete_out[1],  kg_hotels_wood_out[1],  kg_hotels_copper_out[1],  kg_hotels_aluminium_out[1],  kg_hotels_glass_out[1],  kg_hotels_brick_out[1],
             kg_govern_steel_out[1],  kg_govern_cement_out[1],  kg_govern_concrete_out[1],  kg_govern_wood_out[1],  kg_govern_copper_out[1],  kg_govern_aluminium_out[1],  kg_govern_glass_out[1],  kg_govern_brick_out[1],
            
             kg_det_rur_steel_out[2], kg_det_rur_cement_out[2], kg_det_rur_concrete_out[2], kg_det_rur_wood_out[2], kg_det_rur_copper_out[2], kg_det_rur_aluminium_out[2], kg_det_rur_glass_out[2], kg_det_rur_brick_out[2],   
             kg_sem_rur_steel_out[2], kg_sem_rur_cement_out[2], kg_sem_rur_concrete_out[2], kg_sem_rur_wood_out[2], kg_sem_rur_copper_out[2], kg_sem_rur_aluminium_out[2], kg_sem_rur_glass_out[2], kg_sem_rur_brick_out[2],   
             kg_app_rur_steel_out[2], kg_app_rur_cement_out[2], kg_app_rur_concrete_out[2], kg_app_rur_wood_out[2], kg_app_rur_copper_out[2], kg_app_rur_aluminium_out[2], kg_app_rur_glass_out[2], kg_app_rur_brick_out[2],    
             kg_hig_rur_steel_out[2], kg_hig_rur_cement_out[2], kg_hig_rur_concrete_out[2], kg_hig_rur_wood_out[2], kg_hig_rur_copper_out[2], kg_hig_rur_aluminium_out[2], kg_hig_rur_glass_out[2], kg_hig_rur_brick_out[2],  
             kg_det_urb_steel_out[2], kg_det_urb_cement_out[2], kg_det_urb_concrete_out[2], kg_det_urb_wood_out[2], kg_det_urb_copper_out[2], kg_det_urb_aluminium_out[2], kg_det_urb_glass_out[2], kg_det_urb_brick_out[2],   
             kg_sem_urb_steel_out[2], kg_sem_urb_cement_out[2], kg_sem_urb_concrete_out[2], kg_sem_urb_wood_out[2], kg_sem_urb_copper_out[2], kg_sem_urb_aluminium_out[2], kg_sem_urb_glass_out[2], kg_sem_urb_brick_out[2],   
             kg_app_urb_steel_out[2], kg_app_urb_cement_out[2], kg_app_urb_concrete_out[2], kg_app_urb_wood_out[2], kg_app_urb_copper_out[2], kg_app_urb_aluminium_out[2], kg_app_urb_glass_out[2], kg_app_urb_brick_out[2],   
             kg_hig_urb_steel_out[2], kg_hig_urb_cement_out[2], kg_hig_urb_concrete_out[2], kg_hig_urb_wood_out[2], kg_hig_urb_copper_out[2], kg_hig_urb_aluminium_out[2], kg_hig_urb_glass_out[2], kg_hig_urb_brick_out[2], 
             kg_office_steel_out[2],  kg_office_cement_out[2],  kg_office_concrete_out[2],  kg_office_wood_out[2],  kg_office_copper_out[2],  kg_office_aluminium_out[2],  kg_office_glass_out[2],  kg_office_brick_out[2],
             kg_retail_steel_out[2],  kg_retail_cement_out[2],  kg_retail_concrete_out[2],  kg_retail_wood_out[2],  kg_retail_copper_out[2],  kg_retail_aluminium_out[2],  kg_retail_glass_out[2],  kg_retail_brick_out[2],
             kg_hotels_steel_out[2],  kg_hotels_cement_out[2],  kg_hotels_concrete_out[2],  kg_hotels_wood_out[2],  kg_hotels_copper_out[2],  kg_hotels_aluminium_out[2],  kg_hotels_glass_out[2],  kg_hotels_brick_out[2],
             kg_govern_steel_out[2],  kg_govern_cement_out[2],  kg_govern_concrete_out[2],  kg_govern_wood_out[2],  kg_govern_copper_out[2],  kg_govern_aluminium_out[2],  kg_govern_glass_out[2],  kg_govern_brick_out[2] ]

material_output = pd.concat(frames)
material_output.to_csv('output\\' + scenario_select + '\\material_output.csv') # in kt

# SQUARE METERS (results) ---------------------------------------------------

length = 3
tag = ['stock', 'inflow', 'outflow']

# first, define a function to transpose + combine all variables & add columns to identify material, area & appartment type. Only for csv output
def preprocess_m2(stock, inflow, outflow, area, building):
   output_combined = [[]] * length
   output_combined[0] = stock.transpose()
   output_combined[1] = inflow.transpose()
   output_combined[2] = outflow.transpose()
   for item in range(0,length):
      output_combined[item].insert(0,'area', [area] * 26)
      output_combined[item].insert(0,'type', [building] * 26)
      output_combined[item].insert(0,'flow', [tag[item]] * 26)
   return output_combined

m2_det_rur_out  = preprocess_m2(m2_det_rur, m2_det_rur_i, m2_det_rur_o.sum(axis=1, level=0), 'rural', 'detached')
m2_sem_rur_out  = preprocess_m2(m2_sem_rur, m2_sem_rur_i, m2_sem_rur_o.sum(axis=1, level=0), 'rural', 'semi-detached')
m2_app_rur_out  = preprocess_m2(m2_app_rur, m2_app_rur_i, m2_app_rur_o.sum(axis=1, level=0), 'rural', 'appartments')
m2_hig_rur_out  = preprocess_m2(m2_hig_rur, m2_hig_rur_i, m2_hig_rur_o.sum(axis=1, level=0), 'rural', 'high-rise')

m2_det_urb_out  = preprocess_m2(m2_det_urb, m2_det_urb_i, m2_det_urb_o.sum(axis=1, level=0), 'urban', 'detached')
m2_sem_urb_out  = preprocess_m2(m2_sem_urb, m2_sem_urb_i, m2_sem_urb_o.sum(axis=1, level=0), 'urban', 'semi-detached')
m2_app_urb_out  = preprocess_m2(m2_app_urb, m2_app_urb_i, m2_app_urb_o.sum(axis=1, level=0), 'urban', 'appartments')
m2_hig_urb_out  = preprocess_m2(m2_hig_urb, m2_hig_urb_i, m2_hig_urb_o.sum(axis=1, level=0), 'urban', 'high-rise')

# COMMERCIAL
m2_office_out  = preprocess_m2(commercial_m2_office, m2_office_i, m2_office_o.sum(axis=1, level=0), 'commercial', 'office')
m2_retail_out  = preprocess_m2(commercial_m2_retail, m2_retail_i, m2_retail_o.sum(axis=1, level=0), 'commercial', 'retail')
m2_hotels_out  = preprocess_m2(commercial_m2_hotels, m2_hotels_i, m2_hotels_o.sum(axis=1, level=0), 'commercial', 'hotels')
m2_govern_out  = preprocess_m2(commercial_m2_govern, m2_govern_i, m2_govern_o.sum(axis=1, level=0), 'commercial', 'govern')

frames2 = [m2_det_rur_out[0], m2_sem_rur_out[0], m2_app_rur_out[0], m2_hig_rur_out[0], m2_det_urb_out[0], m2_sem_urb_out[0], m2_app_urb_out[0], m2_hig_urb_out[0],
           m2_office_out[0],  m2_retail_out[0],  m2_hotels_out[0],  m2_govern_out[0],
           m2_det_rur_out[1], m2_sem_rur_out[1], m2_app_rur_out[1], m2_hig_rur_out[1], m2_det_urb_out[1], m2_sem_urb_out[1], m2_app_urb_out[1], m2_hig_urb_out[1],
           m2_office_out[1],  m2_retail_out[1],  m2_hotels_out[1],  m2_govern_out[1],
           m2_det_rur_out[2], m2_sem_rur_out[2], m2_app_rur_out[2], m2_hig_rur_out[2], m2_det_urb_out[2], m2_sem_urb_out[2], m2_app_urb_out[2], m2_hig_urb_out[2],
           m2_office_out[2],  m2_retail_out[2],  m2_hotels_out[2],  m2_govern_out[2] ]

sqmeters_output = pd.concat(frames2)
sqmeters_output.to_csv('output\\' + scenario_select + '\\sqmeters_output.csv') # in m2

#%% PLOT Inflow - Outflow results  ----------------------------------------------

# Setting up a comparison between Chinese stock, inflow & outflow 
plotvar = pd.DataFrame(index=list(range(2000,end_year+1)), columns=[0,1,2])
plotvar[0] = m2.loc[2000:end_year,20]          # = China total m2
plotvar[1] = m2_res_i.loc[2000:end_year,20]        # = China total m2
plotvar[2] = m2_res_o.loc[2000:end_year,20]        # = China total m2

# Drawing the plot comparing Chinese stock, inflow & outflow 
fig, ax1 = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax2 = ax1.twinx()
ax1.set_ylim(ymin=0, ymax=max(plotvar[0])*1.1)
ax2.set_ylim(ymin=0, ymax=max(plotvar[1])*1.1)
ax1.plot(plotvar[0], color='black', linewidth=3.3, label="Total stock")
ax2.plot(plotvar[1], '--', color='green', linewidth=3.3, label="Total inflow")
ax2.plot(plotvar[2], '--', color='red', linewidth=3.3, label="Total outflow")

ax1.set_xlabel('years', fontsize=25)
ax1.set_ylabel('stock (millions of m2)', color='black', fontsize=25)
ax2.set_ylabel('flow  (millions of m2)', color='green', fontsize=25)

ax1.tick_params(axis='both', labelsize=20)
ax2.tick_params(labelsize=20, colors='red')

fig.suptitle('Chinese residential building dynamics', fontsize=28)
ax1.legend(loc='lower left', prop={'size': 25}, borderaxespad=0., frameon=False)
ax2.legend(loc='lower right', prop={'size': 25}, borderaxespad=0., frameon=False)
fig.savefig('output\\' + scenario_select + '\\m2_stock_in_out_China.png')
plt.show()