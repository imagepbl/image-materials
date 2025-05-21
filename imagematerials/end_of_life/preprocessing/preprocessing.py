import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# general
start_year  = 1971
end_year    = 2060
full_time = np.arange(start_year, end_year + 1)

# read csv files
collection_in = pd.read_csv("data\\raw\\end_of_life\\SSP2_2D_RE\\collection.csv")
reuse_in = pd.read_csv("data\\raw\\end_of_life\\SSP2_2D_RE\\reuse.csv")
recycling_in = pd.read_csv("data\\raw\\end_of_life\\SSP2_2D_RE\\recycling.csv")

# renaming columns for consistency
collection_in = collection_in.rename(columns={'regions': 'region'})
reuse_in = reuse_in.rename(columns={'Time': 'time', 'Region':'region', 'Sector': 'sector', 'Element': 'category'} )
recycling_in = recycling_in.rename(columns={'Time': 'time', 'Region':'region', 'Sector': 'sector', 'Element': 'category'} )

# melting material columns
id_vars = ['region', 'time', 'sector', 'category'] 
value_vars = ['Steel',	'Concrete',	'Wood',	'Cu','Aluminium', 'Glass']

collection = pd.melt(
    collection_in,
    id_vars = id_vars,
    value_vars = value_vars,
    var_name = 'material',
    value_name = 'value'
)

reuse = pd.melt(
    reuse_in,
    id_vars = id_vars,
    value_vars = value_vars,
    var_name = 'material',
    value_name = 'value'
)

recycling = pd.melt(
    recycling_in,
    id_vars = id_vars,
    value_vars = value_vars,
    var_name = 'material',
    value_name = 'value'
)


# converting melted dataframes to xarrays
xr_collection = collection.set_index(['time', 'region', 'sector', 'category', 'material']) \
                   .to_xarray()['value']

xr_reuse = reuse.set_index(['time', 'region', 'sector', 'category', 'material']) \
                   .to_xarray()['value']

xr_recycling = recycling.set_index(['time', 'region', 'sector', 'category', 'material']) \
                   .to_xarray()['value']


# define interpolation method
def interpolate_waste_data_xr(ds, start_year, end_year, min_value = 0, max_value = 1):              # collection, reuse, and recycling rate cannot be lower than zero or higher than 1

    reindexed = ds.reindex(time=full_time)
    interpolated = reindexed.interpolate_na(dim='time', method='linear', fill_value='extrapolate')
    clipped = interpolated.clip(min=min_value, max = max_value)

    return clipped

# inter and extrapolated collection, reuse and recycling xr 
xr_collection_interp = interpolate_waste_data_xr(xr_collection, start_year= start_year, end_year=end_year,min_value = 0,max_value = 1)
xr_reuse_interp = interpolate_waste_data_xr(xr_reuse, start_year= start_year, end_year=end_year,min_value = 0,max_value = 1)
xr_recycling_interp = interpolate_waste_data_xr(xr_recycling, start_year= start_year, end_year=end_year,min_value = 0,max_value = 1)

collection = xr_collection_interp
reuse = xr_reuse_interp
recycling = xr_recycling_interp

# importing outflows, melting time, converting to xarray 

flows = pd.read_csv("data\raw\buildings\SSP2_2D_RE\Building_materials.csv")
flows = flows.rename(columns = {"Unnamed: 0":"region", "area": "category"})

id_vars = ['region', 'category', 'material', 'flow'] 
value_vars = ['1971', '1972', '1973',
       '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982',
       '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991',
       '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000',
       '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
       '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
       '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027',
       '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035', '2036',
       '2037', '2038', '2039', '2040', '2041', '2042', '2043', '2044', '2045',
       '2046', '2047', '2048', '2049', '2050', '2051', '2052', '2053', '2054',
       '2055', '2056', '2057', '2058', '2059', '2060']

flows_melted = pd.melt(
    flows,
    id_vars = id_vars,
    value_vars = value_vars,
    var_name = 'time',
    value_name = 'value'
)

xr_flows = flows_melted.set_index(['time', 'region', 'category',  'material', 'flow',]) \
                   .to_xarray()['value']

xr_flows['time'] = xr_flows['time'].astype(int)
outflows = xr_flows.sel(flow = 'outflow')

# only buildings ouflows
collection = collection.sel(category = ["commercial", "rural", "urban"], sector = 'Buildings')
reuse = reuse.sel(category = ["commercial", "rural", "urban"], sector = 'Buildings')
recycling = recycling.sel(category = ["commercial", "rural", "urban"], sector = 'Buildings')

# defining function for EoL
def assign_waste_flows (outflows, collection, reuse, recycling):

    # aligning across outflows, collection, reuse and recycling
    outflows, collection, reuse, recycling = xr.align(outflows, collection, reuse, recycling, join='exact')

    # collection and losses (non-collected waste)
    collected = outflows * collection
    losses = outflows - collected           # non-collected waste

    # reuse of materials within the same sector-category
    reusable = collected * reuse
    remaining = collected - reusable        # non-reused but collected waste

    # recycling and losses from collected but unrecycled waste 
    recyclable = remaining * recycling
    losses += (remaining - recyclable)      # non-reused/recycled but collected waste

    return xr.Dataset({
        "losses": losses,
        "reusable": reusable,
        "recyclable": recyclable
    })

result = assign_waste_flows(outflows, collection, reuse, recycling)

# convert to pandas df 
df = result.to_dataframe().reset_index()

# export to csv
df.to_csv("waste_flows.csv", index=False)

# test collection/reuse/recycling rates

xr_reuse_interp.sel(
    region=1,
    sector='Vehicles',
    category='freight',
    material='Glass'
).plot(marker='o')


plt.show()