"""
This script defines functions to process the infrastructure stock demands for each supply  
chain stage: extraction, processing, transport, pipelines and storage.

It gets the stock demand from the infra parameter that is a result of the compute_all_infra() in 
drivers.py. Then it processes it to bring it into the right format for the stock model (right
dimensions, conversion to xarray, adding units).
"""

import pint
import xarray as xr
import prism
import pandas as pd
import numpy as np

from imagematerials.fossil_fuels.constants import EXTRACTION_TYPES, PIPELINE_TYPES

####################################################################################################
#%%Extraction stage (coal, oil, gas)
####################################################################################################
def compute_extraction_capacity(
        infra: dict[str, dict[str, pd.DataFrame]],
        year_start: int,
        year_end: int,
    ) -> xr.DataArray:
    """Build the extraction capacity (stock) of coal, oil and gas as one xarray.

    Takes the extraction stock tables from compute_all_infra() (one per fuel),
    keeps the relevant years and regions, and brings them into the format the
    model needs: one array with dimensions Time, Region and Type, in kg.

    Parameters
    ----------
    infra: Result of compute_all_infra() in drivers.py. Uses infra["coal"]["extraction"], 
        infra["oil"]["extraction"] and infra["gas"]["extraction"]. Each table has the index
        (type, time, fuel, stage, unit) and one column per region (1-28), with the total weight of 
        the infrastructure in kg.
    year_start: First year to keep.
    year_end: Last year to keep (included).

    Returns
    -------
    Extraction stock in kg (units added with prism.Q_), dimensions (Time, Region, Type).
        Type holds the six extraction technologies in a fixed order that matches the lifetime and 
        material intensity data: Coal Opencast, Coal Underground, Gas Offshore, Gas Onshore, 
        Oil Offshore, Oil Onshore.

    """
    #Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year
    #Load data from drivers.py
    extraction_stock_coal = infra["coal"]["extraction"]
    extraction_stock_oil  = infra["oil"]["extraction"]
    extraction_stock_gas  = infra["gas"]["extraction"]

    # --- NOW reset_index safely ---
    df_extraction_coal = extraction_stock_coal.reset_index()
    df_extraction_oil  = extraction_stock_oil.reset_index()
    df_extraction_gas  = extraction_stock_gas.reset_index()

    #Add fuel column to each df so that we can combine them and then split them again into the 
    # different technologies (coal, oil, gas) after melting the df to long format
    df_extraction_coal["fuel"] = "coal"
    df_extraction_oil["fuel"] = "oil"
    df_extraction_gas["fuel"] = "gas"

    # Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
    df_extraction_all = pd.concat(
        [df_extraction_coal, df_extraction_oil, df_extraction_gas],
        ignore_index=True
    )

    df_extraction_all = df_extraction_all.melt(
        id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
        var_name='DIM_1',
        value_name='value'
    )

    #Combine fuel and type columns to one column "Tech Type"
    df_extraction_all['Tech Type'] = (df_extraction_all['fuel'].str.strip() + ' ' + df_extraction_all['type'].str.strip())

    df_extraction_all['Tech Type'] = (
        df_extraction_all['Tech Type']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
        .replace({
            "Coal Open Cast": "Coal Opencast",
            "Gas Off Shore": "Gas Offshore",
            "Gas On Shore": "Gas Onshore"
        })
    )

    #Drop columns that are not needed for the analysis (stage, unit, type, fuel) since we have already combined the relevant information into the "Tech Type" column and we know that all rows are for the extraction stage and in kg units
    df_extraction_all = df_extraction_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  

    # Starting stock (capacity) ------
    df_extraction_all = df_extraction_all.loc[~df_extraction_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    df_extraction_all = df_extraction_all.loc[df_extraction_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    df_extraction_all['DIM_1'] = df_extraction_all['DIM_1'].astype(int)   

    # Convert to 3D array: (Year, Region, Fuel, Type)
    extractioncap_xr = (
        df_extraction_all
        .set_index(['time', 'DIM_1', 'Tech Type'])
        .to_xarray()['value']
        .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
    )  
    extractioncap_xr = prism.Q_(extractioncap_xr, "kg") # add units

    # Reorder technology coordinate in all three datasets (capacity, lifetime, material intensities) 
    # to match the same order of technologies (coal opencast, coal underground, gas offshore, gas 
    # onshore, oil offshore, oil onshore), so that we can easily combine the data in the model 
    # (since they all have to be in the same order of technologies)
    extractioncap_xr = extractioncap_xr.reindex(Type=EXTRACTION_TYPES)
    
    return extractioncap_xr


####################################################################################################
#%% Processing stage (COAL preparation and OIL refinery)
####################################################################################################
def compute_processing_capacity(
        infra: dict[str, dict[str, pd.DataFrame]],
        year_start: int,
        year_end: int,
    ) -> xr.DataArray:
    """Build the processing capacity (stock) of coal, oil and gas as one xarray.

    Processing covers coal preparation plants, oil refineries and gas processing
    plants. The stock tables from compute_all_infra() are combined, restricted to the
    relevant years and regions, and brought into the format the model needs.

    Parameters
    ----------
    infra: Result of compute_all_infra() in drivers.py. Uses
        infra["coal"]["processing"], infra["oil"]["refinery"] and
        infra["gas"]["processing"]. Each table has one column per region (1-28)
        and the index (type, time, fuel, stage, unit); values are the total
        weight of the infrastructure in kg.
    year_start: First year to keep.
    year_end: Last year to keep (included).

    Returns
    -------
    Processing stock in kg, dimensions (Time, Region, Type). Type holds "Coal Preparation", 
        "Gas Processing" and "Oil Refinery".

    """
    #Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year
    preparation_stock_coal = infra["coal"]["processing"]
    refinery_stock_oil  = infra["oil"]["refinery"]
    processing_stock_gas  = infra["gas"]["processing"]

    # --- FIX INDEX NAMES FIRST (IMPORTANT) ---
    preparation_stock_coal.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )
    refinery_stock_oil.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )
    processing_stock_gas.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )    

    df_processing_coal = preparation_stock_coal.reset_index()
    df_processing_refinery_oil = refinery_stock_oil.reset_index()
    df_processing_gas = processing_stock_gas.reset_index()

    df_processing_coal["fuel"] = "coal"
    df_processing_refinery_oil["fuel"] = "oil"  
    df_processing_gas["fuel"] = "gas"

    #Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
    df_processing_all = pd.concat(
        [df_processing_coal, df_processing_refinery_oil, df_processing_gas],
        ignore_index=True
    )

    df_processing_all = df_processing_all.melt(
        id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
        var_name='DIM_1',
        value_name='value'
    )

    #Combine fuel and type columns to one column "Tech Type"
    df_processing_all['Tech Type'] = df_processing_all['fuel'].str.strip() + ' ' + df_processing_all['type'].str.strip()

    # Clean up the "Tech Type" column to have consistent formatting and naming, and to match the technology names in the lifetime and material intensity data (since we will need to combine these later in the model, so they all need to have the same technology names)
    df_processing_all['Tech Type'] = (
        df_processing_all['Tech Type']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
        .replace({
            "coal preparation": "Coal Preparation",
            "gas processing": "Gas Processing",
            "oil refinery": "Oil Refinery",
        })
    )

    # Drop columns that are not needed for the analysis
    df_processing_all = df_processing_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  

    # combine the processing data for coal, oil, gas into one df and then put it into this 
    df_processing_all = df_processing_all.loc[~df_processing_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    df_processing_all = df_processing_all.loc[df_processing_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    df_processing_all['DIM_1'] = df_processing_all['DIM_1'].astype(int)

    # Convert to 3D array: (Year, Region, Fuel, Type)
    processingcap_xr = (
        df_processing_all
        .set_index(['time', 'DIM_1', 'Tech Type'])
        .to_xarray()['value']
        .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
    )
    processingcap_xr = prism.Q_(processingcap_xr, "kg") #Add units

    return processingcap_xr


####################################################################################################
#%% Processing stage (oil storage)
####################################################################################################
def compute_storage_capacity(
        infra: dict[str, dict[str, pd.DataFrame]],
        year_start: int,
        year_end: int,
    ) -> xr.DataArray:
    """Build the oil storage capacity (stock) as one xarray.

    Only oil has storage in this model. The two oil storage types (crude and products) are added up 
    into a single type, "Oil Storage".

    Parameters
    ----------
    infra: Result of compute_all_infra() in drivers.py. Uses infra["oil"]["storage"]: one column per 
        region (1-28), the index (type, time, fuel, stage, unit), with the types "crude" and "products".
        The values are storage volumes in m3.
    year_start: First year to keep.
    year_end: Last year to keep (included).

    Returns
    -------
        Oil storage stock in m3, dimensions (Time, Region, Type).
        Type contains "Oil Storage" (the sum of "Oil Crude" and "Oil Products").

    """
    #Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year
    oil_storage  = infra["oil"]["storage"]

    oil_storage.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )

    df_storage_all = oil_storage.reset_index() 

    df_storage_all["fuel"] = "oil"

    df_storage_all = df_storage_all.melt(
        id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
        var_name='DIM_1',
        value_name='value'
    )

    #Combine fuel and type columns to one column "Tech Type"
    df_storage_all['Tech Type'] = df_storage_all['fuel'].str.strip() + ' ' + df_storage_all['type'].str.strip()

    # Clean up the "Tech Type" column to have consistent formatting and naming, and to match the 
    # technology names in the lifetime and material intensity data (since we will need to combine 
    # these later in the model, so they all need to have the same technology names)
    df_storage_all['Tech Type'] = (
        df_storage_all['Tech Type']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
        )

    # Drop columns that are not needed for the analysis
    df_storage_all = df_storage_all.drop(columns=['stage', 'unit', 'type', 'fuel']) 

    # combine the processing data for coal, oil, gas into one df and then put it into this 
    df_storage_all = df_storage_all.loc[~df_storage_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    df_storage_all = df_storage_all.loc[df_storage_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    df_storage_all['DIM_1'] = df_storage_all['DIM_1'].astype(int)


    # Convert to 3D array: (Year, Region, Fuel, Type)
    storagecap_xr = (
        df_storage_all
        .set_index(['time', 'DIM_1', 'Tech Type'])
        .to_xarray()['value']
        .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
    )
    #Combine oil crude and oil products into one "oil storage" category 
    oil_storage = storagecap_xr.sel(Type=['Oil Crude', 'Oil Products']).sum(dim='Type')
    storagecap_xr = storagecap_xr.drop_sel(Type=['Oil Crude', 'Oil Products'])
    oil_storage = oil_storage.expand_dims({'Type': ['Oil Storage']})
    storagecap_xr = xr.concat([storagecap_xr, oil_storage], dim='Type')

    #Add units
    storagecap_xr = prism.Q_(storagecap_xr, "meter**3")

    return storagecap_xr


####################################################################################################
#%% Transport/Vehicles stage (coal, oil, gas)
####################################################################################################
def compute_transport_capacity(
        infra: dict[str, dict[str, pd.DataFrame]],
        year_start: int,
        year_end: int,
    ) -> xr.DataArray:
    """Build the transport capacity (stock) of coal, oil and gas as one xarray.
    
    Transport covers shipping, rail and trucks. The stock tables from compute_all_infra() are 
    combined, restricted to the relevant years and regions, and brought into the format the model needs.

    Parameters
    ----------
    infra: Result of compute_all_infra() in drivers.py. Uses
        infra["coal"]["processing"], infra["oil"]["refinery"] and
        infra["gas"]["processing"]. Each table has one column per region (1-28)
        and the index (type, time, fuel, stage, unit); values are the total
        weight of the infrastructure in kg.
    year_start: First year to keep.
    year_end: Last year to keep (included).

    Returns
    -------
    Processing stock in kg, dimensions (Time, Region, Type).

    """
    #Transport capacity (stock demand per generation technology) 
    transport_total_coal = infra["coal"]["transport"]
    transport_total_oil  = infra["oil"]["transport"]
    transport_total_gas  = infra["gas"]["transport"]
    
    transport_total_coal.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )
    transport_total_oil.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )

    transport_total_gas.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )

    df_transport_coal = transport_total_coal.reset_index()
    df_transport_oil = transport_total_oil.reset_index()
    df_transport_gas = transport_total_gas.reset_index()

    df_transport_coal["fuel"] = "coal"
    df_transport_oil["fuel"] = "oil"
    df_transport_gas["fuel"] = "gas"

    #Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long 
    # format so that we have one row per combination of time, region, fuel, technology, and value (stock)
    df_transport_all = pd.concat(
        [df_transport_coal, df_transport_oil, df_transport_gas],
        ignore_index=True
    )

    df_transport_all = df_transport_all.melt(
        id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
        var_name='DIM_1',
        value_name='value'
    )

    #Combine fuel and type columns to one column "Tech Type"
    df_transport_all['Tech Type'] = df_transport_all['fuel'].str.strip() + ' ' + df_transport_all['type'].str.strip()

    # Clean up the "Tech Type" column to have consistent formatting and naming, and to match the 
    # technology names in the lifetime and material intensity data (since we will need to combine 
    # these later in the model, so they all need to have the same technology names)
    df_transport_all['Tech Type'] = (
        df_transport_all['Tech Type']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
        .replace({
            "Coal Inland Ships": "Coal Inland Ship",
            "Coal Ocean Ships": "Coal Ocean Ship",
            "Coal Rail Cargo": "Coal Rail",
            "Coal Trucks": "Coal Truck",
            "Gas Inland Ships": "Gas Inland Ship",
            "Gas Ocean Ships": "Gas Ocean Ship",
            "Gas Rail Cargo": "Gas Rail",
            "Gas Trucks": "Gas Truck",
            "Oil Inland Ships": "Oil Inland Ship",
            "Oil Ocean Ships": "Oil Ocean Ship",
            "Oil Rail Cargo": "Oil Rail",
            "Oil Trucks": "Oil Truck"
        })
    )

    #Drop columns that are not needed for the analysis 
    df_transport_all = df_transport_all.drop(columns=['stage', 'unit', 'type', 'fuel']) 
    # Transport capacity ------
    df_transport_all = df_transport_all.loc[~df_transport_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    df_transport_all = df_transport_all.loc[df_transport_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    df_transport_all['DIM_1'] = df_transport_all['DIM_1'].astype(int)

    transportcap_xr = (
        df_transport_all
        .set_index(['time', 'DIM_1', 'Tech Type'])
        .to_xarray()['value']
        .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
    )
    transportcap_xr = prism.Q_(transportcap_xr, "kg") #Add units

    return transportcap_xr


####################################################################################################
#%% Pipelines stage (oil, gas) ---------------------------------------------------------------------
####################################################################################################
def compute_pipelines_capacity(
        infra: dict[str, dict[str, pd.DataFrame]],
        on_offshore_oil_pipeline: pd.DataFrame,
        year_start: int,
        year_end: int,
    ) -> xr.DataArray:
    """Build the pipelines capacity (stock) of oil and gas as one xarray.

    Gas pipelines come in two types (distribution, transmission). The oil stock is
    given only per product type (crude, product), so it is split into onshore and
    offshore pipelines using fixed global shares.

    Parameters
    ----------
    infra: Result of compute_all_infra() in drivers.py. Uses
        infra["oil"]["pipelines"] (types "crude" and "product") and
        infra["gas"]["pipelines"] (types "distribution" and "transmission").
        Each table has one column per region (1-28) and the index
        (type, time, fuel, stage, unit); values are pipeline lengths in km.
    on_offshore_oil_pipeline: Global share of onshore and offshore pipelines
        (= inputs["drivers"]["on_offshore_oil_pipeline"] from inputs.py).
        Columns: "Crude" and "Product". Index: the division ("offshore",
        "onshore"). Each value is multiplied directly with the oil stock, so the
        values must be fractions (summing to 1 per column), not percentages.
    year_start: First year to keep.
    year_end: Last year to keep (included).

    Returns
    -------
    Pipeline stock in km, dimensions (Time, Region, Type).
        Type holds six pipeline types in a fixed order that matches the lifetime and
        material intensity data: Gas Distribution Pipeline, Gas Transmission Pipeline,
        Oil Offshore Crude Pipeline, Oil Onshore Crude Pipeline,
        Oil Offshore Product Pipeline, Oil Onshore Product Pipeline.

    """
    #Load data from drivers.py
    oil_pipelines  = infra["oil"]["pipelines"]
    gas_pipelines  = infra["gas"]["pipelines"]

    oil_pipelines.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )

    gas_pipelines.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )    

    df_pipelines_oil = oil_pipelines.reset_index()
    df_pipelines_gas = gas_pipelines.reset_index()

    df_pipelines_oil["fuel"] = "oil"
    df_pipelines_gas["fuel"] = "gas"

    #Combine the dataframes for oil and gas into one dataframe and then melt it to long format so 
    # that we have one row per combination of time, region, fuel, technology, and value (stock)
    df_pipelines_all = pd.concat(
        [df_pipelines_oil, df_pipelines_gas],
        ignore_index=True
    )

    df_pipelines_all = df_pipelines_all.melt(
        id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
        var_name='DIM_1',
        value_name='value'
    )

    #Combine fuel and type columns to one column "Tech Type"
    df_pipelines_all['Tech Type'] = df_pipelines_all['fuel'].str.strip() + ' ' + df_pipelines_all['type'].str.strip()

    #Drop columns that are not needed for the analysis
    df_pipelines_all = df_pipelines_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  # drop columns that are not needed for the analysis

    # Clean up the "Tech Type" column to have consistent formatting and naming, and to match the 
    # technology names in the lifetime and material intensity data (since we will need to combine 
    # these later in the model, so they all need to have the same technology names)
    df_pipelines_all['Tech Type'] = (
        df_pipelines_all['Tech Type']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
        .replace({
            "Gas Distribution": "Gas Distribution Pipeline",
            "Gas Transmission": "Gas Transmission Pipeline"
        })
    )

    # Pipeline capacity ------
    df_pipelines_all = df_pipelines_all.loc[~df_pipelines_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    df_pipelines_all = df_pipelines_all.loc[df_pipelines_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    df_pipelines_all['DIM_1'] = df_pipelines_all['DIM_1'].astype(int)

    # Convert to 3D array: (Year, Region, Fuel, Type)
    pipelinecap_xr = (
        df_pipelines_all
        .set_index(['time', 'DIM_1', 'Tech Type'])
        .to_xarray()['value']
        .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
    )

    pipelinecap_xr = pipelinecap_xr.assign_coords(
        Type=np.array(pipelinecap_xr.Type.values, dtype=object)
    )

    # Since the oil pipelines in the data are not split into onshore and offshore pipelines, we will 
    # split them based on the fractions of onshore and offshore pipelines in the global pipeline network f

    # First we identify the types of oil pipelines in the capacity data (oil crude and oil products) 
    oil_types = ['Oil Crude', 'Oil Product']  

    # Then we create new entries for onshore and offshore oil pipelines based on the fractions in 
    # the global pipeline network
    expanded_oil_arrays = []
    for oil_type in oil_types:
        key = oil_type.split()[1] 
        for division, fraction in on_offshore_oil_pipeline[key].items():
            expanded_name = f"Oil {division.capitalize()} {key} Pipeline"
            new_arr = pipelinecap_xr.sel(Type=oil_type) * fraction
            new_arr = new_arr.expand_dims(Type=[expanded_name])
            expanded_oil_arrays.append(new_arr)

    # Add new oil pipeline types to the original xarray by concatenating along the 'Type' dimension
    pipelinecap_xr = xr.concat([pipelinecap_xr] + expanded_oil_arrays, dim='Type')

    # Drop original oil pipeline types since they have now been split into onshore and offshore pipelines
    pipelinecap_xr = pipelinecap_xr.drop_sel(Type=oil_types)

    # Add units
    pipelinecap_xr = prism.Q_(pipelinecap_xr, "km")

    #Reorder technology coordinate in all three datasets (capacity, lifetime, material intensities) 
    # to match the same order of technologies, so that we can easily combine the data in the model 
    # (since they all have to be in the same order of technologies)
    pipelinecap_xr = pipelinecap_xr.reindex(Type=PIPELINE_TYPES)

    return pipelinecap_xr