from imagematerials.concepts import KnowledgeGraph, Node
from imagematerials.constants import _IMAGE_REGIONS as image_regions
import pandas as pd
from constants import target_regions, region_mapping
import xarray as xr

def convert_to_xarray(df, value_name, variable):
    df = df.rename(columns={"Unnamed: 0":"year","Unnamed: 1":"region"})
    df = (
        df.astype({"year": int, "region": int})  # Convert to integers
        .query("region not in [27, 28] and year in [2019, 2060]")  # Filter regions and years
    )
    # Convert all other columns (modes) to float
    df.iloc[:, 2:] = df.iloc[:, 2:].astype(float)

    region_mapping = { i+1:region for i, region in enumerate(image_regions)}
    df['region'] = df['region'].map(region_mapping)

    # Melt the DataFrame so that transport modes become a single coordinate
    df = df.melt(id_vars=["year", "region"], var_name= variable, value_name=value_name)

    # Convert to xarray DataArray
    dataarray = df.set_index(["region", "year", "mode"]).to_xarray()[value_name]
    return dataarray

def create_region_graph():
    target_regions= ["Canada","Central Europe","China","India","Japan",
                     "USA","Western Europe","Latin America","Middle East and Northern Africa",
                    "Other Asia","Other OECD","Reforming Economies","Subsaharan Africa"]
    region_knowledge_graph = KnowledgeGraph()
        # Add target regions as main nodes
    for region in target_regions:
        region_knowledge_graph.add(Node(region))
    
    # Assign image regions stepwise to their respective target regions

    for region in ["RCAM", "BRA", "RSAM"]:
        region_knowledge_graph.add(Node(region, inherits_from="Latin America"))
    for region in ["NAF", "ME"]:
        region_knowledge_graph.add(Node(region, inherits_from="Middle East and Northern Africa"))
    for region in ["WAF", "EAF", "SAF", "RSAF"]:
        region_knowledge_graph.add(Node(region, inherits_from="Subsaharan Africa"))
    for region in [ "UKR", "STAN", "RUS"]:
        region_knowledge_graph.add(Node(region, inherits_from="Reforming Economies"))
    for region in ["KOR", "SEAS", "INDO", "RSAS"]:
        region_knowledge_graph.add(Node(region, inherits_from="Other Asia"))
    for region in ["TUR", "OCE", "MEX"]:
        region_knowledge_graph.add(Node(region, inherits_from="Other OECD"))
    region_knowledge_graph["Canada"].synonyms = ["CAN"]
    region_knowledge_graph["USA"].synonyms = ["USA"]
    region_knowledge_graph["Western Europe"].synonyms = ["WEU"]
    region_knowledge_graph["Central Europe"].synonyms = ["CEU"]
    region_knowledge_graph["India"].synonyms = ["INDIA"]
    region_knowledge_graph["China"].synonyms = ["CHN"]
    region_knowledge_graph["Japan"].synonyms = ["JAP"]
    
    return region_knowledge_graph

def verify_image_regions(region_knowledge_graph):
    image_regions = [
        'CAN', 'USA', 'MEX', 'RCAM', 'BRA', 'RSAM', 'NAF', 'WAF', 'EAF',
        'SAF', 'WEU', 'CEU', 'TUR', 'UKR', 'STAN', 'RUS', 'ME', 'INDIA',
        'KOR', 'CHN', 'SEAS', 'INDO', 'JAP', 'OCE', 'RSAS', 'RSAF'
    ]
    
    missing_regions = [region for region in image_regions if region not in region_knowledge_graph]
    
    if missing_regions:
        print("Missing regions in knowledge graph:", missing_regions)
    else:
        print("All image regions are included in the knowledge graph!")

def aggregate_regions(data_array, region_mapping, aggregation='sum'):
    data_array = data_array.assign_coords(region=[region_mapping[r] for r in data_array["region"].values])

    # Aggregate by summing over the mapped regions
    if aggregation == 'sum':
        return data_array.groupby("region").sum()
    elif aggregation == 'mean':
        return data_array.groupby("region").mean()

def extract_population():
    # not used atm
    image_output = pd.read_csv("IMAGE scenario explorer variables.csv", delimiter=";", index_col=0, usecols=range(8))
    image_output = image_output.drop(columns=["model", "scenario","unit"])

    # Step 2: Convert DataFrame into xarray Dataset
    xr_image_output = image_output.set_index(["variable", "region", "year"]).to_xarray()

    xr_image_output = xr_image_output.sel(year=[2020, 2060])

    xr_image_output = xr_image_output.to_dataarray("var2").drop_vars("var2").squeeze()

    xr_regions = xr_image_output.coords["region"].values  # Extract current region names

    # Create a mapping of only existing regions
    region_mapping_filtered = {key: val for key, val in region_mapping.items() if key in xr_regions}

    # Map the dataset’s regions to the target regions
    xr_image_output_filtered = xr_image_output.sel(region=list(region_mapping_filtered.keys()))
    xr_image_output_filtered = xr_image_output_filtered.assign_coords(region=[region_mapping_filtered[r] for r in xr_image_output_filtered["region"].values])

    # Aggregate by summing over the mapped regions
    xr_image_output_aggregated = xr_image_output_filtered.groupby("region").sum()

    pop = xr_image_output_aggregated.sel(variable="Population")  # Extract Population data
    # Update the year coordinate from 2020 to 2019
    #new_years = pop.coords["year"].values.copy()  # Copy the years array
    #new_years[new_years == 2020] = 2019  # Replace 2020 with 2019

    # Assign the modified years back to the coordinate
    #pop.coords["year"] = ("year", new_years)

    # Create a new DataArray for 2019 by copying 2020 data
    pop_2019 = pop.sel(year=2020).assign_coords(year=2019)

    # Concatenate the new 2019 data along the year dimension
    pop = xr.concat([pop_2019, pop], dim="year").sortby("year")
    return pop




