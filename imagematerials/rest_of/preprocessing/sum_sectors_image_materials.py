
from imagematerials.rest_of.const import REGION_TO_CLASS_DICT_IMAGE_MAT_NR

def sum_inflows_for_output(model_name, materials_dict, resource_group, save = True):
    cement_in_concrete = 0.1
    sand_in_cement_conversion = 0.17 #(silica)
    sand_gravel_in_concrete_conversion = 0.7
    sand_in_glass_conversion = 0.7

    only_buildings = ['Cement', 'Concrete']
    only_vehicles = ['Glass']
    not_in_any = ['Sand']
    total_material_dict = {}

    # regions electricity and generation
    inflow_materials_grid = model_name.grid["inflow_materials"].to_array()
    inflow_materials_generation = model_name.generation["inflow_materials"].to_array()

     # replace region names to numbers
    # replace region names to numbers in the 'Region' coordinate
    inflow_materials_grid = inflow_materials_grid.assign_coords(
        Region = inflow_materials_grid.coords["Region"].to_series().map(REGION_TO_CLASS_DICT_IMAGE_MAT_NR)
    )
    inflow_materials_generation = inflow_materials_generation.assign_coords(
        Region = inflow_materials_generation.coords["Region"].to_series().map(REGION_TO_CLASS_DICT_IMAGE_MAT_NR)
    )

    for key, value in materials_dict.items():
        print(key)
        if key not in only_buildings and key not in only_vehicles and not key in not_in_any:
            inflow_buildings = model_name.buildings.get('inflow_materials').to_array().sum(['Type']).sel(material=key).loc[1961:]
            inflow_vehicles = model_name.vehicles.get('inflow_materials').to_array().sum(['Type']).sel(material=value).loc[1961:]
            inflow_electricity_genereation = inflow_materials_generation.sum(['Type']).sel(material=value).loc[1961:]
            inflow_electricity_grid = inflow_materials_grid.sum(['Type']).sel(material=value).loc[1961:]
            total_material = inflow_electricity_genereation + inflow_electricity_grid + inflow_buildings + inflow_vehicles

        if key == 'Cement':
            # add concrete to cement
            inflow_buildings_cement = model_name.buildings.get('inflow_materials').to_array().sum(['Type']).sel(material=key).loc[1961:]
            inflow_buildings_concrete = model_name.buildings.get('inflow_materials').to_array().sum(['Type']).sel(material='Concrete').loc[1961:] * cement_in_concrete
            inflow_electricity_genereation =inflow_materials_generation.sum(['Type']).sel(material='Concrete').loc[1961:] * cement_in_concrete
            inflow_electricity_grid = inflow_materials_grid.sum(['Type']).sel(material='Concrete').loc[1961:] * cement_in_concrete

            total_material = inflow_electricity_genereation + inflow_electricity_grid + inflow_buildings_cement + inflow_buildings_concrete

        if key == 'Sand':
            inflow_buildings_cement_sand = model_name.buildings.get('inflow_materials').to_array().sum(['Type']).sel(material='Cement').loc[1961:]*sand_in_cement_conversion
            inflow_buildings_concrete_sand_via_cement = model_name.buildings.get('inflow_materials').to_array().sum(['Type']).sel(material='Concrete').loc[1961:] * cement_in_concrete * sand_in_cement_conversion
            inflow_buildings_sand_in_concrete = model_name.buildings.get('inflow_materials').to_array().sum(['Type']).sel(material='Concrete').loc[1961:] * sand_gravel_in_concrete_conversion
            inflow_vehicles_sand = model_name.vehicles.get('inflow_materials').to_array().sum(['Type']).sel(material='Glass').loc[1961:] * sand_in_glass_conversion

            inflow_electricity_genereation_sand = inflow_materials_generation.sum(['Type']).sel(material='Glass').loc[1961:] * sand_in_glass_conversion
            inflow_electricity_grid_sand = inflow_materials_grid.sum(['Type']).sel(material='Glass').loc[1961:] * sand_in_glass_conversion

            inflow_electricity_genereation_concrete_sand_via_cement = inflow_materials_generation.sum(['Type']).sel(material='Concrete').loc[1961:] * cement_in_concrete * sand_in_cement_conversion
            inflow_electricity_grid_concrete_sand_via_cement = inflow_materials_grid.sum(['Type']).sel(material='Concrete').loc[1961:] * cement_in_concrete * sand_in_cement_conversion

            total_material = (inflow_buildings_cement_sand + inflow_buildings_concrete_sand_via_cement + inflow_buildings_sand_in_concrete + 
                              inflow_vehicles_sand + inflow_electricity_genereation_sand + inflow_electricity_grid_sand + 
                              inflow_electricity_genereation_concrete_sand_via_cement + inflow_electricity_grid_concrete_sand_via_cement)

        # from total_material create a csv that has the years as rows and regions as columns, mae sure that region names are no just '1' but 'class_ 1'
        # also drop material dimension
        if key not in ['Copper', 'Cement', 'Sand']:
            total_material = total_material.drop_vars('material')
        # change the region coordinate so that it is class_ 1 instead of 1 , ...
        # Get the current region values
        regions = total_material.coords['Region'].values

        # Create new region names
        new_regions = [f'class_ {r}' for r in regions]

        # Assign the new region names to the coordinate
        total_material = total_material.assign_coords(Region=new_regions)
        # to t
        total_material = total_material.pint.to('t')
        # save as pandas to save as csv
        total_material = total_material.rename("total_material")
        # write key with a small letter
        key = key.lower()
        # to pandas
        total_material = total_material.to_dataframe().unstack()
        # drop unessecary column level index
        total_material.columns = total_material.columns.droplevel(0)
        # save as csv
        if key == 'sand':
            key = 'sand_gravel_crushed_rock'
            total_material = total_material.loc[1971:]
        else: 
            pass
        if save == True:
            total_material.to_csv(f'../data/raw/rest-of/{resource_group}/image_materials_{key}.csv')
            print('done', key)

        total_material_dict[key] = total_material

    return total_material_dict

