
from imagematerials.rest_of.const import REGION_TO_CLASS_DICT_IMAGE_MAT_NR
import prism 

def cement_via_concrete_in_sectors(model_name):
    cement_in_concrete = 0.1

    # conrete in residential
    inflow_buildings = model_name.buildings.get('inflow_materials').to_array().sel(material = "Concrete").sum(["Type"]).loc[1961:]*cement_in_concrete
    # concrete in electricity
    inflow_el_grid = model_name.grid.get('inflow_materials').to_array().sel(material = "Concrete").sum(["Type"]).loc[1961:]*cement_in_concrete
    inflow_el_gen = model_name.generation.get('inflow_materials').to_array().sel(material = "Concrete").sum(["Type"]).loc[1961:]*cement_in_concrete

    # replace region names to numbers in the 'Region' coordinate
    inflow_el_grid = inflow_el_grid.assign_coords(
        Region = inflow_el_grid.coords["Region"].to_series().map(REGION_TO_CLASS_DICT_IMAGE_MAT_NR)
    )

    inflow_el_gen = inflow_el_gen.assign_coords(
        Region = inflow_el_gen.coords["Region"].to_series().map(REGION_TO_CLASS_DICT_IMAGE_MAT_NR)
    )

    return inflow_buildings, inflow_el_grid, inflow_el_gen

def sum_inflows_for_output(model_name, materials_dict, resource_group, save = True):
    sand_in_cement_conversion = 0.17 #(silica)
    sand_gravel_in_concrete_conversion = 0.7
    sand_in_glass_conversion = 0.7

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

        if key == 'Copper':
            copper_buildings = model_name.buildings.get('inflow_materials').to_array().sum(['Type']).sel(material='Copper').loc[1971:]
            copper_vehicles = model_name.vehicles.get('inflow_materials').to_array().sum(['Type']).sel(material='Cu').loc[1971:]
            copper_generation = model_name.generation.get('inflow_materials').to_array().sum(['Type']).sel(material='Cu').loc[1971:]
            copper_grid = model_name.grid.get('inflow_materials').to_array().sum(['Type']).sel(material='Cu').loc[1971:]

            copper_grid = copper_grid.assign_coords(
                Region = copper_grid.coords["Region"].to_series().map(REGION_TO_CLASS_DICT_IMAGE_MAT_NR)
            )
            copper_generation = copper_generation.assign_coords(
                Region = copper_generation.coords["Region"].to_series().map(REGION_TO_CLASS_DICT_IMAGE_MAT_NR)
            )   
            total_material = copper_buildings + copper_vehicles + copper_generation + copper_grid

        elif key == 'Cement':
            # cement
            inflow_buildings_cement = model_name.buildings.get('inflow_materials').to_array().sum(['Type']).sel(material=key).loc[1971:]
            # concrete
            inflow_buildings_concrete, inflow_el_grid_concrete, inflow_el_gen_concrete = cement_via_concrete_in_sectors(model_name)

            total_material = inflow_buildings_cement + inflow_buildings_concrete + inflow_el_grid_concrete + inflow_el_gen_concrete

        elif key == 'Sand':
            # sand in cement
            inflow_buildings_cement = model_name.buildings.get('inflow_materials').to_array().sum(['Type']).sel(material='Cement').loc[1971:]*sand_in_cement_conversion
            # sand in concrete
            inflow_buildings_concrete, inflow_el_grid_concrete, inflow_el_gen_concrete = cement_via_concrete_in_sectors(model_name)
            inflow_buildings_concrete = inflow_buildings_concrete * sand_gravel_in_concrete_conversion
            inflow_el_grid_concrete = inflow_el_grid_concrete * sand_gravel_in_concrete_conversion
            inflow_el_gen_concrete = inflow_el_gen_concrete * sand_gravel_in_concrete_conversion


            # sand in glass
            inflow_vehicles_sand_glass = model_name.vehicles.get('inflow_materials').to_array().sel(material = "Glass").sum(["Type"]).loc[1971:]*sand_in_glass_conversion
            inflow_buildings_sand_glass = model_name.buildings.get('inflow_materials').to_array().sel(material = "Glass").sum(["Type"]).loc[1971:]*sand_in_glass_conversion
            inflow_el_grid_sand_glass = model_name.grid.get('inflow_materials').to_array().sel(material = "Glass").sum(["Type"]).loc[1971:]*sand_in_glass_conversion
            inflow_el_gen_sand_glass = model_name.generation.get('inflow_materials').to_array().sel(material = "Glass").sum(["Type"]).loc[1971:]*sand_in_glass_conversion

            inflow_el_grid_sand_glass = inflow_el_grid_sand_glass.assign_coords(
                Region = inflow_el_grid_sand_glass.coords["Region"].to_series().map(REGION_TO_CLASS_DICT_IMAGE_MAT_NR)
            )
            inflow_el_gen_sand_glass = inflow_el_gen_sand_glass.assign_coords(
                Region = inflow_el_gen_sand_glass.coords["Region"].to_series().map(REGION_TO_CLASS_DICT_IMAGE_MAT_NR)
            )

            # add up all sand
            total_material = (inflow_buildings_cement + inflow_buildings_concrete + inflow_el_grid_concrete + inflow_el_gen_concrete +
                              inflow_vehicles_sand_glass + inflow_buildings_sand_glass + inflow_el_grid_sand_glass + inflow_el_gen_sand_glass)
            
        else:
            inflow_buildings = model_name.buildings.get('inflow_materials').to_array().sum(['Type']).sel(material=key).loc[1971:]
            inflow_vehicles = model_name.vehicles.get('inflow_materials').to_array().sum(['Type']).sel(material=value).loc[1971:]
            inflow_electricity_genereation = inflow_materials_generation.sum(['Type']).sel(material=value).loc[1971:]
            inflow_electricity_grid = inflow_materials_grid.sum(['Type']).sel(material=value).loc[1971:]
            total_material = inflow_buildings + inflow_vehicles + inflow_electricity_genereation + inflow_electricity_grid

        # from total_material create a csv that has the years as rows and regions as columns, mae sure that region names are no just '1' but 'class_ 1'
        # also drop material dimension
        total_material = total_material.drop_vars('material')
        # change the region coordinate so that it is class_ 1 instead of 1 , ...
        # Get the current region values
        regions = total_material.coords['Region'].values

        # Create new region names
        new_regions = [f'class_ {r}' for r in regions]

        # Assign the new region names to the coordinate
        total_material = total_material.assign_coords(Region=new_regions)
        # to t
        # check if unit available
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

    return total_material_dict, 

