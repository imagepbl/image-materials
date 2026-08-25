# Appliances preprocessing

from imagematerials.appliances.preprocessing.stocks import calculate_stocks_applainces
from imagematerials.appliances.preprocessing.materials import define_material_intensities
from imagematerials.appliances.preprocessing.lifetimes import define_lifetimes


def appliances_preprocessing(image_directory):

    stock_appliances = calculate_stocks_applainces(image_directory = image_directory)
    material_intensities_appliances = define_material_intensities(stock_appliances)
    lifetimes = define_lifetimes(stock_appliances, stock_appliances.Type.values)

    return {"stocks": stock_appliances, 
            "lifetimes": lifetimes, 
            "material_intensities": material_intensities_appliances,
            "knowledge_graph": None, # not needed but needs to be passed
            "set_unit_flexible": str(stock_appliances.pint.units)}