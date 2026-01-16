import xarray as xr
import pandas as pd

from imagematerials.util import dataset_to_array
from imagematerials.vehicles.constants import (
    maintenance_lifetime_per_mode
)
from imagematerials.vehicles.modelling_functions import interpolate
from imagematerials.vehicles.preprocessing.util import xarray_conversion


def get_maintenance_materials(standard_data_path):
    maintenance_material_pd : pd.DataFrame = pd.read_csv(
        standard_data_path.joinpath("all_vehicle_maintenance_image.csv"),
        index_col=0
    )

    # Calculate maintenace material need in kg material per kg vehicle
    maintenance_material_pd['lithium'] = 0
    maintenance_material_pd['manganese'] = 0
    maintenance_material_pd['nickel'] = 0
    maintenance_material_pd['titanium'] = 0

    stacked_maintenance_material = maintenance_material_pd.set_index("Type").stack().rename_axis(
        index=["Type", "material"]
    ).reset_index(name="value")

    stacked_maintenance_material = stacked_maintenance_material.set_index(["Type", "material"])

    stacked_maintenance_material_xr = stacked_maintenance_material.to_xarray()
    maintenance_material = dataset_to_array(stacked_maintenance_material_xr, ["Type", "material"], [])

    modes = list(maintenance_material.coords['Type'].values)
    expected_lifetimes = xr.DataArray(
        data=[maintenance_lifetime_per_mode[mode] for mode in modes],
        dims=["Type"],
        coords={"Type": modes},
        name="vehicle_lifetime"
    )
    maintenance_material_fractions = (maintenance_material / expected_lifetimes)

    xr_default_maintenance = xr.DataArray(
        0.0,
        dims=("Type", "material"),
        coords={
            "Type": ["Vehicles"],
            "material": maintenance_material_fractions.coords["material"]
        }
    )
    return xr.concat((maintenance_material_fractions, xr_default_maintenance), dim="Type")


def get_material_fractions(data_path):
    # Material fractions in percentages
    material_fractions: pd.DataFrame = pd.read_csv(
        data_path.joinpath("material_fractions_simple.csv"),
        index_col=[0, 1])

    # Material fractions in percentages, by vehicle sub-type
    material_fractions_type: pd.DataFrame = pd.read_csv(
        data_path.joinpath("material_fractions_typical.csv"),
        index_col=[0, 1],
        header=[0, 1]
    )

    # For dynamic variables, apply interpolation and extend over the
    # whole timeframe
    # complete & interpolate the vehicle composition data (simple first)
    material_fractions_simple = material_fractions.rename_axis(
        "mode", axis=1
    ).rename_axis(
        ["year", "material"], axis=0
    ).stack().unstack(["mode", "material"])
    material_fractions_simple = interpolate(pd.DataFrame(material_fractions_simple))

    # complete & interpolate the vehicle composition data (by vehicle
    # sub-type second)
    material_fractions_typical = material_fractions_type.rename_axis(
        ["mode", "type"], axis=1
    ).rename_axis(
        ["year", "material"], axis=0
    ).stack().stack().unstack(["mode", "type", "material"])
    material_fractions_typical = interpolate(pd.DataFrame(material_fractions_typical))

    material_fractions_simple = xarray_conversion(
        material_fractions_simple,
        (["Cohort"], ["Type", "material"],)
    )
    material_fractions_typical = xarray_conversion(
        material_fractions_typical,
        (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
    )
    return xr.concat(
        (material_fractions_simple, material_fractions_typical), dim="Type"
    )

    
