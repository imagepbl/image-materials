import xarray as xr

from imagematerials.electricity.utils import (
    apply_ce_measures_to_elc, 
)




def apply_ce_measures_to_elc_generation(materials_array: xr.DataArray,
                                            lifetime_array: xr.DataArray,
                                            circular_economy_config: dict):
    """Implement circular economy measures for electricity generation.

    Parameters
    ----------
    materials_array:
        material intensities for electricity generation.
    lifetime_array:
        lifetimes for electricity generation.
    circular_economy_config:
        Configuration of the circular economy.

    Returns
    -------
    materials_array:
        Updated material intensities for electricity generation.
    lifetime_array:
        Updated lifetimes for electricity generation.

    """

    if "narrow_product" in circular_economy_config.keys():
        ce_scen = "narrow_product"
        target_year          = circular_economy_config[ce_scen]['electricity']['target_year']
        base_year            = circular_economy_config[ce_scen]['electricity']['base_year']
        implementation_rate  = circular_economy_config[ce_scen]['electricity']['implementation_rate']
        weight_change_pc = circular_economy_config[ce_scen]['electricity']['generation']['weight_change_pc']

        materials_array = apply_ce_measures_to_elc(
            materials_array,
            base_year           = base_year,
            target_year         = target_year,
            change              = weight_change_pc,
            implementation_rate = implementation_rate
        )
        print("narrow|lightweighting applied to ", materials_array.name)

    if "slow" in circular_economy_config.keys():
        ce_scen = "slow"
        target_year          = circular_economy_config[ce_scen]['electricity']['target_year']
        base_year            = circular_economy_config[ce_scen]['electricity']['base_year']
        implementation_rate  = circular_economy_config[ce_scen]['electricity']['implementation_rate']
        lifetime_change_pc = circular_economy_config[ce_scen]['electricity']['generation']['lifetime_increase_percent']

        lifetime_array = apply_ce_measures_to_elc(
            lifetime_array,
            base_year           = base_year,
            target_year         = target_year,
            change              = lifetime_change_pc,
            implementation_rate = implementation_rate,
            data_type           = "lifetime"
        )
        print("slow|lifetime increase applied to ", lifetime_array.name)

    return materials_array, lifetime_array



def apply_ce_measures_to_elc_grid(materials_additions_array: xr.DataArray,
                                lifetime_array: xr.DataArray,
                                circular_economy_config: dict):
    """Implement circular economy measures for electricity generation.

    Parameters
    ----------
    materials_additions_array:
        material intensities for electricity grid additions (transformer, substations).
    lifetime_array:
        lifetimes for electricity grids.
    circular_economy_config:
        Configuration of the circular economy.

    Returns
    -------
    materials_additions_array:
        Updated material intensities for electricity grid additions.
    lifetime_array:
        Updated lifetimes for electricity grids.

    """
    if "narrow_product" in circular_economy_config.keys():
        ce_scen = "narrow_product"

        target_year         = circular_economy_config[ce_scen]['electricity']['target_year']
        base_year           = circular_economy_config[ce_scen]['electricity']['base_year']
        implementation_rate = circular_economy_config[ce_scen]['electricity']['implementation_rate']

        weight_change_pc = circular_economy_config[ce_scen]['electricity']['grid_add']['weight_change_pc']

        materials_additions_array = apply_ce_measures_to_elc(
            materials_additions_array,
            base_year=base_year,
            target_year=target_year,
            change=weight_change_pc,
            implementation_rate=implementation_rate,
            data_sector = "electricity grid"
        )
        print("narrow|lightweighting applied to ", materials_additions_array.name)


    if "slow" in circular_economy_config.keys():
        ce_scen = "slow"
        target_year          = circular_economy_config[ce_scen]['electricity']['target_year']
        base_year            = circular_economy_config[ce_scen]['electricity']['base_year']
        implementation_rate  = circular_economy_config[ce_scen]['electricity']['implementation_rate']
        lifetime_change_pc = circular_economy_config[ce_scen]['electricity']['grid_add']['lifetime_increase_percent']

        lifetime_array = apply_ce_measures_to_elc(
            lifetime_array,
            base_year           = base_year,
            target_year         = target_year,
            change              = lifetime_change_pc,
            implementation_rate = implementation_rate,
            data_sector         = "electricity grid",
            data_type           = "lifetime"
        )
        print("slow|lifetime increase applied to ", lifetime_array.name)

    return materials_additions_array, lifetime_array