import xarray as xr

from imagematerials.electricity.utils import (
    apply_ce_measures_to_elc,
)
from imagematerials.util import flag_enabled


def apply_ce_measures_to_elc_generation(materials_array: xr.DataArray,
                                        lifetime_array: xr.DataArray,
                                        circular_economy_config: dict,
                                        resource_efficiency_flags: dict = None):
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

    if flag_enabled(resource_efficiency_flags, "electricity", "FlagLightweightingGeneration"):
        flag_config          = circular_economy_config['electricity']['FlagLightweightingGeneration']
        target_year          = flag_config['target_year']
        base_year            = flag_config['base_year']
        implementation_rate  = flag_config['implementation_rate']
        weight_change_pc     = flag_config['weight_change_pc']

        materials_array = apply_ce_measures_to_elc(
            materials_array,
            base_year           = base_year,
            target_year         = target_year,
            change              = weight_change_pc,
            implementation_rate = implementation_rate
        )
        print("FlagLightweightingGeneration applied to ", materials_array.name)

    if flag_enabled(resource_efficiency_flags, "electricity", "FlagLifetimeExtensionGeneration"):
        flag_config          = circular_economy_config['electricity']['FlagLifetimeExtensionGeneration']
        target_year          = flag_config['target_year']
        base_year            = flag_config['base_year']
        implementation_rate  = flag_config['implementation_rate']
        lifetime_change_pc   = flag_config['lifetime_increase_percent']

        lifetime_array = apply_ce_measures_to_elc(
            lifetime_array,
            base_year           = base_year,
            target_year         = target_year,
            change              = lifetime_change_pc,
            implementation_rate = implementation_rate,
            data_type           = "lifetime"
        )
        print("FlagLifetimeExtensionGeneration applied to ", lifetime_array.name)

    return materials_array, lifetime_array



def apply_ce_measures_to_elc_grid(materials_additions_array: xr.DataArray,
                                lifetime_array: xr.DataArray,
                                circular_economy_config: dict,
                                resource_efficiency_flags: dict = None):
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
    if flag_enabled(resource_efficiency_flags, "electricity", "FlagLightweightingGrid"):
        flag_config          = circular_economy_config['electricity']['FlagLightweightingGrid']
        target_year          = flag_config['target_year']
        base_year            = flag_config['base_year']
        implementation_rate  = flag_config['implementation_rate']
        weight_change_pc     = flag_config['weight_change_pc']

        materials_additions_array = apply_ce_measures_to_elc(
            materials_additions_array,
            base_year=base_year,
            target_year=target_year,
            change=weight_change_pc,
            implementation_rate=implementation_rate,
            data_sector = "electricity grid"
        )
        print("FlagLightweightingGrid applied to ", materials_additions_array.name)


    if flag_enabled(resource_efficiency_flags, "electricity", "FlagLifetimeExtensionGrid"):
        flag_config          = circular_economy_config['electricity']['FlagLifetimeExtensionGrid']
        target_year          = flag_config['target_year']
        base_year            = flag_config['base_year']
        implementation_rate  = flag_config['implementation_rate']
        lifetime_change_pc   = flag_config['lifetime_increase_percent']

        lifetime_array = apply_ce_measures_to_elc(
            lifetime_array,
            base_year           = base_year,
            target_year         = target_year,
            change              = lifetime_change_pc,
            implementation_rate = implementation_rate,
            data_sector         = "electricity grid",
            data_type           = "lifetime"
        )
        print("FlagLifetimeExtensionGrid applied to ", lifetime_array.name)

    return materials_additions_array, lifetime_array