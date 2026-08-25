from pathlib import Path
import pym
import prism
import xarray as xr

from imagematerials.read_mym import mymarray_to_xarray

from imagematerials.appliances.constants import energy_use_per_appliance


def calculate_stocks_applainces(image_directory: Path,
                                  energy_use_per_appliance=energy_use_per_appliance):
    """Calculate regional appliance stocks (counts) from IMAGE model output.

    Combines appliance diffusion, end-use energy, and household data: stocks for
    most appliances follow directly from diffusion (count/household) times
    households, while cooling appliance (Fan, Air Cooler, Air Conditioner) stocks
    are derived by dividing their end-use energy by `energy_use_per_appliance`,
    since diffusion for those is only available as a household share.

    Parameters
    ----------
    image_directory : Path
        Path to the IMAGE output directory containing the EnergyServices and
        Socioeconomic subfolders.
    energy_use_per_appliance : xr.DataArray
        Energy use per cooling appliance, used to convert cooling end-use energy
        into an appliance count. Defaults to `energy_use_per_appliance` from
        `imagematerials.appliances.constants`.

    Returns
    -------
    xr.DataArray
        Appliance stock counts with dims (time, Region, Type), covering all
        appliance types across all regions 
    """

    appliances_diffusion = mymarray_to_xarray(pym.read_mym('EnergyServices/Appl_diffusion.out',image_directory),
                                        dim_1="NRT", dim_2="TURQ", dim_3="Appliances")

    appliances_end_use_energy = mymarray_to_xarray(pym.read_mym('EnergyServices/ApplEnUse.out',image_directory),
                                        dim_1="NRT", dim_2="TURQ", dim_3="AppliancesOther")
    appliances_end_use_energy = prism.Q_(appliances_end_use_energy, "GJ")

    households = mymarray_to_xarray(pym.read_mym('Socioeconomic/Households.out',image_directory),
                                        dim_1="NRT", dim_2="TURQ")
    households = prism.Q_(households, "household")

    # split up xarray in cooling and other applainces, as unit is different
    # unit for cooling is share of households that have this appliances, while for other is number of appliances per household
    cooling_appliances = appliances_diffusion.sel(Appliances=['Fan', 'Air Cooler', 'Air Conditioner'])
    # unit: share of households with appliance
    cooling_appliances = prism.Q_(cooling_appliances, "share/household")

    # Other appliances are already in number of appliances per household, so we just need to select these and add the unit 
    other_appliances = appliances_diffusion.sel(Appliances=['Refrigerator', 'Microwave',
                                                            'Washing Machine', 'Clothes Dryer', 'Dish Washer', 'TV', 'VCR/DVD',
                                                            'PC/Other'])
    other_appliances = prism.Q_(other_appliances, "count/household")

    # drop empty other appliances
    appliances_end_use_energy = appliances_end_use_energy.drop_sel(AppliancesOther="other")
    # rename dimension name so that it matches the appliances diffusion data
    appliances_end_use_energy = appliances_end_use_energy.rename({"AppliancesOther": "Appliances"})

    # Get count of appliances 
    count_appliances = other_appliances * households

    # Get count of cooling appliances via energy per appliance and total energy use
    cooling_end_use_energy = appliances_end_use_energy.sel(Appliances=['Fan', 'Air Cooler', 'Air Conditioner'])

    # Count of cooling appliances is calculated by dividing the end use energy by the energy use per appliance
    count_cooling_appliances = cooling_end_use_energy / energy_use_per_appliance

    # combine the two counts into one xarray
    # THIS IS STOCKS
    count_appliances = xr.concat([count_cooling_appliances, count_appliances], dim="Appliances")
    # rename to fit stock modelling dimensions
    count_appliances = count_appliances.sel(TURQ="total", drop=True).rename({"t": "time", "NRT": "Region", "Appliances": "Type"})
    count_appliances = count_appliances.drop_sel(Region="WORLD")  # drop world, as we only want to have the regions

    return count_appliances