==================
End-of-Life
==================


The end-of-life (eol) module handles the collection, reuse, recycling, and disposal 
of outflow materials from products that reach their end-of-life within the IMAGE-materials framework.
It tracks how materials from retired products are managed across different sectors and applies scenario-based 
assumptions for future years.

Scope
-----------


The eol module covers:

- **Bulk materials**: Steel, concrete, wood, copper, aluminium, glass, brick, plastics and rubber.
- **Minor and technology metals**: cobalt, lead, lithium, manganese, neodymium, nickel, tantalum, titanium
- **Sectors**: Buildings (urban, rural, commercial), Vehicles (passenger, freight), and Electricity 
  (generation, grid, storage)
- **Processes**: Collection rates, reuse rates, and recycling rates across materials and sectors
- **Regions**: All 26 IMAGE regions
- **Time horizon**: 1920-2100 with interpolation between anchor year (2020) and target year (2060)

It does not cover:

- Energy recovery processes from incineration (handled in the energy module)
- Waste management infrastructure, processes, energy use, and emissions

************
Data Input
************

**Input Data Location**: ``data/raw/end_of_life/``

**Input Files** (CSV format):

- **collection.csv**: Collection rates, which are the share of end-of-life material outflow that is collected for further processing. Range between 0-1 for each material, sector, region, and year.

- **reuse.csv**: Reuse rates, which are the proportion of collected materials that are reused directly within the same sector or application. Range between 0-1 for each material, sector, region, and year.

- **recycling.csv**: Recycling rates, which are the proportion of collected, non-reusable materials that are processed into secondary materials. Range between 0-1 for each material, sector, region, and year.

Preprocessing Structure
----------------------

The preprocessing function - :py:func:`~imagematerials.eol.preprocessing.eol_preprocessing` - performs the following steps:

1. **Data loading and preparation**: Read collection, reuse, and recycling CSV files, and ensuring consistent formatting and region, material, and sector nomenclature.
   
2. **Convert to xarray**: Create xarray DataArrays with dimensions (Time, Region, Type, material)

3. **Apply scenario assumptions**: Implement optional scenario modifications:

   These scenarios apply different rates for buildings, vehicles, and electricity sectors by 2060.

4. **Interpolation and Extrapolation**: Use :py:func:`~imagematerials.eol.preprocessing.interpolate_eol_rates` to:
   
   - Hold 2020 values constant before 2020
   - Linearly interpolate between 2020 and 2060
   - Hold 2060 values constant after 2060
   - Clip all rates to valid range [0, 1]

Output Data
-----------
**End-of-Life types**: The module defines five eol material types:

- ``collected``: Material that has been collected for processing
- ``reusable``: Collected material suitable for direct reuse within the same sector or application
- ``recyclable``: Collected but non-reusable aterial that can be recycled into new products
- ``losses``: Non-collected, non-reusable and non-recyclable material that is lost to landfill, or other disposal methods

Integration with :py:class:`~imagematerials.model.GenericStocks`
------------------------------

The eol module's output rates are applied in the stock outflow calculations:

1. Material outflows from the :py:class:`~imagematerials.model.GenericStocks` model (by cohort) are multiplied by eol rates
2. This distributes outflows into collection, reuse, recycling, and loss pathways
3. Results can be used to:
   
   - calculate final material demand after reuse; 
   - limit secondary production in IMAGE-energy based on available recyclable material.

References
----------

United Nations Environment Programme (2024): Global Resources Outlook 2024: Bend the Trend – Pathways to a liveable planet as resource use spikes. International Resource Panel. Nairobi. 
