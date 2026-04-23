###########
Electricity
###########

*********
Overview
*********
The electricity model covers i) electricity generation, ii) transmission and distribution grid, and 
iii) grid energy storage. 
Multiple generation technologies are covered, both conventional (with and without carbon capture and 
storage and heat production) and renewable. 
Grid is split up into high (aboveground and below ground), medium, and low voltage lines 
and grid additions (transformers and substations). The storage includes hydropower and stationary 
storage in the form of grid-scale batteries, mechanical storage technologies, and fuel-cells. The 
dedicated storage demand can be altered depending on availability assumptions of electric vehicle 
batteries using the vehicle-to-grid technology.

******
Scope
******
The electricity module covers:

- **Bulk materials**: steel, concrete, copper, aluminium, glass, and plastics
- **Minor and technology metals**: cobalt, lead, lithium, manganese, neodymium, nickel, tantalum
- **Sub-sectors**: power generation, grid (lines, transformers, substations), and storage
- **Regions**: all 26 IMAGE regions
- **Time horizon**: 1921-2100


*************
Preprocessing
*************

Data Input
==========

IMAGE provides the following **endogenous inputs**:

- Electricity generation capacity by technology (MW peak capacity)
- Electricity grid storage energy reservoir (MWh)
- Electricity grid storage power capacity (MW)
- GDP per capita (USD/person)

All endogenous variables are indexed over time (*t*), technology (*T*), and/or region (*R*) as applicable.

**Exogenous inputs** are divided into scenario-independent and scenario-dependent data.

Scenario-independent data includes: HV grid line lengths per region in 2016 (km); regional ratios of HV to MV and LV line lengths; 
regression constants relating GDP per capita to the share of underground power lines; the number of transformers and substations 
per km of grid; storage costs ($/ct/kWh) and cost corrections; assumptions on long-term post-2050 storage price decline; 
storage energy density (kg/kWh); and regional hydrodam power capacity (MW).

Scenario-dependent data comprises technology lifetimes and material intensities.


Structure
=========
The electricity preprocessing module is organized into four sub-modules:

- **Generation** (``electricity.generation``): Prepares generation capacity data from IMAGE-energy
  outputs combined with external data on lifetimes and material intensities.
- **Grid** (``electricity.grid``): Infers grid line and grid addition (transformers and substations)
  stocks from IMAGE-energy generation capacity data and external data sources.
- **Storage** (``electricity.storage``): Infers energy storage capacity stocks for pumped hydropower storage (PHS),
  and grid-scale dedicated storage (e.g. batteries, compressed air, flywheels) from IMAGE-energy power and energy storage capacity data 
  and external data sources.
- **Circular Economy** (``electricity.circular_economy_measures``): Applies circular economy measures
  to material intensities and lifetimes for generation and grid additions.

Within the first three sub-modules, the data is processed in a stepwise manner, broadly following:
1. Data loading and preparation: Read IMAGE-energy outputs and external data, and perform necessary
formatting.
1. Convert to xarray: Create xarray DataArrays with dimensions Time, Region, Type, material (where relevant).
2. Interpolation and Extrapolation.
3. Execute relevant calculations (e.g. calculate market shares from costs).

Assumptions
===========



Output
======


**********
References
**********
