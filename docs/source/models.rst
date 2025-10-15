Models 
======================

Image materials core functionality relies on the method of dynamic stock modelling. 

This section provides an overview of the models implemented in IMAGE materials. For detailed documentation of each model, please refer to the :doc:`API reference <api/models_detail>`.

**Subpages**

.. toctree::
   :maxdepth: 1

   model/survival_matrix

Model Overview
----------------------
- :class:`imagematerials.model.GenericStocks`: A class for modeling generic stock dynamics using survival matrices.
- :class:`imagematerials.model.SharesInflowStocks`: A class for modeling stock dynamics with shares inflow.
- :class:`imagematerials.model.GenericMaterials`: A class for handling generic materials.         
- :class:`imagematerials.model.MaterialIntensities`: A class for modeling materials based on material intensities.
- :class:`imagematerials.factory.Sector`: The main orchestrator class that manages model creation and coordination.
- :class:`imagematerials.model.EndOfLife`: A class for modeling end-of-life material flows.
- :class:`imagematerials.model.RestOf`: A class for modeling rest-of-economy material consumption.
- :class:`imagematerials.survival.SurvivalMatrix`: A class for computing survival probabilities based on lifetime distributions.
- :class:`imagematerials.survival.ScipySurvival`: A class providing scipy-based lifetime distributions for survival calculations.

Key Concepts
----------------------

**Dynamic Stock Modeling**: The foundation of IMAGE materials is dynamic stock modeling, which tracks how stocks (e.g. buildings, vehicles, infrastructure) change over time through inflows, outflows, and aging processes. This relies on :doc:`survival matrices <survival_matrix>` to model how stocks age and exit the system.

**Cohort-based Tracking**: Stocks are organized by cohorts (age groups), allowing for different properties of the stock over time such as a change in weight or material composition. Each cohort has its own survival characteristics.

**Material Flow Analysis**: The models track material flows through the economy, from virgin material inputs through use phases to end-of-life treatment.

Model Description and relationships
----------------------

**Model Flow and Relationships:**

.. code-block:: text

                    ┌─────────────────────┐
                    │       Factory       │
                    │   (Orchestrator)    │
                    └─────────┬───────────┘
                              │ coordinates
                              ▼
    ┌──────────────────────┐
    │   SurvivalMatrix     │
    │  (Lifetime/Aging)    │
    │                      │
    │    ScipySurvival     │
    │   (Distributions)    │
    └────────┬─────────────┘
             │ uses survival data
             ▼
    Stock Models:
    ┌─────────────────┐    Alternative     ┌──────────────────────┐
    │  GenericStocks  │ ←────────────────→ │ SharesInflowStocks   │
    └─────────────────┘                    └──────────────────────┘
             │                                    
             │ output feeds into                   
             ▼                                
    Material Models:
    ┌─────────────────┐    Alternative     ┌──────────────────────┐
    │ GenericMaterials│ ←────────────────→ │ MaterialIntensities  │
    └─────────────────┘                    └──────────────────────┘
             │
             │ material flows (inflow/outflow)
             ▼
    ┌─────────────────┐                    ┌──────────────────────┐
    │    EndOfLife    │                    │       RestOf         │
    └─────────────────┘                    │ (Calculate sectors   │
                                           │  not modelled        │
                                           │  explicitly)         │
                                           └──────────────────────┘

**Class description:**

The :class:`GenericStocks <imagematerials.model.GenericStocks>` class provides a flexible framework for modeling stock dynamics. It calculates the initial stock and uses :doc:`survival matrices <survival_matrix>` to determine stock aging and outflows. During simulation, the stock inflow and outflow are calculated based on lifetimes which influence the survival matrix of the stock. For detailed documentation, see :doc:`GenericStocks <api/models_detail>`.

The :class:`SharesInflowStocks <imagematerials.model.SharesInflowStocks>` class extends the GenericStocks class. It is designed for scenarios where stock inflows are influenced by share dynamics, allowing for more complex interactions between different stock types. It is for instance used in the :doc:`electricity <api/electricity>` module. For detailed documentation, see :doc:`SharesInflowStocks <api/models_detail>`.

The :class:`GenericMaterials <imagematerials.model.GenericMaterials>` class is built to manage material properties associated with stocks. It uses the GenericStocks class as its foundation and calculated the material need associated with the inflow, outflow and stock calculated by GenericStocks. For detailed documentation, see :doc:`GenericMaterials <api/models_detail>`.

The :class:`MaterialIntensities <imagematerials.model.MaterialIntensities>` class is an alternative to the GenericMaterials class, focusing on managing material intensities. This is for the standard case where stock weights and material fractions are used to calculate material use. It is for instance used in the :doc:`buildings <api/buildings>` module. For detailed documentation, see :doc:`MaterialIntensities <api/models_detail>`.

The :class:`Factory/Sector <imagematerials.factory.Sector>` class serves as the main orchestrator for model creation and coordination. It manages preprocessing data, coordinates model instantiation, and handles the computational workflow for different sectors. This is the primary entry point for running IMAGE materials models. For detailed documentation, see :doc:`Factory <api/factory>`.

The :class:`EndOfLife <imagematerials.model.EndOfLife>` class models end-of-life material flows, including collection, reuse, and recycling processes. It calculates material flows through circular economy pathways and determines virgin material requirements. For detailed documentation, see :doc:`EndOfLife <api/models_detail>`.

The :class:`RestOf <imagematerials.model.RestOf>` class models material consumption in the rest-of-economy sectors using Gompertz curves based on GDP per capita and population data. For detailed documentation, see :doc:`RestOf <api/models_detail>`.

The :class:`SurvivalMatrix <imagematerials.survival.SurvivalMatrix>` class provides the core functionality for computing survival probabilities over time. It determines how stocks age and when they exit the system based on lifetime distributions. This class is fundamental to all dynamic stock modeling in IMAGE materials. For detailed documentation, see :doc:`SurvivalMatrix <api/models_detail>`.

The :class:`ScipySurvival <imagematerials.survival.ScipySurvival>` class provides scipy-based statistical distributions (normal, Weibull, etc.) for computing survival probabilities. It offers a range of lifetime distribution shapes to model different types of stocks with various aging characteristics. For detailed documentation, see :doc:`ScipySurvival <api/models_detail>`.

**When to Use Each Alternative:**

**Stock Models:**

- **GenericStocks**: Use when you have straightforward stock dynamics with direct inflow data
- **SharesInflowStocks**: Use when stock inflows are determined by share dynamics between different stock types, or when you need to model complex interactions between stock categories

**Material Models:**

- **GenericMaterials**: Use for general material property management when you need flexible material handling capabilities
- **MaterialIntensities**: Use for standard cases where you have stock weights and material fractions data to calculate material use

Complete Modeling Workflow
----------------------

A typical IMAGE materials modeling workflow follows these steps:

1. **Stock Modeling**: Choose between `GenericStocks` or `SharesInflowStocks` to model stock dynamics
2. **Material Calculation**: Use `GenericMaterials` or `MaterialIntensities` to calculate material requirements
3. **Integration**: Use the `Factory/Sector` class to orchestrate all models and coordinate computations.
4. **End-of-Life Processing**: Apply `EndOfLife` model to handle circular economy flows
5. **Rest-of-Economy**: Use `RestOf` model for sectors not explicitly modeled

**Example Sector Applications:**
- **Vehicles**: Uses the standard `GenericStocks` and `GenericMaterials` models. Since this is the only model that also includes maintenance, the `Maintenance` class can be layered upon this module to also include material flows for maintenance of land transport.
- **Buildings**: Uses the standard `GenericStocks` and the specific `MaterialIntensities` for material calculations
- **Electricity**: Uses `SharesInflowStocks` for complex share dynamics in the stock calculation and the standard `GenericMaterials` for material calculations.


**Prerequisites:**
- Understanding of dynamic stock modeling concepts
- Familiarity with cohort-based analysis
- Basic knowledge of material flow analysis
- Experience with xarray and prism frameworks




For more detailed information on each model, including their methods and attributes, please refer to the :doc:`API reference <api/models_detail>`.




