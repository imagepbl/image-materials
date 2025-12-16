Factory and Orchestration
=========================

The Factory system in IMAGE materials provides the main orchestration layer for creating and managing model workflows.

Overview
--------

The Factory system, implemented through the :class:`imagematerials.factory.Sector` class, serves as the primary entry point for running IMAGE materials models. It handles:

- Data preprocessing and validation
- Model instantiation and configuration  
- Coordinate management across models
- Computational workflow orchestration

Sector Class
------------

.. autoclass:: imagematerials.factory.Sector
   :members:
   :show-inheritance:

Key Features
------------

**Data Management**: The Sector class manages both preprocessing data and computational outputs in a unified framework.

**Coordinate Validation**: Ensures consistency of dimensions (Region, Type, Time, etc.) across all models in a sector.

**Model Integration**: Provides a clean interface for creating and running complex model workflows.

**Extensibility**: Supports adding new models and data sources through a flexible architecture.

Usage Pattern
-------------

A typical workflow using the Factory system:

1. **Create Sector**: Initialize with preprocessing data and coordinates
2. **Add Models**: Configure which models to use (stocks, materials, etc.)
3. **Run Simulation**: Execute the computational workflow
4. **Access Results**: Retrieve outputs from the integrated models

Example
-------

.. code-block:: python

   from imagematerials.factory import Sector
   
   # Create a sector with preprocessing data
   sector = Sector(
       name="vehicles",
       data=preprocessing_data,
       coordinates={"Region": regions, "Type": vehicle_types}
   )
   
   # Configure and run models
   sector.run_simulation(timeline)
   
   # Access results
   results = sector.get_results()

Relationship to Other Models
----------------------------

The Factory system coordinates:

- **Stock Models** (:class:`GenericStocks`, :class:`SharesInflowStocks`)
- **Material Models** (:class:`GenericMaterials`, :class:`MaterialIntensities`)  
- **Specialized Models** (:class:`EndOfLife`, :class:`RestOf`)

It replaces the earlier :class:`GenericMainModel` approach with a more flexible and maintainable architecture.

See Also
--------

- :doc:`Models Overview <models>` - How the Factory fits into the overall modeling framework
- :doc:`API Reference <api/modules>` - Technical documentation for factory classes