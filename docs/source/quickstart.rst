Quick start guide
=================

In this quick start guide, we will walk you through the basic steps to get started with IMAGE-Materials. By the end of this guide, you will have a working installation and a simple project set up.
Have a look at our example projects in the `examples/` folder of the repository for more examples of other sectors or combined approaches.

In this quick start guide, we will cover the following steps:

* Importing libraries
* Setting up a basic model for a specific sector (e.g., buildings)
* Running a simple simulation
* Accessing the results

**Note**

A more elaborate version of this page is also available as an interactive tutorial available on the :doc:`/tutorials` page.


1. Importing libraries
----------------------
First, we need to import the necessary libraries. Make sure you have everything installed as per the :doc:`/installation` instructions.
Libraries specific to this model are imagematerials and prism.

.. code-block:: python

       import matplotlib.pyplot as plt

       from pathlib import Path

       import prism

       from imagematerials.eol import eol_preprocess
       from imagematerials.factory import ModelFactory, Sector
       from imagematerials.model import GenericStocks, MaterialIntensities
       from imagematerials.preprocessing import get_preprocessing_data


2. Preparing data for a basic model
-----------------------------------
Next we need to decide on a climate scenario (mostly these are SSP scenarios). In this quick start guide, we will not use circularity strategies, but you can find examples of how to do this in the `examples/` folder of the repository.
Therefore we need to define a scenario and the path to the scenario data. Make sure that the data is available.

.. code-block:: python

       scenario_name = "SSP2_M_CP"
       climate_policy_scenario_dir = Path("..", "data", "raw", "image", scenario_name)


3. Setting up the preprocessing
-------------------------------
We define the timeline for the complete model and the simulation timeline.
We also need to run the preprocessing step to get the data for the buildings sector. This will prepare the input data needed for the simulation.
For buildings this is for example the floorspace, population and service value added. 
Here we also calculate the inflow, stocks and outflow of materials of the buildings sector from before the simulation timeline.

.. code-block:: python

       time_start = 1960
       complete_timeline = prism.Timeline(time_start, 2100, 1)
       simulation_timeline = prism.Timeline(1970, 2100, 1)

       # calculate the preprocessing data for the buildings sector
       bld_sector = get_preprocessing_data("buildings", Path("..", "data", "raw"), 
                                                                      climate_policy_scenario_dir, 
                                                                      circular_economy_scenario_dirs = None) 


4. Setting up the simulation
----------------------------
Before we can run the simulation, we need to set up the model. We will use a factory to create a model for the buildings sector with generic stocks and material intensities.

.. code-block:: python

       factory = ModelFactory(
              [bld_sector], complete_timeline
       ).add(GenericStocks, ["buildings"]
       ).add(MaterialIntensities, ["buildings"]
       )
       model = factory.finish()


5. Running the simulation
-------------------------
Now we can run the simulation. We will use a warning filter to ignore any warnings that may arise during the simulation.

.. code-block:: python

       import warnings
       with warnings.catch_warnings():
              warnings.filterwarnings("ignore")
              model.simulate(simulation_timeline)


6. Analyzing the results
------------------------
Finally, we can analyze the results of the simulation. Here, we will plot the total material stock in buildings over time.  
This you can do my accessing the model like this: 

.. code-block:: python

       model.buildings.keys()

With that you get an list of all potential variables you can access.
To plot the inflow of steel into residential and commercial buildings summed over region (thus global) and type, you can do: 

.. code-block:: python

       model.buildings.get("inflow_materials").to_array().sel(material = 'Steel', Type = ['Appartment - Rural', 'Appartment - Urban', 'Detached - Rural',
                 'Detached - Urban', 'High-rise - Rural', 'High-rise - Urban',
                 'Semi-detached - Rural', 'Semi-detached - Urban']).sum(["Region", "Type"]).plot(label = 'residential')

       model.buildings.get("inflow_materials").to_array().sel(material = 'Steel', Type = ['Office',
                 'Retail+', 'Hotels+', 'Govt+']).sum(["Region", "Type"]).plot(label = 'government')

       plt.legend()
