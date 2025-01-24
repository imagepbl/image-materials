"""Module to dynamically create models."""
from typing import Any, Optional

import xarray as xr

import prism


class ModelFactory():
    """Factory class to create simulation models.

    The model factory is a way to dynamically create prism models.
    It starts with an empty model and adds one feature one at a time.
    The submodels will be exectuted in the order that they were added
    to the factory. After adding all the submodels, the model can be generated
    using the finish method of the factory.

    Examples
    --------
    >>> factory = ModelFactory(prep_data, prism.Timeline(1721, 2060, 1))
    >>> factory.add(GenericStocks)
    >>> factory.add(GenericMaterials)
    >>> model = factory.finish()

    """

    def __init__(self, prep_data: dict[str, Any], complete_timeline: prism.Timeline):
        """Initialize the factory with prepocessing data.

        Parameters
        ----------
        prep_data:
            Preprocessing data
        complete_timeline:
            The complete timeline of the data, including both the historic tail and simulation part.

        """
        self.prep_data = prep_data
        # Preprocessing data + output data of the submodels
        self.all_data = {key: value for key, value in prep_data.items()}
        self.models = []
        self.coordinates = self.create_coordinates()
        self.complete_timeline = complete_timeline

    def create_coordinates(self) -> dict[str, list[Any]]:
        """Get all the available coordinates from the preprocessing data.

        Also validates that there are no mismatched coordinates between data arrays.

        Returns
        -------
        coordinates:
            All coordinates in the preprocessing data arrays as a list.

        Raises
        ------
        ValueError:
            If coordinates with the same dimension do not match up.

        """
        coordinates = {}
        for data_name, data_obj in self.prep_data.items():
            # Ignore preprocessing data that are not xArray DataArrays.
            if not isinstance(data_obj, xr.DataArray):
                continue
            for coor in data_obj.coords.values():
                if coor.name not in coordinates:
                    coordinates[coor.name] = list(coor.values)
                elif not list(coor.values) == coordinates[coor.name]:
                    raise ValueError(f"Mismatched dimensions in input data for {data_name} "
                                     f"for coordinates {coor.name}")
        return coordinates

    def add(self, model_class: str,
            input_data: Optional[tuple[str]] = None,
            optional_input_data: Optional[tuple[str]] = None,
            output_data: Optional[tuple[str]] = None):
        """Add a submodel to the main model.

        Parameters
        ----------
        model_class
            The class of the model to be added uninitialized.
        input_data, optional
            Override the input data names, by default None
        optional_input_data, optional
            Override the optional input data names, by default None
        output_data, optional
            Override the output data name, by default None

        Returns
        -------
            The factory itself, so that the creation of the model can be stacked.

        """
        # Find default data names
        if input_data is None:
            input_data = getattr(model_class, "input_data", tuple())
        if optional_input_data is None:
            optional_input_data = getattr(model_class, "optional_input_data", tuple())
        if output_data is None:
            output_data = getattr(model_class, "output_data", tuple())

        arguments_dict = {}
        # Add coordinates
        for var_name, var_type in model_class.__annotations__.items():
            if isinstance(var_type, prism._typing.CoordsType):
                arguments_dict[var_name] = self.coordinates[var_name]

        # Add input data either from the preprocessing data or other submodels.
        linked_input_data = []
        for var_name in input_data + optional_input_data:
            if var_name in self.prep_data:
                arguments_dict[var_name] = self.prep_data[var_name]
            else:
                assert var_name in self.all_data
                linked_input_data.append(var_name)

        # Initialize the new submodel.
        # print(list(arguments_dict))
        new_sub_model = model_class(self.complete_timeline, **arguments_dict)
        new_sub_model.compute_initial_values(self.complete_timeline)

        # Add output data of sub model to dictionary.
        for output_name in new_sub_model.output_data:
            self.all_data[output_name] = getattr(new_sub_model, output_name)

        # Add the new submodel to the list of submodels.
        new_sub_model.linked_input_data = linked_input_data
        self.models.append(new_sub_model)
        return self

    def finish(self):
        """Finish the creation of the main model.

        Returns
        -------
            A prism model that can be used to do the simulations.

        """
        # For injection of data of the factory class.
        factory = self
        @prism.interface
        class MainModule(prism.Model):
            def compute_initial_values(self, time: prism.Time):
                self.historic_tail_computed = False

            def init_submodels(self, timeline: prism.Timeline):
                self.submodels = factory.models

            def compute_values(self, time: prism.Time):
                t, dt = time.t, time.dt
                if not self.historic_tail_computed:
                    for historic_time in factory.complete_timeline:
                        prism_time = prism.Time(factory.complete_timeline.start,
                                                factory.complete_timeline.end,
                                                dt, historic_time)
                        if historic_time == factory.complete_timeline.start:
                            continue
                        if historic_time >= t:
                            break
                        self._compute_one_timestep(prism_time)
                    self.historic_tail_computed = True
                self._compute_one_timestep(time)

            def _compute_one_timestep(self, time: prism.Time):
                for model in self.submodels:
                    model.compute_values(time, **{var_name: getattr(self, var_name)
                                                  for var_name in model.linked_input_data})
        main_model = MainModule(self.complete_timeline)

        # Link all input data in the main model, so that you can do model.stocks
        # instead of model.submodels[0].stocks.
        for data_name, data in self.all_data.items():
            setattr(main_model, data_name, data)
        return main_model
