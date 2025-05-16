"""Module to dynamically create models."""
from typing import Any

import prism
import xarray as xr

_DEFAULT_NAMESPACE = "default"


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

    def __init__(self, prep_data: dict[str, Any], complete_timeline: prism.Timeline,
                 check_coordinates: bool = True):
        """Initialize the factory with prepocessing data.

        Parameters
        ----------
        prep_data:
            Preprocessing data
        complete_timeline:
            The complete timeline of the data, including both the historic tail and simulation part.
        check_coordinates:
            Whether to check the compatibility of the input coordinates.

        """
        if all([isinstance(p, dict) for p in prep_data.values()]):
            self.prep_data = prep_data
        else:
            self.prep_data = {_DEFAULT_NAMESPACE: prep_data}
        # Preprocessing data + output data of the submodels
        self.all_data = {namespace: {} for namespace in self.prep_data}
        self.coordinates = {namespace: {} for namespace in self.prep_data}
        self.models = []
        self.complete_timeline = complete_timeline
        self.check_coordinates = check_coordinates

    def _add_input_data(self, namespace, input_data, model_class, optional=False):
        for input_name in input_data:
            if input_name in self.all_data[namespace]:
                continue

            if input_name not in self.prep_data[namespace] and not optional:
                raise ValueError(f"Cannot find input data '{input_name}' in namespace '{namespace}'"
                                 f" for model class '{model_class}'.")

            self.all_data[namespace][input_name] = self.prep_data[namespace].get(input_name, None)

            if not isinstance(self.all_data[namespace][input_name], xr.DataArray):
                continue

            # Add the new coordinates
            for coord in self.all_data[namespace][input_name].coords.values():
                coord_list = list(coord.values)
                if coord.name not in self.coordinates[namespace]:
                    self.coordinates[namespace][coord.name] = (coord_list, [input_name])
                    if coord.name == "Time" and "Cohort" not in self.coordinates[namespace]:
                        self.coordinates[namespace]["Cohort"] = (coord_list, [input_name])
                elif self.check_coordinates:
                    if coord_list != self.coordinates[namespace][coord.name][0]:
                        raise ValueError(
                            f"Mismatch in coordinates with dimension '{coord.name}'"
                            f" with data array '{input_name}' having different coordinates"
                            f" than previous models '{self.coordinates[namespace][coord.name][1]}'."
                            f"{coord_list} vs {self.coordinates[namespace][coord.name][0]}")
                    self.coordinates[namespace][coord.name][1].append(input_name)

    def add(self, model_class, namespace=_DEFAULT_NAMESPACE):
        """Add a submodel to the main model.

        Parameters
        ----------
        model_class
            The class of the model to be added uninitialized.
        namespace:
            Name space for the model class to be run on.

        Returns
        -------
            The factory itself, so that the creation of the model can be stacked.

        """
        # Find default data names
        input_data = getattr(model_class, "input_data", tuple())
        optional_input_data = getattr(model_class, "optional_input_data", tuple())
        output_data = getattr(model_class, "output_data", tuple())

        self._add_input_data(namespace, input_data, model_class, False)
        self._add_input_data(namespace, optional_input_data, model_class, True)

        arguments_dict = {}
        # Add coordinates
        for dim_name, dim_type in model_class.__annotations__.items():
            if isinstance(dim_type, prism._typing.CoordsType):
                arguments_dict[dim_name] = self.coordinates[namespace][dim_name][0]

        # Add input data either from the preprocessing data or other submodels.
        linked_input_data = []
        for var_name in input_data + optional_input_data:
            prep_data = self.prep_data[namespace]
            all_data = self.all_data[namespace]
            if var_name in prep_data or var_name in optional_input_data:
                arguments_dict[var_name] = all_data[var_name]
            else:
                linked_input_data.append((namespace, var_name))

        # Initialize the new submodel.
        new_sub_model = model_class(self.complete_timeline, **arguments_dict)
        new_sub_model.compute_initial_values(self.complete_timeline)

        # Add output data of sub model to dictionary.
        for output_name in output_data:
            self.all_data[namespace][output_name] = getattr(new_sub_model, output_name)

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
                    model.compute_values(time, **{var_name: getattr(self, namespace)[var_name]
                                                  for namespace, var_name in model.linked_input_data})

            def __getattribute__(self, attr):
                try:
                    return super().__getattribute__(attr)
                except AttributeError:
                    if len(factory.all_data) == 1:
                        return getattr(self, list(factory.all_data)[0])[attr]
                    return {ns: getattr(self, ns)[attr] for ns in factory.all_data}

        main_model = MainModule(self.complete_timeline)

        # Link all input data in the main model, so that you can do model.stocks
        # instead of model.submodels[0].stocks.
        for data_name, data in self.all_data.items():
            setattr(main_model, data_name, data)
        return main_model
