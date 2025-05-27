"""Module to dynamically create models."""
from typing import Any, Optional

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

    def __init__(self, namespaces: dict[str, Any], complete_timeline: prism.Timeline,
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
        self.namespaces = {ns.name: ns for ns in namespaces}
        self.models = []
        self.complete_timeline = complete_timeline
        self.check_coordinates = check_coordinates

    def _get_linked_input_data(self, namespace, input_sources, input_data, optional_input_data):
        linked_input_data = []
        arguments_dict = {}
        for var_name in input_data + optional_input_data:
            cur_namespace = input_sources.get(var_name, namespace)
            if isinstance(cur_namespace, str):
                ns = self.namespaces[cur_namespace]
                if var_name in ns.prep_data:
                    arguments_dict[var_name] = ns.all_data[var_name]
                elif var_name in optional_input_data:
                    arguments_dict[var_name] = None
                else:
                    linked_input_data.append((cur_namespace, var_name))
            else:
                if all(var_name in self.namespaces[ns_name].prep_data for ns_name in cur_namespace):
                    arguments_dict[var_name] = [self.namespaces[ns].all_data[var_name] for ns in cur_namespace]
                else:
                    linked_input_data.append((cur_namespace, var_name))
        return arguments_dict, linked_input_data

    def add(self, model_class, namespace=_DEFAULT_NAMESPACE, input_sources: Optional[dict] = None):
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
        if isinstance(namespace, list):
            for ns in namespace:
                self.add(model_class, namespace=ns, input_sources=input_sources)
            return self

        if namespace not in self.namespaces:
            raise KeyError(f"Cannot find namespace '{namespace}'. Available: {self.name_list}.")

        # Find default data names
        input_data = getattr(model_class, "input_data", tuple())
        optional_input_data = getattr(model_class, "optional_input_data", tuple())
        output_data = getattr(model_class, "output_data", tuple())

        input_sources = {} if input_sources is None else input_sources

        arguments_dict = {}
        # Add coordinates
        for dim_name, dim_type in model_class.__annotations__.items():
            if isinstance(dim_type, prism._typing.CoordsType):
                arguments_dict[dim_name] = self.namespaces[namespace].coordinates[dim_name]

        # Add input data either from the preprocessing data or other submodels.
        new_arguments_dict, linked_input_data = self._get_linked_input_data(
            namespace, input_sources, input_data, optional_input_data)

        arguments_dict.update(new_arguments_dict)

        # Initialize the new submodel.
        new_sub_model = model_class(self.complete_timeline, **arguments_dict)
        new_sub_model.compute_initial_values(self.complete_timeline)

        # Add output data of sub model to dictionary.
        for output_name in output_data:
            self.namespaces[namespace].all_data[output_name] = getattr(new_sub_model, output_name)

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
                    linked_args = {}
                    for namespace, var_name in model.linked_input_data:
                        if isinstance(namespace, str):
                            linked_args[var_name] = getattr(self, namespace)[var_name]
                        else:
                            linked_args[var_name] = [getattr(self, ns)[var_name] for ns in namespace]
                    model.compute_values(time, **linked_args)

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
        for ns in self.namespaces.values():
            setattr(main_model, ns.name, ns.all_data)
        return main_model


class Namespace():
    def __init__(self, name, data, coordinates = None, check_coordinates = True):
        self.name = name
        self.prep_data = {k: v for k, v in data.items()}
        self.all_data = {k: v for k, v in data.items()}
        self.check_coordinates = check_coordinates

        coordinates = {} if coordinates is None else {k: v for k, v in coordinates.items()}

        self.coordinates, self.coordinate_sources = self._add_data_coordinates(data, coordinates)

    def _add_data_coordinates(self, data, coordinates):
        coordinate_sources = {name: "manually set" for name in coordinates}
        # Add the new coordinates
        for input_name, array in data.items():
            if not isinstance(array, xr.DataArray):
                continue
            for coord in array.coords.values():
                coord_list = list(coord.values)
                if coord.name not in coordinates:
                    coordinates[coord.name] = coord_list
                    coordinate_sources[coord.name] = [input_name]
                elif self.check_coordinates:
                    if coord_list != coordinates[coord.name]:
                        raise ValueError(
                            f"Mismatch in coordinates with dimension '{coord.name}'"
                            f" with data array '{input_name}' having different coordinates"
                            f" than previously assumed in '{self.coordinate_sources[coord.name]}'."
                            f"New: {coord_list}\n\nOld:{self.coordinates[coord.name]}")
                    coordinate_sources[coord.name].append(input_name)
        return coordinates, coordinate_sources


