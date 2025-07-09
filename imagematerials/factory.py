"""Module to dynamically create models."""
import pickle as pkl
from pathlib import Path
import pickle as pkl
from pathlib import Path
from typing import Any, Optional, Union

import prism
import xarray as xr


class Sector():
    """Sector containing its preprocessing data and coordinates."""

    def __init__(self, name: str,
                 data: dict[str, Any],
                 coordinates: Optional[dict[str, list]] = None,
                 check_coordinates: bool = True):
        """Initialize the sector.

        Parameters
        ----------
        name
            Name of the sector, e.g. vehicles, buildings.
        data
            Preprocessing data for the sector.
        coordinates, optional
            (Extra) coordinates for the sectors. Any coordinates available in the preprocessing
            are added when creating the sector, by default None.
        check_coordinates, optional
            Whether to check the compatibility of the coordinates of the sector, by default True

        """
        self.name = name
        self.prep_data = {k: v for k, v in data.items()}  # Read-only copy
        self.all_data = {k: v for k, v in data.items()}  # Outputs of calculations + preprocessing
        self.check_coordinates = check_coordinates

        coordinates = {} if coordinates is None else {k: v for k, v in coordinates.items()}

        # Add the coordinates present in the data and check compatibility.
        self.coordinates, self.coordinate_sources = self._add_data_coordinates(data, coordinates)

    def _add_data_coordinates(self, data: dict[str, Any], coordinates: dict[str, list]):
        # Keep track of the arrays in which coordinates were present, used for error messages.
        coordinate_sources = {name: "manually set" for name in coordinates}
        # Add the new coordinates
        for input_name, array in data.items():
            if not isinstance(array, xr.DataArray):
                continue
            for coord in array.coords.values():
                coord_list = list(coord.values)
                # New coordinate
                if coord.name not in coordinates:
                    coordinates[coord.name] = coord_list
                    coordinate_sources[coord.name] = [input_name]
                elif self.check_coordinates:
                            try:
                                if coord_list != coordinates[coord.name]:
                                    raise ValueError(
                                        f"Mismatch in coordinates with dimension '{coord.name}'"
                                        f" with data array '{input_name}' having different coordinates"
                                        f" than previously assumed in '{coordinate_sources[coord.name]}'."
                                        f"New: {coord_list}\n\nOld:{coordinates[coord.name]}")
                                coordinate_sources[coord.name].append(input_name)
                            except ValueError as e:
                                print(e)
                                continue  # Skip this coordinate and continue with the next
        return coordinates, coordinate_sources


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

    def __init__(self, sectors: list[Sector],
                 complete_timeline: prism.Timeline):
        """Initialize the factory with prepocessing data.

        Parameters
        ----------
        sectors:
            Preprocessing data and coordinates for a sector, see :class:`Sector`.
        complete_timeline:
            The complete timeline of the data, including both the historic tail and simulation part.
        check_coordinates:
            Whether to check the compatibility of the input coordinates.

        """
        if isinstance(sectors, Sector):
            sectors = [sectors]
        self.sectors = {sec.name: sec for sec in sectors}
        self.models = []
        self.complete_timeline = complete_timeline
        self.datamap = []

    def add(self,
            model_class: prism.Model,
            sector_name: Union[None, str, list[str]] = None,
            input_sources: Optional[dict] = None):
        """Add a submodel to the main model.

        Parameters
        ----------
        model_class
            The class of the model to be added uninitialized.
        sector_name:
            Sectors for the model class to be run on, can be one, multiple or None in which case
            there is assumed to be a single sector.
        input_sources:
            Dictionary to find the input data of the model if they are not (only) in the current
            sector. Example: {"stocks": "bld"} or {"stocks": ["bld", "vhc"]}.

        Returns
        -------
            The factory itself, so that the creation of the model can be stacked.

        """
        # Parse sectors argument
        if isinstance(sector_name, (list, tuple)):
            for sec in sector_name:
                self.add(model_class, sector_name=sec, input_sources=input_sources)
            return self
        if sector_name is None:
            if len(self.sectors) != 1:
                raise ValueError(f"Cannot add model '{model_class}', need a value for sector,"
                                 f"since multiple sectors are available: {self.sectors}.")
            sector_name = list(self.sectors)[0]
        sector = self.sectors[sector_name]

        if sector_name not in self.sectors:
            raise KeyError(f"Cannot find sector '{sector_name}'. Available: {list(self.sectors)}.")

        # Find default data names
        input_data = getattr(model_class, "input_data", tuple())
        output_data = getattr(model_class, "output_data", tuple())

        input_sources = {} if input_sources is None else input_sources

        init_args = {}  # Arguments to be passed to the model at initialization.
        compute_args = {}
        # Add coordinates
        for dim_name, dim_type in model_class.__annotations__.items():
            if isinstance(dim_type, prism._typing.CoordsType):
                init_args[dim_name] = sector.coordinates[dim_name]

        def _get_data(var_name):
            """Get the preprocessing/output data from the sectors."""
            cur_sector_names = input_sources.get(var_name, sector_name)
            # Only one sector to get the data from.
            if isinstance(cur_sector_names, str):
                cur_sector = self.sectors[cur_sector_names]
                if var_name not in cur_sector.all_data:
                    raise KeyError(f"Cannot find '{var_name}' in sector '{cur_sector_names}'")
                return cur_sector.all_data[var_name], [cur_sector.name]
            # Multiple sectors -> return a list of data for each of the sectors.
            else:
                data = []
                sec_sources = []
                for cur_sec_name in cur_sector_names:
                    cur_sector = self.sectors[cur_sec_name]
                    if var_name not in cur_sector.all_data:
                        raise KeyError(f"Cannot find '{var_name}' in sector '{cur_sector_names}'")
                    data.append(cur_sector.all_data[var_name])
                    sec_sources.append(cur_sec_name)
                return data, sec_sources

        self.datamap.append({
            "model_name": model_class.__name__,
            "input": [],
            "output": output_data,
            "sector": sector_name,
        })
        for var_name in input_data:
            if var_name in model_class.__annotations__:  # Static data
                init_args[var_name], sec_sources = _get_data(var_name)
            else:  # Dynamic data
                compute_args[var_name], sec_sources = _get_data(var_name)
            self.datamap[-1]["input"].append((var_name, sec_sources))

        # Initialize the new submodel.
        new_sub_model = model_class(self.complete_timeline, **init_args)
        new_sub_model.compute_initial_values(self.complete_timeline)

        # Add output data of sub model to dictionary.
        for output_name in output_data:
            if not hasattr(new_sub_model, output_name):
                raise ValueError(f"Error in model '{new_sub_model}': "
                                 f"initialization did not create '{output_name}'")
            sector.all_data[output_name] = getattr(new_sub_model, output_name)

        # Add the new submodel to the list of submodels.
        new_sub_model.compute_args = compute_args
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
                        if historic_time >= t:
                            break
                        self._compute_one_timestep(prism_time)
                    self.historic_tail_computed = True
                self._compute_one_timestep(time)

            def _compute_one_timestep(self, time: prism.Time):
                print(f"{time.t}", end="\r")
                for model in self.submodels:
                    model.compute_values(time, **model.compute_args)

            def __getattribute__(self, attr):
                """Get the attributes of the main module.

                It also returns the data as attributes. So model.stocks will
                give the stocks of all sectors.
                """
                try:
                    return super().__getattribute__(attr)
                except AttributeError as exc:
                    if len(factory.sectors) == 1:
                        try:
                            return list(factory.sectors.values())[0].all_data[attr]
                        except KeyError as exc:
                            raise AttributeError(f"Cannot find attribute {attr} for "
                                                 f"{self.__class__}") from exc
                    all_data = {sec_name: factory.sectors[sec_name].all_data[attr]
                                for sec_name in factory.sectors
                                if attr in factory.sectors[sec_name].all_data}
                    if len(all_data) == 0:
                        raise AttributeError(f"Cannot find attribute {attr} for "
                                             f"{self.__class__}")
                    return all_data

            def save_pkl(self, data_fp: Union[Path, str]):
                all_data = {sec_name: getattr(self, sec_name) for sec_name in factory.sectors}
                with open(data_fp, "wb") as handle:
                    pkl.dump(all_data, handle)

        main_model = MainModule(self.complete_timeline)

        # Link all input data in the main model, so that you can do model.stocks
        # instead of model.submodels[0].stocks.
        for sec_name, sec in self.sectors.items():
            setattr(main_model, sec_name, sec.all_data)
        return main_model

    def visualize(self, px_x: str = "500px", px_y: str = "500px",
                  notebook: bool = True):
        try:
            import networkx as nx
            from pyvis.network import Network
        except ImportError:
            raise ImportError("Install networkx and pyvis to use visualization: pip install networkx pyvis")

        label_id = 0
        data_nodes = {}
        model_nodes = {}
        prep_nodes = {}

        sec_groups = {sec: i for i, sec in enumerate(self.sectors)}
        graph = nx.DiGraph()
        # # Add sector preprocessing node
        # for sec in self.sectors:
        #     graph.add_node(label_id, group=sec_groups[sec], label=f"{sec}.preprocessing")
        #     prep_nodes[f"{sec}.preprocessing"] = label_id
        #     label_id += 1

        for map in self.datamap:
            # Add model node
            sector_name = map["sector"]
            model_name = map["model_name"]
            full_model_name = f"{sector_name}.{model_name}"
            model_nodes[full_model_name] = label_id
            graph.add_node(label_id, label=model_name, group=sec_groups[sector_name])
            label_id += 1

            # add inputs
            for var_name, sectors in map["input"]:
                if isinstance(sectors, str):
                    sectors = [sectors]
                for sec in sectors:
                    full_var_name = f"{sec}.{var_name}"
                    if full_var_name not in data_nodes:
                        if f"{sec}.preprocessing" not in prep_nodes:
                            graph.add_node(label_id, group=sec_groups[sec], label=f"{sec}.preprocessing")
                            prep_nodes[f"{sec}.preprocessing"] = label_id
                            label_id += 1
                        source_id = prep_nodes[f"{sec}.preprocessing"]
                        graph.add_node(label_id, label=full_var_name, group=sec_groups[sec])
                        data_nodes[full_var_name] = label_id
                        graph.add_edge(source_id, label_id)
                        source_id = label_id
                        label_id += 1
                    else:
                        source_id = data_nodes[full_var_name]
                    graph.add_edge(source_id, model_nodes[full_model_name])

            # Add outputs
            for output in map["output"]:
                full_output_name = f"{sec}.{output}"
                assert full_output_name not in data_nodes
                graph.add_node(label_id, label=full_output_name, group=sec_groups[sector_name])
                graph.add_edge(model_nodes[full_model_name], label_id, group=sec_groups[sector_name])
                data_nodes[full_output_name] = label_id
                label_id += 1

        net = Network(px_y, px_x, notebook=notebook, directed=True, cdn_resources="in_line")
        net.from_nx(graph)
        return net
