

from dataclasses import dataclass
from typing import Optional

import xarray as xr


class KnowledgeGraph():
    def __init__(self, *args):
        self._items = args
        self._items_dict = {item.name: item for item in self._items}
        self.validate()

    def validate(self):
        if len(self._items) != len(self._items_dict):
            raise ValueError("Validation error: items with the same name in the knowledge graph.")
        for item in self._items:
            if item.inherits_from is not None and item.inherits_from not in self._items_dict:
                raise ValueError(f"Validation error: {item.name} inherits from unknown item "
                                 f"{item.inherits_from}")
        # TODO: check for cycles

    def _find_closest_parent(self, key: str, input_coords):
        if key in input_coords:
            return key
        parent = self._items_dict[key].inherits_from
        if parent is not None and parent in input_coords:
            return parent
        # TODO: Also allow parents more than one level up.
        raise KeyError(f"Cannot find any available parent for {key}")

    def rebroadcast_xarray(self, input_array, output_coords, dim="Type"):
        new_coords = {coord.name: coord for coord in input_array.coords.values()}
        new_coords[dim] = output_coords
        input_coords = input_array.coords[dim].values

        keep_coords = []
        new_array = xr.DataArray(0.0, dims=input_array.dims, coords=new_coords)
        for cur_coord in output_coords:
            if cur_coord in input_coords:
                keep_coords.append(cur_coord)
                continue
            parent = self._find_closest_parent(cur_coord, input_coords)
            new_array.loc[{dim: cur_coord}] = input_array.loc[{dim: parent}]
        new_array.loc[{dim: keep_coords}] = input_array.loc[{dim: keep_coords}]
        return new_array

    def rebroadcast_xarray_impute(self, input_array, output_coords, dim="Type", imputation_value=0.0):
        new_coords = {coord.name: coord for coord in input_array.coords.values()}
        new_coords[dim] = output_coords
        input_coords = input_array.coords[dim].values

        keep_coords = []
        new_array = xr.DataArray(0.0, dims=input_array.dims, coords=new_coords)
        for cur_coord in output_coords:
            if cur_coord in input_coords:
                keep_coords.append(cur_coord)
                continue
            try:
                parent = self._find_closest_parent(cur_coord, input_coords)
                new_array.loc[{dim: cur_coord}] = input_array.loc[{dim: parent}]
            except KeyError:
                new_array.loc[{dim: cur_coord}] = imputation_value
        new_array.loc[{dim: keep_coords}] = input_array.loc[{dim: keep_coords}]
        return new_array

@dataclass
class Node():
    name: str
    inherits_from: Optional[str] = None

vehicle_subtypes = ["BEV", "FCV", "HEV", "ICE", "PHEV", "Trolley"]
vehicle_supertypes = ["Cars", "Heavy Freight Trucks", "Light Commercial Vehicles",
                      "Medium Freight Trucks", "Regular Buses", "Midi Buses"]

vehicle_nodes = []

for super_type in vehicle_supertypes:
    vehicle_nodes.append( Node(super_type))
    for sub_type in vehicle_subtypes:
        vehicle_nodes.append(Node(f"{super_type} - {sub_type}", super_type))
vehicle_knowledge_graph = KnowledgeGraph(*vehicle_nodes)

building_nodes = []
for supertype in ["Detached", "Semi-detached", "Appartment", "High-rise"]:
    building_nodes.append(Node(supertype))

    for subtype in ["Urban", "Rural"]:
        building_nodes.append(Node(f"{supertype} - {subtype}", supertype))


building_knowledge_graph = KnowledgeGraph(*building_nodes)
knowledge_graph = KnowledgeGraph(*building_nodes, *vehicle_nodes)


# building_knowledge_graph = KnowledgeGraph(
#     Node("Detached"),
#     Node("Detached - Urban", "Detached"),
#     Node("Detached - Rural", "Detached"),
#     Node("Detached"),
#     Node("Detached - Urban", "Detached"),
#     Node("Detached - Rural", "Detached"),
    
# )