from dataclasses import dataclass, field
from typing import Optional

import xarray as xr
import json

@dataclass
class Node():
    name: str
    synonyms: list[str] = field(default_factory=list)
    inherits_from: Optional[str] = None

    def match_coords(self, input_coords):
        if isinstance(input_coords, str):
            input_coords = [input_coords]
        name_list = [self.name] + self.synonyms
        if not set(name_list).intersection(input_coords):
            raise KeyError()
        for name in name_list:
            if name in input_coords:
                return name

    def to_dict(self) -> str:
        return {"name": self.name, "synonyms": self.synonyms,
                           "inherits_from": self.inherits_from}
    @classmethod
    def from_dict(cls, node_dict):
        # node_dict = json.loads(json_str)
        return cls(**node_dict)

class KnowledgeGraph():
    def __init__(self, *items):
        # nx.Graph()
        self._items: list[Node] = []
        for item in items:
            self.add(item)
    # def __init__(self, *args):
        # self._items = args
        # self._items_dict = {item.name: item for item in self._items}
        # self.validate()

    def add(self, node: Node):
        """_summary_

        Parameters
        ----------
        node
            _description_

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        if node.name in self._items:
            raise ValueError("Node already exists.")
        if node.inherits_from is not None and node.inherits_from not in self:
            print(node.inherits_from is not None, node.inherits_from not in self)
            raise ValueError(f"Parent {node.inherits_from} of {node.name} does not exist.")
        self._items.append(node)

    def __contains__(self, node_name) -> bool:
        try:
            self[node_name]
        except KeyError:
            return False
        return True

    def __getitem__(self, node_name) -> Node:
        for item in self._items:
            if item.name == node_name:
                return item
            if node_name in item.synonyms:
                return item
        raise KeyError("Key with {}")

    # def validate(self):
    #     if len(self._items) != len(self._items_dict):
    #         raise ValueError("Validation error: items with the same name in the knowledge graph.")
    #     for item in self._items:
    #         if item.inherits_from is not None and item.inherits_from not in self._items_dict:
    #             raise ValueError(f"Validation error: {item.name} inherits from unknown item "
    #                              f"{item.inherits_from}")
    #     # TODO: check for cycles

    def find_relations(self, input_coords, output_coords):
        relations = {}
        for cur_out_coord in output_coords:
            relations[cur_out_coord] = self.find_one_relation(input_coords, cur_out_coord)
            # if cur_out_coord in input_coords:
            #     relations[cur_out_coord] = [cur_out_coord]
            #     continue

            # output_node = self[cur_out_coord]
            # print([output_node.name], output_node.synonyms)
            # if set([output_node.name] + output_node.synonyms).intersection(input_coords):
            #     for syn in set([output_node.name] + output_node.synonyms):
            #         if syn in input_coords:
            #             relations[cur_out_coord] = [syn]
            #             break
            #     continue
            # try:
            #     ancestors = self._find_ancestors(input_coords, output_node)
            # except KeyError:
            #     ancestors = []
            # try:
            #     descendants = self._find_descendants(input_coords, output_node)
            # except KeyError:
            #     descendants = []

            # if len(ancestors) > 0 and len(descendants) > 0:
            #     raise ValueError(f"Cannot find relations, because {output_node} has both ancestors "
            #                      "and descendants")
            # if len(ancestors) + len(descendants) == 0:
            #     raise ValueError("Cannot find relations of {output_node} in input coordinates.")

            # print(ancestors, descendants)
            # relations[cur_out_coord] = ancestors + descendants
        return relations

    def find_one_relation(self, input_coords: list[str], output_name: str) -> list[str]:
        if output_name in input_coords:
            return [output_name]

        output_node = self[output_name]
        try:
            return [output_node.match_coords(input_coords)]
        except KeyError:
            pass

        try:
            ancestors = self._find_ancestors(input_coords, output_node)
        except KeyError:
            ancestors = []
        try:
            descendants = self._find_descendants(input_coords, output_node)
        except KeyError:
            descendants = []

        if len(ancestors) > 0 and len(descendants) > 0:
            raise ValueError(f"Cannot find relations, because {output_node} has both ancestors "
                                "and descendants")
        if len(ancestors) + len(descendants) == 0:
            raise ValueError(f"Cannot find relations of {output_node} in input coordinates.")

        return ancestors + descendants

    def find_relations_inverse(self, input_coords, output_coords):
        relations = self.find_relations(input_coords, output_coords)
        inverse_relations = {}
        for cur_output, cur_input_list in relations.items():
            inverse_relations.update({cur_input: cur_output for cur_input in cur_input_list})
        return inverse_relations

    def _find_ancestors(self, input_coords, output_node):
        try:
            return [output_node.match_coords(input_coords)]
        except KeyError:
            if output_node.inherits_from is None:
                raise KeyError("Ancestor does not exist.")
            return self._find_ancestors(input_coords, self[output_node.inherits_from])


    def _find_descendants(self, input_coords, output_node):
        try:
            return [output_node.match_coords(input_coords)]
        except KeyError:
            pass

        all_descendants = []
        for item in self._items:
            if item.inherits_from == output_node.name:
                all_descendants.extend(self._find_descendants(input_coords, item))
        return all_descendants


    # def _find_closest_parent(self, key: str, input_coords):
    #     if key in input_coords:
    #         return key
    #     parent = self._items_dict[key].inherits_from
    #     if parent is not None and parent in input_coords:
    #         return parent
    #     # TODO: Also allow parents more than one level up.
    #     raise KeyError(f"Cannot find any available parent for {key}")

    def rebroadcast_xarray(self, input_array, output_coords, dim="Type"):
        if list(input_array.coords.values()) == list(output_coords):
            return input_array
        new_coords = {coord.name: coord for coord in input_array.coords.values()}
        new_coords[dim] = output_coords
        input_coords = input_array.coords[dim].values

        keep_coords = []
        new_array = xr.DataArray(0.0, dims=input_array.dims, coords=new_coords)
        for cur_coord in list(output_coords):
            if cur_coord in input_coords:
                keep_coords.append(cur_coord)
                continue
            parent = self.find_relations(input_coords, [cur_coord])[cur_coord][0]
            new_array.loc[{dim: cur_coord}] = input_array.loc[{dim: parent}]
        new_array.loc[{dim: keep_coords}] = input_array.loc[{dim: keep_coords}]
        return new_array

    def to_netcdf(self, out_fp, **kwargs):
        json_items = [x.to_dict() for x in self._items]
        data = xr.DataArray()
        data.attrs["knowledge_graph"] = json.dumps(json_items)
        data.to_netcdf(out_fp, **kwargs)

    @classmethod
    def from_dataarray(cls, data):
        # data = xr.open_dataarray(in_fp, **kwargs)
        json_str = data.attrs["knowledge_graph"]
        knowledge_list = json.loads(json_str)
        items = [Node.from_dict(item) for item in knowledge_list]
        return cls(*items)

    # def to_json_str(self):


def create_vehicle_graph():
    vehicle_subtypes = ["BEV", "FCV", "HEV", "ICE", "PHEV", "Trolley"]
    vehicle_supertypes = ["Cars", "Heavy Freight Trucks", "Light Commercial Vehicles",
                        "Medium Freight Trucks", "Regular Buses", "Midi Buses"]

    vehicle_knowledge_graph = KnowledgeGraph()
    vehicle_knowledge_graph.add(Node("Bikes"))
    vehicle_knowledge_graph.add(Node("Airplanes"))
    for subtype in ["Freight Planes", "Passenger Planes"]:
        vehicle_knowledge_graph.add(Node(subtype, inherits_from="Airplanes"))

    # Ships
    vehicle_knowledge_graph.add(Node("Ships"))
    for subtype in ["Small Ships", "Medium Ships", "Large Ships", "Very Large Ships",
                    "Inland Ships"]:
        vehicle_knowledge_graph.add(Node(subtype, inherits_from="Ships"))
    vehicle_knowledge_graph.add(Node("Trains"))

    # Trains
    for subtype in ["Freight Trains", "Trains", "High Speed Trains"]:
        vehicle_knowledge_graph.add(Node(subtype, inherits_from="Trains"))

    # Cars, Trucks, Buses
    vehicle_knowledge_graph.add(Node("Buses"))
    vehicle_knowledge_graph.add(Node("Trucks"))
    for super_type in vehicle_supertypes:
        if super_type in ["Heavy Freight Trucks", "Light Commercial Vehicles",
                          "Medium Freight Trucks"]:
            vehicle_knowledge_graph.add(Node(name=super_type, inherits_from="Trucks"))
        elif super_type in ["Regular Buses", "Midi Buses"]:
            vehicle_knowledge_graph.add(Node(name=super_type, inherits_from="Buses"))
        else:
            vehicle_knowledge_graph.add(Node(name=super_type))
        for sub_type in vehicle_subtypes:
            vehicle_knowledge_graph.add(Node(f"{super_type} - {sub_type}", inherits_from=super_type))
    return vehicle_knowledge_graph


def create_building_graph():
    building_nodes = []
    for super_type in ["Detached", "Semi-detached", "Appartment", "High-rise"]:
        building_nodes.append(Node(super_type))

        for sub_type in ["Urban", "Rural"]:
            building_nodes.append(Node(f"{super_type} - {sub_type}", inherits_from=super_type))


    building_knowledge_graph = KnowledgeGraph(*building_nodes)
    # knowledge_graph = KnowledgeGraph(*building_nodes)


    # building_knowledge_graph = KnowledgeGraph(
    #     Node("Detached"),
    #     Node("Detached - Urban", "Detached"),
    #     Node("Detached - Rural", "Detached"),
    #     Node("Detached"),
    #     Node("Detached - Urban", "Detached"),
    #     Node("Detached - Rural", "Detached"),
        
    # )
    return building_knowledge_graph

knowledge_graph = KnowledgeGraph(*create_building_graph()._items, *create_vehicle_graph()._items)
