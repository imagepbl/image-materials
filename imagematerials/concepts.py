from dataclasses import dataclass, field
from typing import Optional

import xarray as xr

@dataclass
class Node():
    name: str
    synonyms: list[str] = field(default_factory=list)
    inherits_from: Optional[str] = None


class KnowledgeGraph():
    def __init__(self):
        # nx.Graph()
        self._items: list[Node] = []
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
            if cur_out_coord in input_coords:
                relations[cur_out_coord] = [cur_out_coord]
                continue

            output_node = self[cur_out_coord]
            if set([output_node.name] + output_node.synonyms).intersection(input_coords):
                for syn in set([output_node.name] + output_node.synonyms):
                    if syn in input_coords:
                        relations[cur_out_coord] = [syn]
                        break
                continue
            try:
                ancestors = self._find_ancestors(output_node, input_coords)
            except KeyError:
                ancestors = []
            try:
                descendants = self._find_descendants(output_node, input_coords)
            except KeyError:
                descendants = []

            if len(ancestors) > 0 and len(descendants) > 0:
                raise ValueError(f"Cannot find relations, because {output_node} has both ancestors "
                                 "and descendants")
            if len(ancestors) + len(descendants) == 0:
                raise ValueError("Cannot find relations of {output_node} in input coordinates.")
            # if output_node.name in input_coords:
            relations[cur_out_coord] = ancestors + descendants
        return relations

    def find_relations_inverse(self, input_coords, output_coords):
        relations = self.find_relations(input_coords, output_coords)
        inverse_relations = {}
        for cur_output, cur_input_list in relations.items():
            inverse_relations.update({cur_input: cur_output for cur_input in cur_input_list})
        return inverse_relations

    def _find_ancestors(self, output_node, input_coords):
        if isinstance(output_node, str):
            output_node = self[output_node]
        if output_node.name in input_coords or set(output_node.synonyms).intersection(input_coords):
            return output_node.name
        if output_node.inherits_from is None:
            raise KeyError("Ancestor does not exist.")
        return self._find_ancestors(output_node.inherits_from, input_coords)

    def _find_descendants(self, output_node, input_coords):
        if isinstance(output_node, str):
            output_node = self[output_node]
        if output_node.name in input_coords or set(output_node.synonyms).intersection(input_coords):
            return [output_node.name]
        all_descendants = []
        for item in self._items:
            if item.inherits_from == output_node.name:
                all_descendants.extend(self._find_descendants(item, input_coords))
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

# building_nodes = []
# for supertype in ["Detached", "Semi-detached", "Appartment", "High-rise"]:
#     building_nodes.append(Node(supertype))

#     for subtype in ["Urban", "Rural"]:
#         building_nodes.append(Node(f"{supertype} - {subtype}", supertype))


# building_knowledge_graph = KnowledgeGraph(*building_nodes)
# knowledge_graph = KnowledgeGraph(*building_nodes, *vehicle_nodes)


# building_knowledge_graph = KnowledgeGraph(
#     Node("Detached"),
#     Node("Detached - Urban", "Detached"),
#     Node("Detached - Rural", "Detached"),
#     Node("Detached"),
#     Node("Detached - Urban", "Detached"),
#     Node("Detached - Rural", "Detached"),
    
# )