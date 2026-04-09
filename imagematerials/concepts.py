import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import prism
import xarray as xr


@dataclass
class Node():
    """Node for one concept in the knowledge graph.

    Raises
    ------
    ValueError
        When the name is the same as one of the synonyms or if the node inherits from itself.

    """

    name: str
    synonyms: list[str] = field(default_factory=list)
    inherits_from: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Do some post init validation checks."""
        if self.inherits_from is None:
            self.inherits_from = []
        elif isinstance(self.inherits_from, str):
            self.inherits_from = [self.inherits_from]

        if self.name in self.synonyms:
            raise ValueError(f"Cannot create node that has its own name as a synonym: {self}.")

        for parent in self.inherits_from:
            try:
                self.match_coords(parent)
                raise ValueError(f"Cannot create node that inherits from itself: {self}.")
            except KeyError:
                pass

    def match_coords(self, input_coords: Union[list[str], str]):
        """Find the node inside a list of coordinates.

        When the coordinates contain a synonym, that synonym is returned instead of the
        main name of the node.

        Parameters
        ----------
        input_coords
            List of coordinates to find the node in.

        Returns
        -------
            Name or synonym of the node that is in the input list of coordinates.

        Raises
        ------
        KeyError
            If the name or synonyms are not present in the input list of coordinates.

        """
        if isinstance(input_coords, str):
            input_coords = [input_coords]
        name_list = [self.name] + self.synonyms
        if not set(name_list).intersection(input_coords):
            raise KeyError()
        for name in name_list:
            if name in input_coords:
                return name

    def to_dict(self) -> str:
        """Serizalize the node to a dictionary.

        Returns
        -------
            A dictionary with the name synonyms and relationships to other concepts.

        """
        return {"name": self.name, "synonyms": self.synonyms,
                "inherits_from": self.inherits_from}

    @classmethod
    def from_dict(cls, node_dict: dict):
        """Create a node from a dictionary.

        Used to deserialize complete knowledge graphs.

        Parameters
        ----------
        node_dict
            Dictionary containing the name, synonyms and inheritance properties of the node.

        Returns
        -------
            A newly created node.

        """
        return cls(**node_dict)

class KnowledgeGraph():
    """Contains concepts and the relations between those concepts.

    Concepts are for example: vehicle, car, BEV car. The relationships between concepts are
    mostly the "is a" type of relations. For example a car is a vehicle would be denoted by
    the "inherits_from='vehicle'" attribute. Note that concepts can inherit from multiple sources.
    So, BEV could inherit from car and it can also inherit from has_battery for example. This can
    be useful if you want to aggregate over both vehicles and object that have batteries (which
    can include objects that are not vehicles of course).

    One of the goals of the knowledge graph is to facilitate easier computation when concept
    synonyms or concept inheritance is used inside of the data. For example, say that you have data
    for the average lifespan of cars, but not specifically for BEV cars. Then you can use the
    knowledge graph to use the lifespan of cars for all cars that do not have more specific data
    available. This feature is written for xarray DataArrays.

    The knowledge graph class is built progressively, so the order in which the nodes are added is
    important. You should add the root nodes (most basic concept such as "vehicle") first and then
    more detailed concepts. Note that you can't create cycles of A -> B -> C -> A with inheritance,
    since you will get an error message beforehand.

    Parameters
    ----------
    items:
        Nodes to add to the knowledge graph. Note that the order matters.
        A general way to build this knowledge graph is to initialize the knowledge graph without
        any items, but add them later one by one with the :meth:`add` method.

    Examples
    --------
    >>> KnowledgeGraph(Node("A"), Node("B", inherits_from="A"))

    """

    def __init__(self, *items):
        """Initialize the knowledge graph."""
        self._items: list[Node] = []
        for item in items:
            self.add(item)

    def add(self, node: Node):
        """Add new node to knowledge graph and validate it.

        Parameters
        ----------
        node
            New node to be added.

        Raises
        ------
        ValueError
            If the node already exists or the parent does not exist.

        """
        for name in [node.name] + node.synonyms:
            if name in self:
                raise ValueError(f"Node {name} already exists.")
        for parent in node.inherits_from:
            if parent not in self:
                raise ValueError(f"Parent {parent} of {node.name} does not exist.")

        self._items.append(node)

    def __contains__(self, node_name: str) -> bool:
        """Check whether a node_name already exists in the knowledge tree.

        Examples
        --------
        >>> kgraph = KnowledgeGraph(Node("a", synonyms=["b"]))
        >>> ("a" in kgraph, "b" in kgraph, "c" in kgraph)
        (True, True, False)

        """
        try:
            self[node_name]
        except KeyError:
            return False
        return True

    def __getitem__(self, node_name) -> Node:
        """Get a node with a name or synonym."""
        for item in self._items:
            if item.name == node_name:
                return item
            if node_name in item.synonyms:
                return item
        raise KeyError(f"Key {node_name} does not exist")

    def find_relations(self, input_coords: Sequence[str], output_coords) -> dict[str, str]:
        """Find the connections from the input coordinates to output coordinates.

        Parameters
        ----------
        input_coords:
            Coordinates in the xarray DataArray that you might want to transform.
        output_coords:
            Coordinates in the output xarray DataArray that you want to transform to.

        Examples
        --------
        >>> kg = KnowledgeGraph(Node("car", synonyms=["automobile"]))
        >>> kg.find_relations(["automobile"], ["car"])
        {
            "car": ["automobile"]
        }

        Returns
        -------
        relations:
            Dictionary containing keys for each output coordinate, and values for each input
            that is linked by synonym, parent or child node and is present in the list of
            input coordinates.

        """
        relations = {}
        for cur_out_coord in output_coords:
            relations[cur_out_coord] = self.find_one_relation(input_coords, cur_out_coord)
        return relations

    def find_one_relation(self, input_coords: list[str], output_name: str) -> list[str]:
        """Find a related object from a list of inputs.

        Differs from :meth:`KnowledgeGraph.find_relations` through finding the related inputs
        for one output coordinate, instead of multiple.

        Parameters
        ----------
        input_coords
            Coordinates in an xarray DataArray in which related concepts are attempted to be
            found for the output name.
        output_name
            Output name for which related terms are to be found.

        Returns
        -------
            A list of related terms to the output_name.

        Raises
        ------
        ValueError
            If the output_name has both ancestors and descendants in the input coordinates.
        ValueError
            If no known relations are known.


        Examples
        --------
        >>> kg = KnowledgeGraph(Node("vehicle", synonyms=["transport_object"])
        >>>                     Node("car", synonyms=["automobile"], inherits_from="vehicle"),
        >>>                     Node("bicycle", synonyms=["bike"], inherits_from="vehicle),
        >>> )
        >>> kg.find_one_relation(["car, "bike"], "bicycle")
        ["bike"]
        >>> kg.find_one_relation(["car", "bike"], "vehicle")
        ["car", "bike"]

        """
        if output_name in input_coords:
            return [output_name]

        output_node = self[output_name]
        try:
            return [output_node.match_coords(input_coords)]
        except KeyError:
            pass

        ancestors = self._find_ancestors(input_coords, output_node)

        try:
            descendants = self._find_descendants(input_coords, output_node)
        except KeyError:
            descendants = []

        if len(ancestors) > 0 and len(descendants) > 0:
            raise ValueError(f"Cannot find relations, because {output_node} has both ancestors "
                             f"({ancestors}) and descendants ({descendants})")
        if len(ancestors) + len(descendants) == 0:
            raise ValueError(f"Cannot find relations of {output_node} in input coordinates.")

        return list(set(ancestors + descendants))

    def find_relations_inverse(self, input_coords, output_coords):
        """Find which coordinates in the input match the coordinates in the output.

        Similar to :meth:`find_relations`, but inverse of that. Assumes that the input coordinates
        are unique.

        Parameters
        ----------
        input_coords
            Input coordinates for the xarray DataArray.
        output_coords
            Output coordinates for which the inputs need to be transformed.

        Returns
        -------
        inverse_relations:
            A dictionary with keys that input coordinates as keys, and lists of
            output coordinates as values.

        Examples
        --------
        >>> kg = KnowledgeGraph(Node("vehicle", synonyms=["transport_object"])
        >>>                     Node("car", synonyms=["automobile"], inherits_from="vehicle"),
        >>>                     Node("bicycle", synonyms=["bike"], inherits_from="vehicle),
        >>> )
        >>> kg.find_relations_inverse(["automobile", "bicycle", "vehicle"], ["car", "bicycle"])
        {"automobile": ["car"], "bicycle": ["vehicle"], "vehicle": ["car", "bicycle"]}

        """
        relations = self.find_relations(input_coords, output_coords)
        inverse_relations = {}
        for cur_output, cur_input_list in relations.items():
            inverse_relations.update({cur_input: cur_output for cur_input in cur_input_list})
        return inverse_relations

    def _find_ancestors(self, input_coords, output_node):
        """Find an ancestor/parent using recursion."""
        try:
            return [output_node.match_coords(input_coords)]
        except KeyError:
            ancestors = []
            for parent in output_node.inherits_from:
                try:
                    ancestors.extend(self._find_ancestors(input_coords, self[parent]))
                except KeyError:
                    pass
        return ancestors


    def _find_descendants(self, input_coords, output_node):
        """Find a descendant/child using recursion."""
        try:
            return [output_node.match_coords(input_coords)]
        except KeyError:
            pass

        all_descendants = []
        for item in self._items:
            if output_node.name in item.inherits_from:
                all_descendants.extend(self._find_descendants(input_coords, item))
        return all_descendants

    def rebroadcast_xarray(self, input_array, output_coords, dim="Type", shares=None,
                           dim_shares=None):
        """Disaggregate supertypes into subtypes.

        If shares for the subtypes are provided,
        the values of the input_array data is adjusted accordingly
        (value of supertype * shares = values of subtypes).
        If no shares are provided, the value of the supertype is taken for all subtypes.

        Parameters
        ----------
        input_array
            The xr.DataArray to be disaggregated.
        output_coords
            The output coordinates (subtypes).
        dim, optional
            Dimension of the input_array to disaggregate over, by default "Type"
        shares, optional
            Shares to be used for rebroadcasting, by default None. Must have the dimension "Type".
        dim_shares, optional
            Dimension of the shares DataArray. By default None. If shares is defined and
            dim_shares not, dim is used. In case the shares have a different dimension name than
            the input_array, it can be specified here.

        Returns
        -------
            Disaggregated xr.DataArray.

        """
        if shares is not None and dim_shares is None:
            dim_shares = dim

        input_coords = input_array.coords[dim].values
        if list(input_coords) == list(output_coords):
            return input_array
        new_coords = {coord.name: coord for coord in input_array.coords.values()}
        new_coords[dim] = output_coords

        keep_coords = []
        new_array = xr.DataArray(0.0, dims=input_array.dims, coords=new_coords)
        if input_array.pint.units:
            # check if input array had a unit and if so, reapply this unit to new array
            new_array = prism.Q_(new_array, input_array.pint.units)
        for cur_coord in list(output_coords):
            if cur_coord in input_coords:
                keep_coords.append(cur_coord)
                continue
            relations = self.find_relations(input_coords, [cur_coord])[cur_coord]
            if len(relations) > 1:
                raise ValueError("Cannot rebroadcast DataArray, because multiple input relations "
                                 f"exist ({relations}) for output coordinate {cur_coord}. For "
                                 "aggregation use the aggregate_sum method instead.")
            parent = relations[0]
            if shares is not None and cur_coord in shares.coords[dim_shares]:
                # disaggregate according to shares
                new_array.loc[{dim: cur_coord}] = (input_array.loc[{dim: parent}]
                                                   * shares.loc[{dim_shares: cur_coord}])
            else:
                # disaggregate, by just taking the value of the parent
                new_array.loc[{dim: cur_coord}] = input_array.loc[{dim: parent}]
        # Copy data for all coordinates that exist in both the input_array and output_coords
        # (`keep_coords`) -> only new coordinates are filled through disaggregation logic above
        new_array.loc[{dim: keep_coords}] = input_array.loc[{dim: keep_coords}]
        # Preserve metadata
        new_array.attrs = input_array.attrs.copy()
        new_array.name = input_array.name
        # new_array.encoding = input_array.encoding.copy() # do wee need this?

        return new_array

    def aggregate_sum(self, input_array, output_coords, dim="Type"):
        """Aggregate the data over subtypes and sums them.

        Parameters
        ----------
        input_array
            The xr.DataArray to be aggregated/summed.
        output_coords
            The output coordinates over which to aggregate/sum.
        dim, optional
            Dimension to aggregate/sum over, by default "Type"

        Returns
        -------
            Aggregated xr.DataArray.

        """
        output_to_input = self.find_relations(input_array.coords[dim].values, output_coords)
        output_arrays = []
        for output, input_list in output_to_input.items():
            output_arrays.append(input_array.sel(**{dim: input_list}).sum(dim))
        output_xr = xr.concat(output_arrays, dim=dim)
        output_xr.coords[dim] = output_coords
        return output_xr

    def to_netcdf(self, out_fp: Path|str, **kwargs):
        """Store the knowledge graph in a netCDF file as an attribute.

        The attribute will be called 'knowledge_graph' and is stored
        in the root of the netCDF file.

        Parameters
        ----------
        out_fp
            Output netCDF file.
        kwargs:
            Extra keyword arguments that are passed on to the xr.DataArray.to_netcdf() method.

        """
        json_items = [x.to_dict() for x in self._items]
        data = xr.DataArray()
        data.attrs["knowledge_graph"] = json.dumps(json_items)
        data.to_netcdf(out_fp, **kwargs)

    @classmethod
    def from_dataarray(cls, data: xr.DataArray|Path|str):
        """Create the knowledge graph from a xarray data array.

        Parameters
        ----------
        data
            xr.DataArray that contains an attribute called 'knowledge_graph' which is
            a serialized version of the knowledge graph.

        Returns
        -------
        knowledge_graph:
            An initialized knowledge graph.

        """
        json_str = data.attrs["knowledge_graph"]
        knowledge_list = json.loads(json_str)
        items = [Node.from_dict(item) for item in knowledge_list]
        return cls(*items)

    # def to_json_str(self):


def create_vehicle_graph():
    """Create the knowledge graph for the vehicle/transport sector."""
    vehicle_subtypes = ["BEV", "FCV", "HEV", "ICE", "PHEV", "Trolley"]
    vehicle_supertypes = ["Cars", "Heavy Freight Trucks", "Light Commercial Vehicles",
                        "Medium Freight Trucks", "Regular Buses", "Midi Buses"]

    vehicle_knowledge_graph = KnowledgeGraph(Node("Vehicles"))
    vehicle_knowledge_graph.add(Node("Bikes", inherits_from="Vehicles"))
    vehicle_knowledge_graph.add(Node("Airplanes", inherits_from="Vehicles"))
    for subtype in ["Freight Planes", "Passenger Planes"]:
        vehicle_knowledge_graph.add(Node(subtype, inherits_from="Airplanes"))

    # Ships
    vehicle_knowledge_graph.add(Node("Ships", inherits_from="Vehicles"))
    for subtype in ["Small Ships", "Medium Ships", "Large Ships", "Very Large Ships",
                    "Inland Ships"]:
        vehicle_knowledge_graph.add(Node(subtype, inherits_from="Ships"))
    #vehicle_knowledge_graph.add(Node("Trains", inherits_from="Vehicles"))

    # Trains
    # TODO: Fix Trains -> Regular Trains
    for subtype in ["Freight Trains", "Trains", "High Speed Trains"]:
        vehicle_knowledge_graph.add(Node(subtype, inherits_from="Vehicles"))

    # Cars, Trucks, Buses
    vehicle_knowledge_graph.add(Node("Buses", inherits_from="Vehicles"))
    vehicle_knowledge_graph.add(Node("Trucks", inherits_from="Vehicles"))
    for super_type in vehicle_supertypes:
        if super_type in ["Heavy Freight Trucks", "Light Commercial Vehicles",
                          "Medium Freight Trucks"]:
            vehicle_knowledge_graph.add(Node(name=super_type, inherits_from="Trucks"))
        elif super_type in ["Regular Buses", "Midi Buses"]:
            vehicle_knowledge_graph.add(Node(name=super_type, inherits_from="Buses"))
        else:
            vehicle_knowledge_graph.add(Node(name=super_type))
        for sub_type in vehicle_subtypes:
            vehicle_knowledge_graph.add(
                Node(f"{super_type} - {sub_type}", inherits_from=super_type))
    return vehicle_knowledge_graph


def create_building_graph():
    """Create the knowledge graph for the buildings sector."""
    building_nodes = [
    Node("Buildings"),

]
    for super_type in ["Detached", "Semi-detached", "Appartment", "High-rise"]:
        building_nodes.append(Node(super_type, inherits_from="Buildings"))

        for sub_type in ["Urban", "Rural"]:
            building_nodes.append(Node(f"{super_type} - {sub_type}", inherits_from=super_type))

    for c in ["Retail+", "Hotels+", "Office", "Govt+"]:
        building_nodes.append(Node(c, inherits_from="Buildings"))

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


def create_class_region_graph():
    #TODO move to seperate file
    
    class_region_knowledge_graph = KnowledgeGraph()

    # Add numeric region nodes first with full synonyms
    numeric_region_map = {
        "class_ 1": ["Canada", "CAN"],
        "class_ 2": ["USA", "US"],
        "class_ 3": ["Mexico", "MEX"],
        "class_ 4": ["Rest of Central America", "RCAM", "Rest C.Am."],
        "class_ 5": ["Brazil", "BRA"],
        "class_ 6": ["Rest of South America", "RSAM", "Rest S.Am."],
        "class_ 7": ["Northern Africa", "NAF", "N.Africa"],
        "class_ 8": ["Western Africa", "WAF", "W.Africa"],
        "class_ 9": ["Eastern Africa", "EAF", "E.Africa"],
        "class_ 10": ["South Africa", "SAF"],
        "class_ 11": ["Western Europe", "WEU", "W.Europe"],
        "class_ 12": ["Central Europe", "CEU", "C.Europe"],
        "class_ 13": ["Turkey", "TUR"],
        "class_ 14": ["Ukraine +", "UKR", "Ukraine"],
        "class_ 15": ["Asian-Stan", "STAN", "Stan"],
        "class_ 16": ["Russia +", "RUS", "Russia"],
        "class_ 17": ["Middle East", "ME", "M.East"],
        "class_ 18": ["India +", "INDIA", "India"],
        "class_ 19": ["Korea", "KOR"],
        "class_ 20": ["China +", "CHN", "China"],
        "class_ 21": ["Southeastern Asia", "SEAS", "SE.Asia"],
        "class_ 22": ["Indonesia +", "INDO", "Indonesia"],
        "class_ 23": ["Japan", "JAP"],
        "class_ 24": ["Oceania", "OCE"],
        "class_ 25": ["Rest of South Asia", "RSAS", "Rest S.Asia"],
        "class_ 26": ["Rest of Southern Africa", "RSAF", "Rest S.Africa"]
    }

    for number, synonyms in numeric_region_map.items():
        inherits_from = None
        class_region_knowledge_graph.add(
            Node(number, synonyms=synonyms, inherits_from=inherits_from))

    return class_region_knowledge_graph

def create_image_region_graph():
    #TODO move to seperate file
    
    image_region_knowledge_graph = KnowledgeGraph()

    # --- 1. Define IMAGE regions  ---
    numeric_region_map = {
        "region_1": ["CAN", "Canada region"],
        "region_2": ["USA", "US region"],
        "region_3": ["MEX", "Mexico region"],
        "region_4": ["RCAM", "Rest of Central America","Rest C.Am."],
        "region_5": ["BRA", "Brazil region"],
        "region_6": ["RSAM", "Rest of South America", "Rest S.Am."],
        "region_7": ["NAF", "Northern Africa", "N.Africa"],
        "region_8": ["WAF", "Western Africa", "W.Africa"],
        "region_9": ["EAF", "Eastern Africa", "E.Africa"],
        "region_10": ["SAF", "South Africa"],
        "region_11": ["WEU", "Western Europe", "W.Europe"],
        "region_12": ["CEU", "Central Europe", "C.Europe"],
        "region_13": ["TUR", "Turkey"],
        "region_14": ["UKR", "Ukraine", "Ukraine +", "Ukraine region"],
        "region_15": ["STAN", "Asian-Stan", "Central Asia"],
        "region_16": ["RUS", "Russia +", "Russia", "Russia region"],
        "region_17": ["ME", "Middle East", "M.East"],
        "region_18": ["INDIA", "India +", "India", "India region"],
        "region_19": ["KOR", "Korea", "Korea region"],
        "region_20": ["CHN", "China +", "China", "China region"],
        "region_21": ["SEAS", "Southeastern Asia", "SE.Asia"],
        "region_22": ["INDO", "Indonesia +", "Indonesia"],
        "region_23": ["JAP", "Japan"],
        "region_24": ["OCE", "Oceania"],
        "region_25": ["RSAS", "Rest of South Asia", "Rest S.Asia"],
        "region_26": ["RSAF", "Rest of Southern Africa", "Rest S.Africa"]
    }

    # Add region nodes
    for region_number, synonyms in numeric_region_map.items():
        image_region_knowledge_graph.add(Node(region_number, synonyms=synonyms, inherits_from=None))

    # --- 2. Countries: ISO → synonyms  ---
    iso_region_map = {
        # Canada (1)
        "124": ["Canada"],

        # USA (2)
        "666": ["St. Pierre and Miquelon"],
        "840": ["USA", "United States", "US"],

        # Mexico (3)
        "484": ["Mexico"],

        # Central America (4)
        "44": ["Bahamas, The", "Bahamas", "The Bahamas"],
        "52": ["Barbados"],
        "60": ["Bermuda"],
        "84": ["Belize"],
        "92": ["Virgin Islands (British)", "Virgin Isl. (Br.)"],
        "136": ["Cayman Islands"],
        "188": ["Costa Rica"],
        "212": ["Dominica"],
        "214": ["Dominican Republic"],
        "222": ["El Salvador"],
        "308": ["Grenada"],
        "312": ["Guadeloupe"],
        "320": ["Guatemala"],
        "332": ["Haiti"],
        "340": ["Honduras"],
        "388": ["Jamaica"],
        "474": ["Martinique"],
        "500": ["Montserrat"],
        "533": ["Aruba"],
        "530": ["Netherlands Antilles"],
        "558": ["Nicaragua"],
        "591": ["Panama"],
        "630": ["Puerto Rico"],
        "659": ["St. Kitts and Nevis"],
        "660": ["Anguilla"],
        "662": ["St. Lucia"],
        "670": ["St. Vincent and the Grenadines"],
        "780": ["Trinidad and Tobago"],
        "796": ["Turks and Caicos Islands", "Turks and Caicos Isl."],
        "850": ["Virgin Islands (U.S.)"],

        # Brazil (5)
        "76": ["Brazil"],

        # Rest of South America (6)
        "32": ["Argentina"],
        "68": ["Bolivia"],
        "152": ["Chile"],
        "170": ["Colombia"],
        "218": ["Ecuador"],
        "238": ["Falkland Islands", "Falklands Isl."],
        "254": ["French Guiana", "French Guyana"],
        "328": ["Guyana"],
        "600": ["Paraguay"],
        "604": ["Peru"],
        "740": ["Suriname"],
        "858": ["Uruguay"],
        "862": ["Venezuela", "Bolivarian Republic of Venezuela", "Venezuela, RB"],

        # Northern Africa (7)
        "12": ["Algeria"],
        "434": ["Libya"],
        "504": ["Morocco"],
        "732": ["Western Sahara"],
        "788": ["Tunisia"],
        "818": ["Egypt", "Egypt, Arab Rep.", "Arab Republic of Egypt"],

        # Western Africa (8)
        "120": ["Cameroon"],
        "132": ["Cape Verde"],
        "140": ["Central African Republic"],
        "148": ["Chad"],
        "178": ["Congo", "Congo, Rep.", "Republic of the Congo", "Congo Republic"],
        "180": ["Congo (Democratic Republic)", "Congo, Dem. Rep.", "Democratic Republic of Congo", "DRC", "D.R.C."],
        "204": ["Benin"],
        "226": ["Equatorial Guinea"],
        "266": ["Gabon"],
        "270": ["Gambia, The", "Gambia"],
        "288": ["Ghana"],
        "324": ["Guinea"],
        "384": ["Cote d'Ivoire", "Ivory Coast"],
        "430": ["Liberia"],
        "466": ["Mali"],
        "478": ["Mauritania"],
        "562": ["Niger"],
        "566": ["Nigeria"],
        "624": ["Guinea-Bissau"],
        "654": ["St. Helena"],
        "678": ["Sao Tome and Principe"],
        "686": ["Senegal"],
        "694": ["Sierra Leone"],
        "768": ["Togo"],
        "854": ["Burkina Faso"],

        # Eastern Africa (9)
        "108": ["Burundi"],
        "174": ["Comoros"],
        "231": ["Ethiopia"],
        "232": ["Eritrea"],
        "262": ["Djibouti"],
        "404": ["Kenya"],
        "450": ["Madagascar"],
        "480": ["Mauritius"],
        "638": ["Reunion"],
        "646": ["Rwanda"],
        "690": ["Seychelles"],
        "706": ["Somalia"],
        "736": ["Sudan"],
        "800": ["Uganda"],

        # South Africa (10)
        "710": ["South Africa"],

        # Western Europe (11)
        "20": ["Andorra"],
        "40": ["Austria"],
        "56": ["Belgium"],
        "208": ["Denmark"],
        "234": ["Faroe Islands"],
        "246": ["Finland"],
        "250": ["France"],
        "276": ["Germany"],
        "292": ["Gibraltar"],
        "300": ["Greece"],
        "336": ["Vatican City", "Vatican City State"],
        "352": ["Iceland"],
        "372": ["Ireland"],
        "380": ["Italy"],
        "438": ["Liechtenstein"],
        "442": ["Luxembourg"],
        "470": ["Malta"],
        "492": ["Monaco"],
        "528": ["Netherlands", "The Netherlands"],
        "578": ["Norway"],
        "620": ["Portugal"],
        "674": ["San Marino"],
        "724": ["Spain"],
        "752": ["Sweden"],
        "756": ["Switzerland"],
        "826": ["United Kingdom", "UK"],

        # Central Europe (12)
        "8": ["Albania"],
        "70": ["Bosnia and Herzegovina"],
        "100": ["Bulgaria"],
        "191": ["Croatia"],
        "196": ["Cyprus"],
        "203": ["Czech Republic"],
        "233": ["Estonia"],
        "348": ["Hungary"],
        "428": ["Latvia"],
        "440": ["Lithuania"],
        "616": ["Poland"],
        "642": ["Romania"],
        "703": ["Slovakia", "Slovak Republic"],
        "705": ["Slovenia"],
        "807": ["North Macedonia"],
        "891": ["Serbia and Montenegro"],

        # Turkey (13)
        "792": ["Turkey", "Türkiye", "Republic of Türkiye"],

        # Ukraine region (14)
        "112": ["Belarus"],
        "498": ["Moldova"],
        "804": ["Ukraine"],

        # Central Asia (15)
        "398": ["Kazakhstan"],
        "417": ["Kyrgyz Republic"],
        "762": ["Tajikistan"],
        "795": ["Turkmenistan"],
        "860": ["Uzbekistan"],

        # Russia region (16)
        "31": ["Azerbaijan"],
        "51": ["Armenia"],
        "268": ["Georgia"],
        "643": ["Russia", "Russian Federation"],

        # Middle East (17)
        "48": ["Bahrain"],
        "364": ["Iran", "Iran, Islamic Rep.", "Islamic Republic of Iran"],
        "368": ["Iraq"],
        "376": ["Israel"],
        "400": ["Jordan"],
        "414": ["Kuwait"],
        "422": ["Lebanon"],
        "512": ["Oman"],
        "634": ["Qatar"],
        "682": ["Saudi Arabia"],
        "760": ["Syria", "Syrian Arab Republic"],
        "784": ["United Arab Emirates", "UAE"],
        "887": ["Yemen", "Yemen, Rep.", "Republic of Yemen"],

        # India (18)
        "356": ["India"],

        # Korea region (19)
        "408": ["North Korea", "Korea, Dem. Rep.", "Democratic People's Republic of Korea", "DPRK"],
        "410": ["South Korea", "Republic of Korea", "Korea, Rep."],

        # China region (20)
        "156": ["China"],
        "158": ["Taiwan"],
        "344": ["Hong Kong"],
        "446": ["Macao"],
        "496": ["Mongolia"],

        # Southeastern Asia (21)
        "96": ["Brunei"],
        "104": ["Myanmar"],
        "116": ["Cambodia"],
        "418": ["Laos", "Lao PDR", "Lao People's Democratic Republic"],
        "458": ["Malaysia"],
        "608": ["Philippines"],
        "702": ["Singapore"],
        "704": ["Vietnam", "Viet Nam"],
        "764": ["Thailand"],

        # Indonesia region (22)
        "360": ["Indonesia"],
        "598": ["Papua New Guinea"],
        "626": ["Timor-Leste", "East Timor", "Democratic Republic of Timor-Leste"],

        # Japan (23)
        "392": ["Japan"],

        # Oceania (24)
        "16": ["American Samoa"],
        "36": ["Australia"],
        "90": ["Solomon Islands"],
        "184": ["Cook Islands", "Cook Isl."],
        "242": ["Fiji"],
        "258": ["French Polynesia"],
        "296": ["Kiribati"],
        "520": ["Nauru"],
        "540": ["New Caledonia"],
        "548": ["Vanuatu"],
        "554": ["New Zealand"],
        "570": ["Niue"],
        "580": ["Northern Mariana Islands"],
        "583": ["Micronesia", "Micronesia, Fed. Sts.", "Federated States of Micronesia"],
        "584": ["Marshall Islands"],
        "585": ["Palau"],
        "612": ["Pitcairn"],
        "772": ["Tokelau"],
        "776": ["Tonga"],
        "798": ["Tuvalu"],
        "876": ["Wallis and Futuna", "Territory of the Wallis and Futuna Islands"],
        "882": ["Samoa"],

        # Rest of South Asia (25)
        "4": ["Afghanistan"],
        "50": ["Bangladesh"],
        "64": ["Bhutan"],
        "144": ["Sri Lanka"],
        "462": ["Maldives"],
        "524": ["Nepal"],
        "586": ["Pakistan"],

        # Rest of Southern Africa (26)
        "24": ["Angola"],
        "72": ["Botswana"],
        "426": ["Lesotho"],
        "454": ["Malawi"],
        "508": ["Mozambique"],
        "516": ["Namibia"],
        "716": ["Zimbabwe"],
        "748": ["Eswatini", "Swaziland"],
        "834": ["Tanzania"],
        "894": ["Zambia"],
    }

    # --- 3. ISO → region mapping ---
    iso_to_class = {
        # Canada
        "124": "region_1",
        # USA
        "840": "region_2", "666": "region_2",
        # Mexico
        "484": "region_3",
        # Central America
        "44": "region_4", "52": "region_4", "60": "region_4", "84": "region_4", "92": "region_4", 
        "136": "region_4", "188": "region_4", "212": "region_4", "214": "region_4", "222": "region_4",
        "308": "region_4", "312": "region_4", "320": "region_4", "332": "region_4", "340": "region_4",
        "388": "region_4", "474": "region_4", "500": "region_4", "533": "region_4", "530": "region_4", 
        "558": "region_4", "591": "region_4", "630": "region_4", "659": "region_4", "660": "region_4",
        "662": "region_4", "670": "region_4", "780": "region_4", "796": "region_4", "850": "region_4",
        # Brazil
        "76": "region_5",
        # Rest South America
        "32": "region_6", "68": "region_6", "152": "region_6", "170": "region_6", "218": "region_6",
        "238": "region_6", "254": "region_6", "328": "region_6", "600": "region_6", "604": "region_6",
        "740": "region_6", "858": "region_6", "862": "region_6",
        # Northern Africa
        "12": "region_7", "434": "region_7", "504": "region_7", "732": "region_7", "788": "region_7", 
        "818": "region_7",
        # Western Africa
        "120": "region_8", "132": "region_8", "140": "region_8", "148": "region_8", "178": "region_8",
        "180": "region_8", "204": "region_8", "226": "region_8", "266": "region_8", "270": "region_8",
        "288": "region_8", "324": "region_8", "384": "region_8", "430": "region_8", "466": "region_8", 
        "478": "region_8", "562": "region_8", "566": "region_8", "624": "region_8", "654": "region_8",
        "678": "region_8", "686": "region_8", "694": "region_8", "768": "region_8", "854": "region_8",
        # Eastern Africa
        "108": "region_9", "174": "region_9", "231": "region_9", "232": "region_9", "262": "region_9",
        "404": "region_9", "450": "region_9", "480": "region_9", "638": "region_9", "646": "region_9", 
        "690": "region_9", "706": "region_9", "736": "region_9", "800": "region_9",
        # South Africa
        "710": "region_10",
        # Western Europe
        "20": "region_11", "40": "region_11", "56": "region_11", "208": "region_11", "234": "region_11",
        "246": "region_11", "250": "region_11", "276": "region_11", "292": "region_11", "300": "region_11",
        "336": "region_11", "352": "region_11", "372": "region_11", "380": "region_11", "438": "region_11", 
        "442": "region_11", "470": "region_11", "492": "region_11", "528": "region_11", "578": "region_11", 
        "620": "region_11", "674": "region_11", "724": "region_11", "752": "region_11", "756": "region_11",
        "826": "region_11",
        # Central Europe
        "8": "region_12", "70": "region_12", "100": "region_12", "191": "region_12", "196": "region_12",
        "203": "region_12", "233": "region_12", "348": "region_12", "428": "region_12", "440": "region_12",
        "616": "region_12", "642": "region_12", "703": "region_12", "705": "region_12", "807": "region_12", 
        "891": "region_12",
        # Turkey
        "792": "region_13",
        # Ukraine region
        "804": "region_14", "112": "region_14", "498": "region_14",
        # Central Asia
        "398": "region_15", "417": "region_15", "762": "region_15", "795": "region_15", "860": "region_15",
        # Russia region
        "31": "region_16", "51": "region_16", "268": "region_16", "643": "region_16",
        # Middle East
        "48": "region_17", "364": "region_17", "368": "region_17", "376": "region_17", "400": "region_17",
        "414": "region_17", "422": "region_17", "512": "region_17", "634": "region_17", "682": "region_17",
        "760": "region_17", "784": "region_17", "887": "region_17",
        # India
        "356": "region_18",
        # Korea
        "410": "region_19", "408": "region_19",
        # China region
        "156": "region_20", "158": "region_20", "344": "region_20", "446": "region_20", "496": "region_20",
        # Southeastern Asia
        "96": "region_21", "104": "region_21", "116": "region_21", "418": "region_21", "458": "region_21",
        "608": "region_21", "702": "region_21", "704": "region_21", "764": "region_21",
        # Indonesia region
        "360": "region_22", "598": "region_22", "626": "region_22",
        # Japan
        "392": "region_23",
        # Oceania
        "16": "region_24", "36": "region_24", "90": "region_24", "184": "region_24", "242": "region_24", 
        "258": "region_24", "296": "region_24", "520": "region_24", "540": "region_24", "548": "region_24",
        "554": "region_24", "570": "region_24", "580": "region_24", "583": "region_24", "584": "region_24", 
        "585": "region_24", "612": "region_24", "772": "region_24", "776": "region_24", "798": "region_24", 
        "876": "region_24", "882": "region_24",

        # Rest of South Asia
        "4": "region_25", "50": "region_25", "64": "region_25", "144": "region_25", "462": "region_25", 
        "524": "region_25", "586": "region_25", 

        # Rest of Southern Africa
        "24": "region_26", "72": "region_26", "426": "region_26", "454": "region_26", "508": "region_26", 
        "516": "region_26", "716": "region_26", "748": "region_26", "834": "region_26", "894": "region_26",
    }

    # --- 4. Add country nodes ---
    for iso, synonyms in iso_region_map.items():
        parent = iso_to_class.get(iso)
        image_region_knowledge_graph.add(Node(iso, synonyms=synonyms, inherits_from=parent))

    return image_region_knowledge_graph


def create_region_graph():
    #TODO move to seperate file

    region_knowledge_graph = KnowledgeGraph()

    # Add target regions as main nodes (non-numeric)
    target_regions = [
        "Latin America", "Middle East and Northern Africa",
        "Other Asia", "Other OECD", "Reforming Economies", "Subsaharan Africa"
    ]
    for region in target_regions:
        region_knowledge_graph.add(Node(region))

    # Add numeric region nodes first with full synonyms
    numeric_region_map = {
        "1": ["Canada", "CAN"],
        "2": ["USA", "US"],
        "3": ["Mexico", "MEX"],
        "4": ["Rest of Central America", "RCAM", "Rest C.Am."],
        "5": ["Brazil", "BRA"],
        "6": ["Rest of South America", "RSAM", "Rest S.Am."],
        "7": ["Northern Africa", "NAF", "N.Africa"],
        "8": ["Western Africa", "WAF", "W.Africa"],
        "9": ["Eastern Africa", "EAF", "E.Africa"],
        "10": ["South Africa", "SAF"],
        "11": ["Western Europe", "WEU", "W.Europe"],
        "12": ["Central Europe", "CEU", "C.Europe"],
        "13": ["Turkey", "TUR"],
        "14": ["Ukraine +", "UKR", "Ukraine"],
        "15": ["Asian-Stan", "STAN", "Stan"],
        "16": ["Russia +", "RUS", "Russia"],
        "17": ["Middle East", "ME", "M.East"],
        "18": ["India +", "INDIA", "India"],
        "19": ["Korea", "KOR"],
        "20": ["China +", "CHN", "China"],
        "21": ["Southeastern Asia", "SEAS", "SE.Asia"],
        "22": ["Indonesia +", "INDO", "Indonesia"],
        "23": ["Japan", "JAP"],
        "24": ["Oceania", "OCE"],
        "25": ["Rest of South Asia", "RSAS", "Rest S.Asia"],
        "26": ["Rest of Southern Africa", "RSAF", "Rest S.Africa"]
    }

    subregion_division = {
        "Latin America": {"RCAM", "BRA", "RSAM"},
        "Middle East and Northern Africa": {"NAF", "ME"},
        "Subsaharan Africa": {"WAF", "EAF", "SAF", "RSAF"},
        "Reforming Economies": {"UKR", "STAN", "RUS"},
        "Other Asia": {"KOR", "SEAS", "INDO", "RSAS"},
        "Other OECD": {"TUR", "OCE", "MEX"},
    }

    for number, synonyms in numeric_region_map.items():
        inherits_from = None
        for super_region, regions in subregion_division.items():
            if len(regions.intersection(synonyms)) > 0:
                inherits_from = super_region
                break
        region_knowledge_graph.add(Node(number, synonyms=synonyms, inherits_from=inherits_from))

    return region_knowledge_graph

knowledge_graph = KnowledgeGraph(*create_building_graph()._items, *create_vehicle_graph()._items)


def create_class_region_graph():
    #TODO move to seperate file
    
    region_knowledge_graph = KnowledgeGraph()

    # Add target regions as main nodes (non-numeric)
    target_regions = [
        "Latin America", "Middle East and Northern Africa",
        "Other Asia", "Other OECD", "Reforming Economies", "Subsaharan Africa"
    ]
    for region in target_regions:
        region_knowledge_graph.add(Node(region))

    # Add numeric region nodes first with full synonyms
    numeric_region_map = {
        "class_ 1": ["Canada", "CAN"],
        "class_ 2": ["USA", "US"],
        "class_ 3": ["Mexico", "MEX"],
        "class_ 4": ["Rest of Central America", "RCAM", "Rest C.Am."],
        "class_ 5": ["Brazil", "BRA"],
        "class_ 6": ["Rest of South America", "RSAM", "Rest S.Am."],
        "class_ 7": ["Northern Africa", "NAF", "N.Africa"],
        "class_ 8": ["Western Africa", "WAF", "W.Africa"],
        "class_ 9": ["Eastern Africa", "EAF", "E.Africa"],
        "class_ 10": ["South Africa", "SAF"],
        "class_ 11": ["Western Europe", "WEU", "W.Europe"],
        "class_ 12": ["Central Europe", "CEU", "C.Europe"],
        "class_ 13": ["Turkey", "TUR"],
        "class_ 14": ["Ukraine +", "UKR", "Ukraine"],
        "class_ 15": ["Asian-Stan", "STAN", "Stan"],
        "class_ 16": ["Russia +", "RUS", "Russia"],
        "class_ 17": ["Middle East", "ME", "M.East"],
        "class_ 18": ["India +", "INDIA", "India"],
        "class_ 19": ["Korea", "KOR"],
        "class_ 20": ["China +", "CHN", "China"],
        "class_ 21": ["Southeastern Asia", "SEAS", "SE.Asia"],
        "class_ 22": ["Indonesia +", "INDO", "Indonesia"],
        "class_ 23": ["Japan", "JAP"],
        "class_ 24": ["Oceania", "OCE"],
        "class_ 25": ["Rest of South Asia", "RSAS", "Rest S.Asia"],
        "class_ 26": ["Rest of Southern Africa", "RSAF", "Rest S.Africa"]
    }

    subregion_division = {
        "Latin America": {"RCAM", "BRA", "RSAM"},
        "Middle East and Northern Africa": {"NAF", "ME"},
        "Subsaharan Africa": {"WAF", "EAF", "SAF", "RSAF"},
        "Reforming Economies": {"UKR", "STAN", "RUS"},
        "Other Asia": {"KOR", "SEAS", "INDO", "RSAS"},
        "Other OECD": {"TUR", "OCE", "MEX"},
    }

    for number, synonyms in numeric_region_map.items():
        inherits_from = None
        for super_region, regions in subregion_division.items():
            if len(regions.intersection(synonyms)) > 0:
                inherits_from = super_region
                break
        region_knowledge_graph.add(Node(number, synonyms=synonyms, inherits_from=inherits_from))

    return region_knowledge_graph


def create_electricity_graph():
    """
    Constructs and returns a hierarchical knowledge graph representing the electricity system for the subsystems:
    generation, transmission, and storage components.

    The function builds a structured ontology using `KnowledgeGraph` and `Node` objects to
    define relationships between electricity-related entities. It organizes technologies and
    infrastructure into supertype and subtype categories.

    Structure:

    Electricity
    ├── Generation
    │   ├── Solar PV
    │   ├── Conv. Coal
    │   ├── ...
    │   └── CHP Biomass + CCS
    │
    ├── Transmission
    │   ├── High Voltage (HV)
    │   │   ├── Lines
    │   │   │   ├── Overhead
    │   │   │   └── Underground
    │   │   ├── Substations
    │   │   └── Transformers
    │   ├── Medium Voltage (MV)
    │   │   ├── ...
    │   └── Low Voltage (LV)
    │       ├── ...
    │
    └── Storage
        ├── PHS
        ├── V2G-Batteries
        └── Other Storage
            ├── mechanical storage
            │   ├── Flywheel
            │   └── Compressed Air
            ├── lithium batteries
            │   ├── LMO
            │   ├── ...
            │   └── Lithium-air
            ├── molten salt and flow batteries
            │   ├── Zinc-Bromide
            │   ├── ...
            └── other
                ├── Hydrogen FC
                ├── NiMH
                └── Deep-cycle Lead-Acid


    Returns:
        KnowledgeGraph: A fully constructed knowledge graph object describing the
        electricity system hierarchy.

    Notes:
        - `V2G-Batteries` (Vehicle-to-Grid) could later be linked to vehicle systems.
        - The graph currently does not include sub-technologies (e.g. different PV techologies) for generation types, but
          placeholders are present for future expansion.
    """

    # Generation ======================================================================================
    
    numeric_generation_map = {
        "1": ["Solar PV", "SPV"],               # Solar PV power (central)
        "2": ["Solar PV residential", "SPVR"],  # Solar PV power (decentral/residential)
        "3": ["CSP"],                           # Concentrated Solar Power 
        "4": ["Wind onshore", "WON"],           # Onshore wind power
        "5": ["Wind offshore", "WOFF"],         # Offshore wind power
        "6": ["Wave", "WAVE"],                  # Wave power
        "7": ["Hydro", "HYD"],                  # Hydro power
        "8": ["Other Renewables", "OREN"],      # Other renewables (tidal and geothermal power)
        "9": ["Geothermal", "GEO"],             # Geothermal power
        "10": ["Hydrogen power", "H2P"],        # Hydrogen to power
        "11": ["Nuclear", "NUC"],               # Nuclear
        "12": ["<EMPTY>", "FREE12"],            # Free spot
        "13": ["Conv. Coal", "ClST"],           # Coal steam turbine
        "14": ["Conv. Oil", "OlST"],            # Oil steam turbine
        "15": ["Conv. Natural Gas", "NGOT"],    # NG open cycle turbine
        "16": ["Waste", "BioST"],               # Biomass steam turbine
        "17": ["IGCC"],                         # Integrated gasification combined cycle
        "18": ["OGCC", "OlCC"],                 # Oil combined cycle
        "19": ["NG CC", "NGCC"],                # NG combined cycle
        "20": ["Biomass CC", "BioCC"],          # Biomass combined cycle
        "21": ["Coal + CCS", "ClCS"],           # Coal carbon capture and storage
        "22": ["Oil/Coal + CCS", "OlCS"],       # Oil carbon capture and storage
        "23": ["Natural Gas + CCS", "NGCS"],    # NG carbon capture and storage
        "24": ["Biomass + CCS", "BioCS"],       # Biomass carbon capture and storage
        "25": ["CHP Coal", "ClCHP"],            # Coal combined heat and power
        "26": ["CHP Oil", "OlCHP"],             # Oil combined heat and power
        "27": ["CHP Natural Gas", "NGCHP"],     # NG combined heat and power
        "28": ["CHP Biomass", "BioCHP"],        # Biomass combined heat and power
        "29": ["CHP Coal + CCS", "ClCHPCS"],    # Coal combined heat and power carbon capture and storage
        "30": ["CHP Oil + CCS", "OlCHPCS"],     # Oil combined heat and power carbon capture and storage
        "31": ["CHP Natural Gas + CCS", "NGCHPCS"], # NG combined heat and power carbon capture and storage
        "32": ["CHP Biomass + CCS", "BioCHPCS"],    # Biomass combined heat and power carbon capture and storage
        "33": ["CHP Geothermal", "GeoCHP"],         # Geothermal combined heat and power
        "34": ["CHP Hydrogen", "H2CHP"]             # Hydrogen combined heat and power
    }


    electricity_knowledge_graph = KnowledgeGraph(Node("Electricity"))
    electricity_knowledge_graph.add(Node("Generation", inherits_from="Electricity"))
    for number, synonyms in numeric_generation_map.items():
        electricity_knowledge_graph.add(Node(number, synonyms=synonyms, inherits_from="Generation"))

    # # Sub-Technologies --------------------------

    # for subtype in ["c-Si", "a-Si", "CIGS", "CdTe", "Perovskite"]:
    #     electricity_knowledge_graph.add(Node(subtype, inherits_from="Solar PV"))
    #     electricity_knowledge_graph.add(Node(subtype, inherits_from="Solar PV residential")) # is this possible?

    # for subtype in ["Fresnel Reflector", "Central Receiver", "Parabolic Trough", "Parabolic Dish"]:
    #     electricity_knowledge_graph.add(Node(subtype, inherits_from="CSP"))

    # for subtype in ["Geared - High Speed", "Geared - Medium speed", "Direct Drive"]:
    #     electricity_knowledge_graph.add(Node(subtype, inherits_from="Wind onshore"))
    #     electricity_knowledge_graph.add(Node(subtype, inherits_from="Wind offshore"))

    # # Category Relations -------------------------

    # # Renewables
    # electricity_knowledge_graph.add(Node("Renewables", inherits_from="Generation"))
    # for supertype in ["Solar PV", "Solar PV residential", "CSP", "Wind onshore", "Wind offshore",
    #                   "Wave", "Other Renewables", "Geothermal", "Biomass CC","Biomass + CCS",
    #                   "CHP Biomass","CHP Geothermal", "CHP Biomass + CCS"]:
    #     electricity_knowledge_graph.add(Node(supertype, inherits_from="Renewables"))

    # # Climate Neutral
    # electricity_knowledge_graph.add(Node("Climate Neutral", inherits_from="Generation"))
    # for supertype in ["Solar PV", "Solar PV residential", "CSP", "Wind onshore", "Wind offshore",
    #                   "Wave", "Other Renewables", "Geothermal", "Biomass CC","Biomass + CCS",
    #                   "CHP Biomass","CHP Geothermal", "CHP Biomass + CCS","Coal + CCS", "Oil/Coal + CCS", 
    #                   "Natural Gas + CCS", "Nuclear"]:
    #     electricity_knowledge_graph.add(Node(supertype, inherits_from="Climate Neutral"))

    # # Fossil
    # electricity_knowledge_graph.add(Node("Fossil", inherits_from="Generation"))
    # for supertype in ["Conv. Coal", "Conv. Oil", "Conv. Natural Gas"]:
    #     electricity_knowledge_graph.add(Node(supertype, inherits_from="Fossil"))


    # Transmission Grid ============================================================================
    transmission_types = ["HV", "MV", "LV"]
    transmission_subtypes = ["Lines", "Substations", "Transformers"]
    line_subtypes = ["Overhead", "Underground"]
  
    electricity_knowledge_graph.add(Node("Transmission", inherits_from="Electricity"))
    for t_type in transmission_types:
        electricity_knowledge_graph.add(Node(t_type, inherits_from="Transmission"))
        for sub_type in transmission_subtypes:
            electricity_knowledge_graph.add(Node(f"{t_type} - {sub_type}", inherits_from=t_type))
            if sub_type == "Lines":
                for line_subtype in line_subtypes:
                    electricity_knowledge_graph.add(Node(f"{t_type} - {sub_type} - {line_subtype}", inherits_from=f"{t_type} - {sub_type}"))


    # Storage ======================================================================================
    storage_supertypes = ["PHS", "V2G-Batteries", "Other Storage"] # Pumped Hydro Storage, Vehicle-to-Grid Batteries
    # Storage calculations follow a 3 tiered structure: Demand is filled first with PHS, then with (anyway available) V2G-Batteries, and 
    # the residual demand with Other Storage
    storage_subtypes_categories = ["mechanical storage", "lithium battery", "flow battery", "molten salt battery", "lead acid and nickel battery", "fuel cell"]
    storage_subtypes = ["Flywheel", "Compressed Air", "Hydrogen FC", "NiMH", "Deep-cycle Lead-Acid", "LMO",
                        "NMC", "NCA", "LFP", "LTO", "Zinc-Bromide", "Vanadium Redox", "Sodium-Sulfur", "ZEBRA",
                        "Lithium Sulfur", "Lithium Ceramic", "Lithium-air"]

    electricity_knowledge_graph.add(Node("Storage", inherits_from="Electricity"))

    for supertype in storage_supertypes:
        electricity_knowledge_graph.add(Node(supertype, inherits_from="Storage"))
    # TODO: V2G Batteries are also related to Vehicles + Add V2G-Batteries subtypes? Problem with that is, that these are the same sub types as for Other Storage, how to do this?
    for subtype_category in storage_subtypes_categories:
        electricity_knowledge_graph.add(Node(subtype_category, inherits_from="Other Storage"))
    for subtype in ["Flywheel", "Compressed Air"]:
        electricity_knowledge_graph.add(Node(subtype, inherits_from="mechanical storage"))
    for subtype in ["lithium ion battery","lithium metal battery"]:
        electricity_knowledge_graph.add(Node(subtype, inherits_from="lithium battery"))
    for subtype in ["LMO","NMC","NCA", "LFP", "LTO"]:
        electricity_knowledge_graph.add(Node(subtype, inherits_from="lithium ion battery"))
    for subtype in ["Lithium Sulfur", "Lithium Ceramic", "Lithium-air"]:
        electricity_knowledge_graph.add(Node(subtype, inherits_from="lithium metal battery"))
    for subtype in ["Zinc-Bromide", "Vanadium Redox"]:
        electricity_knowledge_graph.add(Node(subtype, inherits_from="flow battery"))
    for subtype in ["Sodium-Sulfur", "ZEBRA"]:
        electricity_knowledge_graph.add(Node(subtype, inherits_from="molten salt battery"))
    for subtype in ["NiMH", "Deep-cycle Lead-Acid"]:
        electricity_knowledge_graph.add(Node(subtype, inherits_from="lead acid and nickel battery"))
    for subtype in ["Hydrogen FC"]:
        electricity_knowledge_graph.add(Node(subtype, inherits_from="fuel cell"))
    

    return electricity_knowledge_graph


knowledge_graph = KnowledgeGraph(*create_building_graph()._items, *create_vehicle_graph()._items, *create_electricity_graph()._items)

