
import pytest

import xarray as xr

from imagematerials.concepts import KnowledgeGraph, Node

multiple_inheritance = KnowledgeGraph(
    Node("base"),
    Node("a", inherits_from="base"),
    Node("b", inherits_from="base"),
    Node("1", inherits_from="base"),
    Node("2"),
    Node("b1", inherits_from=["b", "1"]),
    Node("a1", inherits_from=["a", "1"]),
    Node("a2", inherits_from=["a", "2"]),
    Node("b2", inherits_from=["b", "2"]),
)

def test_multiple_inheritance():
    all_twos = multiple_inheritance.find_one_relation(["a2", "b2", "a1", "a2"], "2")
    assert set(all_twos) == {"a2", "b2"}
    all_as = multiple_inheritance.find_one_relation(["a2", "b2", "a1", "a2"], "a")
    assert set(all_as) == {"a1", "a2"}

    all_desc_as = multiple_inheritance.find_one_relation(["a", "b", "1", "2"], "a1")
    assert set(all_desc_as) == {"a", "1"}

    with pytest.raises(ValueError):
        multiple_inheritance.find_one_relation(["b"], "a1")

    all_base = multiple_inheritance.find_one_relation(["a1", "a2", "b1", "b2"], "base")
    assert set(all_base) == {"a1", "a2", "b1", "b2"}
    assert len(all_base) == 4

def test_non_unique_error():
    with pytest.raises(ValueError):
        KnowledgeGraph(
            Node("a"),
            Node("a")
        )
    with pytest.raises(ValueError):
        KnowledgeGraph(
            Node("a"),
            Node("b", synonyms=["a"])
        )
    with pytest.raises(ValueError):
        KnowledgeGraph(
            Node("a", synonyms=["a"])
        )

def test_rebroadcast_error():
    array = xr.DataArray(0.0, dims=("Type",), coords={"Type": ["a1", "a2", "b1", "b2"]})
    with pytest.raises(ValueError):
        multiple_inheritance.rebroadcast_xarray(array, ["1", "2"])
    with pytest.raises(ValueError):
        multiple_inheritance.rebroadcast_xarray(array, ["a", "b"])