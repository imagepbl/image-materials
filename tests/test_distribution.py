import numpy as np
from pytest import mark

from imagematerials.distribution import NAME_TO_DIST


@mark.parametrize(
    "dist_name,parameter_names",
    [("folded_norm", ("mean", "stdev")),
     ("weibull", ("shape", "scale"))]
)
def test_distributions(dist_name, parameter_names):
    dist = NAME_TO_DIST[dist_name]
    for pname in parameter_names:
        assert dist.has_param(parameter_names)
    assert not dist.has_param(("some-unknown-param"))
    scipy_params = dist.get_param({pname: 1 for pname in parameter_names})
    assert set(scipy_params.keys()) == set(("c", "scale", "loc"))
    assert isinstance(scipy_params, dict)
    dist_vals = dist.method(np.arange(10), **scipy_params)
    assert isinstance(dist_vals, np.ndarray)
    assert len(dist_vals) == 10


@mark.parametrize(
    "dist_name,parameters",
    [("folded_norm", {"mean": np.arange(100), "stdev": np.random.randn(100)}),
     ("weibull", {"shape": 1+np.random.randn(100), "scale": 1+np.random.randn(100)})]
)
def test_vectorized(dist_name, parameters):
    dist = NAME_TO_DIST[dist_name]
    scipy_param = dist.get_param(parameters)
    max_len = 1
    for val in parameters.values():
        if not isinstance(val, (float, int)):
            max_len = max(max_len, len(val))
    for par_name, values in scipy_param.items():
        if isinstance(values, np.ndarray):
            assert len(values) == max_len

    dist_vals = dist.method(np.arange(max_len*10).reshape(10, -1), **scipy_param)
    assert dist_vals.shape == (10, max_len)
