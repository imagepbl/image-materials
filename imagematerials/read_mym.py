import functools
import numpy as np
import re as re
import os.path
import pandas as pd
import yaml

from csv import Sniffer as Sniffer
from pathlib import Path
from string import whitespace as whitespace

# Default location of the dimension label lookup file, relative to the
# package root (imagematerials/../data/raw/dimensions.yaml).
DEFAULT_DIMENSIONS_YAML = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'dimensions.yaml'


@functools.lru_cache(maxsize=None)
def _load_dimensions_yaml(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, 'r', encoding='utf-8') as yaml_file:
        return yaml.safe_load(yaml_file) or {}
# ............................................................................ #
def mymarray_to_xarray(mym_array, dim_prefix='dim', dimensions_yaml=DEFAULT_DIMENSIONS_YAML, **dimension_overrides):
    """
    Convert a ``pym.MymArray`` object to an ``xarray.DataArray``.

    Parameters
    ----------
    mym_array : pym.MymArray
        Input object returned by ``pym.read_mym`` / ``pym.load_mym``.
    dim_prefix : str, optional
        Prefix used for generated non-domain dimension names.
    dimensions_yaml : str or pathlib.Path, optional
        Path to a YAML file mapping dimension names to their coordinate
        labels, e.g. ``FOSSIL: [coal, oil, natural gas]``. Defaults to
        ``data/raw/dimensions.yaml`` in the repository. Dimension names
        resolved via ``dim_prefix``/``dimension_overrides`` (or parsed from
        file annotations) are looked up in this file to fill in coordinate
        labels.
    **dimension_overrides
        Optional dimension name overrides keyed by non-domain position, e.g.
        ``dim_1='PRIM+4'`` or ``dim_2='NRT'``. Provided names are also used
        to look up coordinate labels in ``constants.dimensions`` when present.
        You can also pass an explicit coordinate sequence, e.g.
        ``dim_1=['coal', 'oil', ...]``, or a dict like
        ``dim_1={'name': 'primary', 'coords': ['coal', 'oil', ...]}``.

    Returns
    -------
    xarray.DataArray
        DataArray with coordinates for domain (if present) and index-like
        coordinates for all other dimensions.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError('xarray is required for mymarray_to_xarray()') from exc

    if not hasattr(mym_array, 'data') or not hasattr(mym_array, 'metadata'):
        raise TypeError('mym_array must have data and metadata attributes')

    def _coerce_coord_sequence(values):
        if isinstance(values, np.ndarray):
            return values.tolist()
        if isinstance(values, (list, tuple, pd.Index)):
            return list(values)
        return None

    def _resolve_dim_spec(local_axis, default_name):
        override_keys = [f'{dim_prefix}_{local_axis + 1}']
        if dim_prefix != 'dim':
            override_keys.append(f'dim_{local_axis + 1}')

        for key in override_keys:
            if key in dimension_overrides and dimension_overrides[key] is not None:
                override_value = dimension_overrides[key]

                if isinstance(override_value, str):
                    return str(override_value), None

                if isinstance(override_value, dict):
                    dim_name = str(override_value.get('name', default_name))
                    explicit_coords = _coerce_coord_sequence(override_value.get('coords'))
                    return dim_name, explicit_coords

                explicit_coords = _coerce_coord_sequence(override_value)
                if explicit_coords is not None:
                    return default_name, explicit_coords

                return str(override_value), None

        return default_name, None

    def _extract_dim_names_from_annotations(annotations, expected_count):
        if expected_count <= 0:
            return []

        symbol_pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        bracket_pattern = re.compile(r'\[([^\[\]]+)\]')

        for line in annotations or []:
            matches = bracket_pattern.findall(str(line))
            for match in reversed(matches):
                labels = [token.strip() for token in match.split(',')]
                if len(labels) != expected_count:
                    continue
                if all(symbol_pattern.match(label) for label in labels) and len(set(labels)) == len(labels):
                    return labels
        return None

    def _get_yaml_dimensions(dim_name):
        dimensions = _load_dimensions_yaml(dimensions_yaml)
        return dimensions.get(dim_name)

    def _get_const_dimensions(dim_name):
        try:
            import constants as const
        except ImportError:
            return None

        alias_map = {
            'PRIM+4': 'DIM2_primary_dict',
        }

        alias_target = alias_map.get(dim_name)
        if alias_target is not None:
            return getattr(const, alias_target, None)

        dimensions = getattr(const, 'dimensions', None)
        if isinstance(dimensions, dict) and dim_name in dimensions:
            return dimensions[dim_name]

        return getattr(const, dim_name, None)

    def _mapped_coord_from_const(dim_name, size):
        dim_values = _get_yaml_dimensions(dim_name)
        if dim_values is None:
            dim_values = _get_const_dimensions(dim_name)
        if dim_values is None:
            return None

        if isinstance(dim_values, dict):
            expected_index = list(range(1, size + 1))

            if set(expected_index).issubset(set(dim_values.keys())):
                return [dim_values[index] for index in expected_index]

            if set(expected_index).issubset(set(dim_values.values())):
                reverse_map = {value: key for key, value in dim_values.items()}
                return [reverse_map[index] for index in expected_index]

            return None

        if isinstance(dim_values, (list, tuple)):
            if len(dim_values) < size:
                return None
            return list(dim_values[:size])

        return None

    data = mym_array.data
    metadata = mym_array.metadata

    has_domain = bool(getattr(metadata, 'has_domain', False))

    dims = []
    coords = {}

    axis_offset = 0
    if has_domain:
        domain_name = metadata.domain if getattr(metadata, 'domain', None) else 'domain'
        dims.append(domain_name)
        coords[domain_name] = mym_array.domain
        axis_offset = 1

    non_domain_ndim = data.ndim - axis_offset
    annotation_dims = _extract_dim_names_from_annotations(
        getattr(metadata, 'annotations', []),
        expected_count=non_domain_ndim,
    )

    for axis in range(axis_offset, data.ndim):
        local_axis = axis - axis_offset
        if annotation_dims:
            default_dim_name = annotation_dims[local_axis]
        else:
            default_dim_name = f'{dim_prefix}_{local_axis + 1}'
        dim_name, explicit_coords = _resolve_dim_spec(local_axis, default_dim_name)
        dims.append(dim_name)

        if explicit_coords is not None:
            if len(explicit_coords) != data.shape[axis]:
                raise ValueError(
                    f'Explicit coordinates for {dim_name!r} have length '
                    f'{len(explicit_coords)}, expected {data.shape[axis]}'
                )
            coords[dim_name] = explicit_coords
        else:
            mapped_coord = _mapped_coord_from_const(dim_name, data.shape[axis])
            if mapped_coord is not None:
                coords[dim_name] = mapped_coord
            else:
                coords[dim_name] = np.arange(1, data.shape[axis] + 1)

    attrs = {
        'full_name': '.'.join(getattr(metadata, 'full_name', [])),
        'annotations': list(getattr(metadata, 'annotations', [])),
    }

    return xr.DataArray(
        data,
        dims=dims,
        coords=coords,
        name=getattr(metadata, 'name', None),
        attrs=attrs,
    )

def process_header(line):
    raw_data = ''
    # > the header line contains "[d1,d2,...,dn](t) = [t1", with n [di]
    #   dimensions, where the time dimension "(t)" is optional.
    # >> define [endmarker] of variable dependent on time-dependency
    has_time = True if '(t)' in line else False
    
    # >> parse the [di] dimension values
    dim_match = re.search('\[(.*?)\]',line).group(1)
    dim_match = re.split(',',dim_match)
    dims = [int(d) for d in dim_match]
    # >> the data values length includes 1 year value if [has_time]
    data_length = np.prod(dims) + 1 if has_time else np.prod(dims)
    
    # > the following "[" and "t1" are only there in the case of a
    #   time-dependent variable; t1 can also be on the next line(s).
    # >> parse the remainder of the line
    if has_time:
        start_array = line.rfind('[') + 1
        if line[start_array:].isdigit():
            raw_data = line[start_array:]

    return has_time, dims, data_length, raw_data

    
def produce_df(data, rows, columns, row_names=None, column_names=None):
    """rows is a list of lists that will be used to build a MultiIndex
    columns is a list of lists that will be used to build a MultiIndex"""
    row_index = pd.MultiIndex.from_product(rows, names=row_names)
    col_index = [i for i in range(1,len(columns[0])+1)]
    return pd.DataFrame(data, index=row_index, columns=col_index)


def read_mym_df(filename,path=''):
    """
    PURPOSE:
    Read a MyM data file with or without a time component, assuming a single
    variable per file. 
    
    INPUT:
    [filename]  name of the file(s) to be read in
    [path]      path to the filename, preferably relative to the working
                directory of [read_mym]
    
    OUTPUT:
    [header]    contains the leading header lines
                where the last line ends with [a1,...,an](t) = [
    [data]      contains the data as a [t,a1,...,an] shaped numpy matrix
    [time]      numpy vector with time entries (length t)

    VERSION:
    1.0 | 2018.10.10 | MvdB
    """

    # Process path
    filename = os.path.join(path,filename)
    
    # Open file and read data 
    with open(filename,'r') as mym_file:
        header = []
        line = mym_file.readline().strip(whitespace + ',')
        while line[0] == '!':
            header.append(line)
            line = mym_file.readline().strip(whitespace + ',')
        # > process header and put data into string 
        has_time, dims, data_length, raw_data = process_header(line)
        content = mym_file.read().replace('\n',' ').rstrip('];' + whitespace)
    
    # Process data
    # > transform to numpy array, using the csv.Sniffer to find the delimiter
    # >> first_chunk is a large enough string for the sniffer to find
    #    the delimiter and small enough to be fast.
    first_chunk = 64
    mym_format = Sniffer().sniff(content[:first_chunk],delimiters=''.join([',',whitespace]))
    # >> add potential initial data point from the header in [raw_data]
    raw_data = mym_format.delimiter.join([raw_data,content])
    raw_data = np.fromstring(raw_data,sep=mym_format.delimiter)

    # > transform data array to correct dimensions
    if raw_data.size % data_length == 0:
        time_length = int(raw_data.size / data_length)
        raw_dimensions = (time_length,data_length)
        multiply = dims[0:-1]                         #for dataframe ordering we need to know the length of the dataframe as the time times all but the last dimension
        multiplier = [1]
        for dimension in range(0,len(multiply)):
            multiplier[0] = multiplier[0] * multiply[dimension]
        target_dimensions_time = tuple([time_length * multiplier[0]] + [dims[-1]])
        target_dimensions = tuple([multiplier[0]] + [dims[-1]])
    else:
        raise RuntimeError('file dimensions are parsed incorrectly')

    # > reshape data and split off time vector = [data[:,0]]
    data = np.reshape(raw_data,raw_dimensions)
    
    if has_time:
        time, data = np.split(data,[1],axis=1)
        # >> reshape data to reflect original dimensions, here we use that the
        #    first dimension is the time dimension as reading from file. 
        #    The returned array will have the time array as the last dimension.
        rows_in = []
        cols_in = []
        row_names = ['time']
        rows_in.append(list(np.squeeze(time).astype(int))) # start with the time dimension (as integers)
        for dimension in range(0,len(dims)-1):
            rows_in.append([i for i in range(1,dims[dimension]+1)])
            row_names.append('DIM_' + str(dimension +1))
        cols_in.append([i for i in range(1,dims[-1]+1)]) # columns are defined by the last dimension
        
        data = np.reshape(data,target_dimensions_time)
        #data = np.moveaxis(np.reshape(data,target_dimensions), -1, 0)
        # generate empty multi-demensional dataframe
        df = produce_df(data, rows_in,cols_in,row_names=row_names)
        df.reset_index(inplace=True)   # chnage tupled multi-index to explicit columns
        if len(rows_in)<2:
            df_out = df.set_index('time')  # if only 2 dimensions, set time column to be the index (in case of more dimensions time would generate a non-unique index)
        else:
            df_out = df
        time = np.squeeze(time)
        return df_out
    else:
        rows_in = []
        cols_in = []
        row_names = []
        for dimension in range(0,len(dims)-1):
            rows_in.append([i for i in range(1,dims[dimension]+1)])
            row_names.append('DIM_' + str(dimension +1))
        cols_in.append([i for i in range(1,dims[-1]+1)]) # columns are defined by the last dimension
        
        data = np.reshape(data,target_dimensions)
        #data = np.moveaxis(np.reshape(data,target_dimensions), -1, 0)
        # generate empty multi-demensional dataframe
        df = produce_df(data, rows_in,cols_in,row_names=row_names)
        df.reset_index(inplace=True) 
        return df