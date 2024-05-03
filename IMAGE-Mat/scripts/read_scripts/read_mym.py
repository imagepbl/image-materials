import numpy as np
import re as re
import os.path
import pandas as pd

from csv import Sniffer as Sniffer
from string import whitespace as whitespace
# ............................................................................ #
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
# ............................................................................ #
    
def produce_df(data, rows, columns, row_names=None, column_names=None):
    """rows is a list of lists that will be used to build a MultiIndex
    columns is a list of lists that will be used to build a MultiIndex"""
    row_index = pd.MultiIndex.from_product(rows, names=row_names)
    col_index = [i for i in range(1,len(columns[0])+1)]
    return pd.DataFrame(data, index=row_index, columns=col_index)
# ............................................................................ #

def read_mym(filename,path=''):
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
        target_dimensions = tuple([time_length] + dims)
    else:
        raise RuntimeError('file dimensions are parsed incorrectly')

    # > reshape data and split off time vector = [data[:,0]]
    data = np.reshape(raw_data,raw_dimensions)
    if has_time:
        time, data = np.split(data,[1],axis=1)
        # >> reshape data to reflect original dimensions, here we use that the
        #    first dimension is the time dimension as reading from file. 
        #    The returned array will have the time array as the last dimension.
        data = np.moveaxis(np.reshape(data,target_dimensions), 0, -1)
        time = np.squeeze(time)
        return data, time
    else:
        return data

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


#if __name__ == "__main__":
#    print('zeroth order test of read_mym() by reading a file')
#    try:
#        read_mym("data\\SSP2_450\\enemisbc.out")
#    except:
#        raise