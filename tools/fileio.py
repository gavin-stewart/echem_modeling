"""A collection of functions dealing with file IO for electrochemistry data."""

import numpy as np
import os.path
import json

def read_time_current_data(file_name):
    """Returns time and current data contained in the given file as two
    numpy arrays.


    The function expects the file to contain two columns of data, separated
    by whitespace.  The first column should contain time data, and the
    second the current data.  It is up to the user to perform any necessary
    unit conversions on these data.
    """

    # Redo to use preallocated arrays
    time = []
    current = []
    with open(file_name, 'r') as data_file:
        for line in data_file:
            split = line.split()
            if len(split) != 2:
                raise ValueError("In file" + file_name + "the line:\n\t"
                                 + line + "had " + str(len(split))
                                 + "whitespace separated columns"
                                 + "(2 columns expected).")
            time.append(float(split[0]))
            current.append(float(split[1]))
    return np.array(time), np.array(current)

def write_time_current_data(file_name, time, current, overwrite=False):
    """Writes the given time and current data to the specified file in a
    human-readable format.
    """

    if len(time) != len(current):
        msg = """\
                 Expected time and current lists to have the same length.  \
                 Time length was {0} and current length was {1}.\
                 """.format(len(time), len(current))
        raise ValueError(msg)

    if os.path.exists(file_name) and not overwrite:
        msg = """\
                 File {0} exists and overwriting was not requested.\
                 """.format(file_name)
        raise ValueError(msg)

    with open(file_name, 'w') as out_file:
        for time_val, current_val in zip(time, current):
            out_file.write("{0}\t{1}\n".format(time_val, current_val))

def write_time_current_bin_cmp(
        file_name, time, current, overwrite=False):
    """Writes the given time and current data to the specified file in
    binary.  This will result in a compressed .npz file.
    """

    if len(time) != len(current):
        msg = """\
                 Expected time and current lists to have the same length.  \
                 Time length was {0} and current length was {1}.\
                 """.format(len(time), len(current))
        raise ValueError(msg)

    if os.path.exists(file_name) and not overwrite:
        msg = """\
                 File {0} exists and overwriting was not requested.\
                 """.format(file_name)
        raise ValueError(msg)

    data = {'tStart' : time[0], 'tEnd' : time[-1], 'nt' : len(time),
            'I' : current}
    np.savez_compressed(file_name, **data)

def write_time_current_data_bin(
        file_name, time, current, overwrite=False):
    """Writes the given time and current data to the specified file in binary.
    This will result in a .npz file.
    """
    if len(time) != len(current):
        msg = """\
                 Expected time and current lists to have the same length.  \
                 Time length was {0} and current length was {1}.\
                 """.format(len(time), len(current))
        raise ValueError(msg)

    if os.path.exists(file_name) and not overwrite:
        msg = "File exists and overwriting was not requested."
        raise ValueError(msg)

    data = {'tStart' : time[0], 'tEnd' : time[-1], 'nt' : len(time),
            'I' : current}
    np.savez(file_name, **data)

def read_time_current_data_bin(file_name):
    """Read time and current data stored in binary format in a npz file."""
    data = np.load(file_name)
    if 't' in data: #Old-style
        time = data['t']
    else: #New-style
        t_start = data['tStart']
        t_end = data['tEnd']
        num_time_pts = data['nt']
        time = np.linspace(t_start, t_end, num_time_pts)
    current = data['I']
    return time, current

def read_json_params(file_name, data_name, type_name=None):
    """Returns a dictionary from the specified file named dataName of type
    typeName if one is found, None otherwise.
    """
    if not os.path.exists(file_name):
        raise ValueError("File {0} does not exist.".format(file_name))
    with open(file_name, 'r') as in_file:
        loaded = json.load(in_file)
    for entry in loaded:
        if entry['name'] == data_name:
            if not type_name is None and entry['type'] != type_name:
                msg = """\
                         Parameters {0} in file {1} are not of type {2}.  \
                         The parameters have type {3}\
                         """.format(data_name, file_name,
                                    type_name, entry['type'])
                raise ValueError(msg)
            return entry
    return None

def read_json_dimen_params(file_name, data_name):
    """Reads non-dimensionalized experiment parameters from the specified
    datafile and returns a dictionary object if the experiment name was found
    or None otherwise.
    """
    return read_json_params(file_name, data_name, 'dimensional')

def write_json_params(file_name, parameters):
    """Write experimental parameters to the specified datafiles.  If a
    set of parameters of the same name already exists, it will be
    overwritten.
    """
    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be a dictionary.")
    elif not ('name' in parameters and 'type' in parameters):
        raise ValueError("Parameters must contain keys 'name' and 'type'")

    if os.path.exists(file_name):
        data = json.load(file_name)
    else:
        data = []

    preexisting = False
    for ind, entry in enumerate(data):
        if entry['name'] == parameters['name']:
            data[ind] = parameters
            preexisting = True
            break
    if not preexisting:
        data.append(parameters)
    with open(file_name, "w") as out_file:
        json.dump(data, out_file)
