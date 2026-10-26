from scipy.signal import decimate
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import h5py
import math
import re
from datetime import datetime
from tqdm import tqdm


def sequence_number(path):
    filename = os.path.basename(path) #pull filename

    #downsampled format
    match = re.search(r'(\d{4}\D+\d{2}\D+\d{2}\D+\d{2}\D+\d{2}\D+\d{2})', filename) #looks for a string of dates - regex. '\D+' matches any character
    if match:
        timestamp = re.sub(r'\D+', '', match.group())
        return datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        # return datetime.strptime(match.group(1), "%Y-%m-%d_%H.%M.%S")

    #old raw format
    match = re.search(r'_(\d+)\.h5$', filename) #looks for a string of the form "_<number>.h5" at the end of the filename
    if match:
        return int(match.group(1)) #return the number as an integer

    # search for a number anywhere in the filepath, return that
    match = re.search(r'\d+', path)
    if match:
        return int(match.group(0)) #returns the first match in the group

    raise ValueError(f"Filename format not recognized: {filename}")

def read_dasFile(file_path:str):

    data_vars = {}
    with h5py.File(file_path) as f:

        # keys = list(f.keys()) #list all keys

        if "Acquisition" in f:
            raw = f['Acquisition']['Raw[0]']  #headers in h5 structure

        else:
            raw = f  #already decimated files without the header

        #read through each key and save as a variable to the dictionary
        for key in raw.keys():
            data_vars[key] = np.array(raw[key])

    return data_vars


def decimation_steps(initial_frequency, decimated_frequency, max_limit):
    '''Uses Prime factorization to find the optimal number of decimation steps and downsampling factor values'''

    total_q = int(initial_frequency // decimated_frequency) #check downsample factor between initial and final frequency

    if total_q <= max_limit: #if the downsample factor is less than or equal to 10 - do the decimation in one step
        return [total_q]

    #preform prime factorization to split the downsamling factor 'n' into smaller integer stages
    factors = []
    n = total_q

    #handle even prime: 2
    while n % 2 == 0: #while downsample factor is divisible by two:
        factors.append(2) #append 2 to the factor list
        n //= 2 # in place floor division of the downsample factor by two

    #handle odd factors up to the sqrt(n) - every composite number 'n' has at least one prime less than or equal to sqrt(n) - see below
    # Fundamental rule: if a number 'n' has a factor, at least one of its factors must be less than or equal to the sqrt(n)
    #the step size is 2 because all factors of 2 (all even numbers) have been removed. Stepping by 2 skips all these numbers
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        while n % i == 0:  #while the downsample factor is divisible by an element in range (3, sqrt(n), 2):
            factors.append(i) #append the prime factor
            n //= i  #dynamically adjust n
    if n > 1:  #if a remaining prime factor exists
        factors.append(n) #append to the factors list

    #check for indivisible prime numbers left in the factors list that are above the maximum allowed limit
    if any(p > max_limit for p in factors):
        raise ValueError(f'Cannot safely decimate by {total_q}. It contains a prime factor: {max(factors)}, larger than \
         max limit: {max_limit} which cannot be split')

    #greedy grouping of the factors list so that we can combine small primes into larger numbers that are under the maximum allowed limit
    factors.sort(reverse=True) # sorted factors from greatest to least
    steps = [] #empty list for stages of decimation
    while factors: #while there are values in factors
        value = factors.pop(0) #pop the first value out
        i = 0 #index of the factors list to loop over
        while i < len(factors):
            if value * factors[i] <= max_limit: #if the value * factors at index i is less than max limit
                value *= factors.pop(i) #multiply the value inplace by the factor at index i - that factor is popped from the factors list.
            else:
                i += 1  #if the value * factor[i] is too large, add to the index - moves down the sorted list of factors to smaller values
        steps.append(value) #store the value in steps list
    steps.sort(reverse=True) # sort the steps list from greatest to least

    return steps


def filter_signal(raw_signal, time, initial_frequency:float, decimated_frequency:float, filter_type:str='iir', zero_phase:bool=True, max_limit:int=13, axis:int=0):

    '''
    Decimate the signal using the defined filtering parameters.
    Multiple steps must be used when decimating a signal with a downsampling factor higher than 13 -> recommended by scipy.decimate documentation

    :parameter
        - initial_frequency (float): The initial frequency of the signal in Hz
        - decimated_frequency (float): The desired output frequency to decimate to in Hz
        - filter_type (str): The type of filter to decimate with. Default='iir' (infinite impulse response)
        - zero_phase (bool): Allow for zero_phase filtering - filter without phase distortion - important for DAS. Default=True
        - axis (int): The axis over which to preform the decimation. Default=0
        - max_limit (int): The maximum limit of the decimation factor 'q'

    :returns
        - filtered_signal (array) - processed signal
    '''

    factor = initial_frequency / decimated_frequency # downsampling factor between the two frequencies

    if factor <= 0 or not np.isclose(factor, round(factor)):  #make sure initial frequency is a positive scalar multiple of the decimated frequency - if not raise an error
        raise TypeError(f'Invalid factor: {factor}, Initial Frequency: {initial_frequency} '
                        f'must be a positive scalar multiple of the Decimated Frequency: {decimated_frequency}')

    factor = int(round(factor)) #change the factor from float to int to use for indexing

    qs = decimation_steps(initial_frequency, decimated_frequency, max_limit) # pull the downsampling factors for decimation

    if math.prod(qs) != factor:  #ensure the product of the decimation steps and the downsampling factor are equal
        raise ValueError(
            f"Product of decimation steps ({math.prod(qs)}) "
            f"does not equal the total downsampling factor ({factor})."
        )

    stage = raw_signal #array used for overwriting -> with the partially decimated signal inside the while loop
    while qs: #while qs has values:
        q = qs.pop(0) #remove the first q from the list of qs

        stage = decimate(stage, q, ftype=filter_type, axis=axis, zero_phase=zero_phase) #decimate the signal.

    decimated_signal = stage #rename for clarity
    decimated_time = time[::factor] # downsample the time array to match the signal array

    if decimated_signal.shape[axis] != len(decimated_time):  #ensure that the decimated signal array and time array match along the time axis
        raise ValueError(
            f"Signal length along axis {axis} ({decimated_signal.shape[axis]}) "
            f"does not match time length ({len(decimated_time)})."
        )

    return decimated_signal, decimated_time


def decimate_data(file_path, save_dir, initial_frequency:float, decimated_frequency:float, filter_type:str='iir', zero_phase:bool=True, max_limit:int=13):

    rawdata = read_dasFile(file_path) #pull the data from the das files

    raw_signal = rawdata['RawData']  #das signal
    raw_time = rawdata['RawDataTime'] #timestamps

    decimated_signal, decimated_time = filter_signal(raw_signal, raw_time, initial_frequency, decimated_frequency, filter_type, zero_phase, max_limit) #decimate the signal and downsample time

    decimated_count = np.arange(0, decimated_signal.shape[0]) #build the count variable

    #write to hdf5 file - And copy/modify the metadata from the original file to the new file
    write_to_hdf5(
        decimated_signal,
        decimated_frequency,
        decimated_time,
        decimated_count,
        file_path,
        save_dir
    )


def write_to_hdf5(signal, decimated_frequency, time, count, file_path, save_dir):

    # start_datetime = time2dateTime(time[0] / 1e6)  #convert from microseconds to seconds

    filename = os.path.basename(file_path) #keep the same name as the original file
    save_path = os.path.join(save_dir, filename)

    #MODIFY METADATA
    #paths to specific groups/datasets in the hdf5 file - needs user modification depending on what in the metadata is to be modified.
    acquisition_path = '/Acquisition/Raw[0]'
    signal_path = '/Acquisition/Raw[0]/RawData'
    time_path = '/Acquisition/Raw[0]/RawDataTime'
    count_path = '/Acquisition/Raw[0]/RawDataSampleCount'

    with h5py.File(file_path, "r") as src:

        acquisition_attrs = dict(src[acquisition_path].attrs) #attributes saved under the acquisition group
        signal_attrs = dict(src[signal_path].attrs)  #attributes under the signal dataset
        time_attrs = dict(src[time_path].attrs)  #attributes under the timestamp dataset
        count_attrs = dict(src[count_path].attrs)  #attributes under the count dataset

        # update metadata dictionaries with new data
        signal_attrs['Count'] = signal.size  # replace the count of the signal dataset
        time_attrs['Count'] = time.size
        acquisition_attrs['OutputDataRate'] = decimated_frequency  # replace the sampling frequency attribute with the new decimated frequency

        replacements = {   #create a dictionary with the replacement data
            signal_path: (signal, signal_attrs),
            time_path: (time, time_attrs),
            count_path: (count, count_attrs),
        }

        # Open a fresh output file instead of copying the original file.
        with h5py.File(save_path, "w") as dst:

            #preserve the attributes in the root group
            for key, value in src.attrs.items():
                dst.attrs[key] = value

            def copy_group_contents(src_group, dst_group, parent_path=""):

                # Replace attributes if the parent path is the acquisition path
                if parent_path == acquisition_path:
                    attrs = acquisition_attrs
                else:  #otherwise continue
                    attrs = src_group.attrs

                # Preserve attributes on a particular group
                for key, value in attrs.items():  #retrieve attribute names and values from the original file
                    dst_group.attrs[key] = value  #write them to the new file

                for name in src_group:  #iterating over the names direcly inside the source group: ex: acquisition/ -> raw[0] / -> rawdata, rawdatatime, rawdatasamplecount
                    obj = src_group[name]
                    path = f"{parent_path}/{name}"  #construct the objects full path using the parent path and name - needed for recursive iteration down hdf5 tree

                    #recursively recreate groups so that the selected datasets can be replaced - Ex: the 'for name in src_group' encounters another group with objects in it - see above comment
                    if isinstance(obj, h5py.Group):  #if the object pulled is an hdf5 group:
                        new_group = dst_group.create_group(name) #create new group
                        copy_group_contents(obj, new_group, path)  #copy the group contents from the old object to the new group

                    #if the path exists in the replacements dict
                    elif path in replacements:
                        data, attrs = replacements[path] #get the data and attributes from the path inside the replacements dict

                        new_dataset = dst_group.create_dataset(name, data=data) #create a new dataset in the destination file with the replacement data

                        for key, value in attrs.items(): #write the new attributes to the destination file
                            new_dataset.attrs[key] = value

                    else:
                        # Copy all unchanged datasets directly through
                        # h5py, preserving their dataset properties
                        # and attributes.
                        src.copy(obj, dst_group, name=name)

            copy_group_contents(src, dst) #run the above helper function on the source and destination files.


def sanity_plot(processed_file_path, raw_file_path, initial_frequency, decimated_frequency, channel=402):

    data_vars = {}
    with h5py.File(processed_file_path) as f:
        #read through each key and save as a variable to the dictionary
        for key in f.keys():
            data_vars[key] = np.array(f[key])

        # #print attributes in the file
        # for value in f.attrs.items():
        #     print(value)

    raw_data = read_dasFile(raw_file_path)
    raw_signal = raw_data["RawData"]
    raw_time = raw_data["RawDataTime"] / 1e6 # convert μs to s

    processed_data = read_dasFile(processed_file_path)
    signal = processed_data["RawData"]
    time = processed_data["RawDataTime"] / 1e6  # convert μs to s

    # Convert times to seconds and make them relative to the start
    raw_time = raw_time - raw_time[0]
    time = time - time[0]

    plt.figure(figsize=(12, 4))

    # Raw signal
    plt.plot(
        raw_time,
        raw_signal[:, channel],
        color='C0',
        linewidth=0.5,
        alpha=0.5,
        label=f'Raw ({initial_frequency} Hz)'
    )

    # Downsampled signal
    plt.plot(
        time,
        signal[:, channel],
        color='C1',
        markersize=3,
        linewidth=1,
        label=f'Decimated ({decimated_frequency} Hz)'
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Phase Shift (rad)")
    plt.title(f"Channel {channel}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close('all')


if __name__ == "__main__":

    # #serial processing / debugging section
    datafolder = '/mnt/d/For_OSU_raw/20260601/'
    save_dir = os.getcwd() + '/data/for_OSU/20260601'
    all_files = sorted(glob.glob(os.path.join(datafolder, '*.h5')), key=sequence_number)
    #
    initial_frequency = 500
    decimated_frequency = 2
    #
    # decimation_steps(initial_frequency, decimated_frequency, max_limit=13)

    #loop through file list
    for file in tqdm(all_files):
        decimate_data(file, save_dir, initial_frequency, decimated_frequency)

    #possible jitter in the instrument sampling rate could leave it short of 500Hz, therefore a decimation down to 2Hz using every 250th point would leave the data slightly short of true 2Hz signal.

    '''EXTRAS'''
    #sanity plot
    processed_file = os.getcwd() + '/data/processed_66/test/decimator_2026-06-06_14.00.01_UTC_011457.h5'
    raw_file = os.getcwd() + '/data/raw_66/test/decimator_2026-06-06_14.00.01_UTC_011457.h5'
    sanity_plot(processed_file, raw_file, initial_frequency, decimated_frequency, channel=402)





