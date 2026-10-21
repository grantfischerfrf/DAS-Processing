from scipy.signal import decimate
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import h5py
import math
from tqdm import tqdm
from utils import sequence_number, time2dateTime, read_dasFile


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



def filter_signal(raw_signal, time, initial_frequency:float, decimated_frequency:float, filter_type:str='iir', zero_phase:bool=True, axis:int=0, max_limit:int=13):

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

    if not isinstance(factor, int):  #make sure factor is an integer - if not raise an error
        raise TypeError(f'Invalid factor: {factor}, Initial Frequency: {initial_frequency} \
         must be a scalar multiple of the Decimated Frequency: {decimated_frequency}')

    qs = decimation_steps(initial_frequency, decimated_frequency, max_limit) # pull the downsampling factors for decimation

    stage = raw_signal #array used for overwriting -> with the partially decimated signal inside the while loop
    while qs: #while qs has values:
        q = qs.pop(0) #remove the first q from the list of qs

        stage = decimate(stage, q, ftype=filter_type, axis=axis, zero_phase=zero_phase) #decimate the signal.

    decimated_signal = stage #rename for clarity
    decimated_time = time[::factor] # downsample the time array to match the signal array

    return decimated_signal, decimated_time


def decimate_data(file_path, save_dir, initial_frequency:float, decimated_frequency:float, filter_type:str='iir', zero_phase:bool=True, max_limit:int=13):

    rawdata = read_dasFile(file_path) #pull the data from the das files

    raw_signal = rawdata['RawData']
    raw_time = rawdata['RawDataTime']

    decimated_signal, decimated_time = filter_signal(raw_signal, raw_time, initial_frequency, decimated_frequency, filter_type, zero_phase, max_limit) #decimate

    #write to hdf5 file
    write_to_hdf5(
        decimated_signal,
        decimated_time,
        save_dir
    )


def write_to_hdf5(signal, time, save_dir):

    start_datetime = time2dateTime(time[0] / 1e6)  #convert from microseconds to seconds

    filename = 'decimated_' + start_datetime.strftime("%Y-%m-%d_%H.%M.%S") + "_UTC.h5"
    save_path = os.path.join(save_dir, filename)

    with h5py.File(save_path, "w") as f:

        # Signal data
        f.create_dataset(
            "RawData",
            data=signal,
            dtype=np.float64
        )

        # Time vector
        f.create_dataset(
            "RawDataTime",
            data=time,
            dtype=np.float64
        )

        # Metadata about the DAS data - #TODO: probably reorganize this - into signal and time attributes
        # f.attrs["SampleRate - Hz"] = str(2.0)
        # f.attrs["Gauge Length - m"] = str(1.6)
        # f.attrs["Pulse Width - m"] = str(1.02)
        # f.attrs["Channel Spacing - m"] = str(1.6)
        # f.attrs["nChannels"] = str(signal.shape[1])
        # f.attrs["Start Time"] = start_datetime.strftime("%Y-%m-%d_%H.%M.%S")
        # f.attrs["End Time"] = time2dateTime(time[-1] / 1e6).strftime("%Y-%m-%d_%H.%M.%S")


def sanity_plot(file_path, raw_file_path, initial_frequency, decimated_frequency, channel=402):

    data_vars = {}
    with h5py.File(file_path) as f:
        #read through each key and save as a variable to the dictionary
        for key in f.keys():
            data_vars[key] = np.array(f[key])

        # #print attributes in the file
        # for value in f.attrs.items():
        #     print(value)

    raw_data = read_dasFile(raw_file_path)
    raw_signal = raw_data["RawData"]
    raw_time = raw_data["RawDataTime"] / 1e6 # convert μs to s

    signal = data_vars["RawData"]
    time = data_vars["RawDataTime"] / 1e6  # convert μs to s

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

    #serial processing / debugging section
    datafolder = '/mnt/d/For_OSU_raw/20260601/'
    save_dir = os.getcwd() + '/data/for_OSU/20260601'
    all_files = sorted(glob.glob(os.path.join(datafolder, '*.h5')), key=sequence_number)

    initial_frequency = 500
    decimated_frequency = 2

    # decimation_steps(initial_frequency, decimated_frequency, max_limit=13)

    #loop through file list
    for file in tqdm(all_files):
        decimate_data(file, save_dir, initial_frequency, decimated_frequency)

    #possible jitter in the instrument sampling rate could leave it short of 500Hz, therefore a decimation down to 2Hz using every 250th point would leave the data slightly short of true 2Hz signal.

    '''EXTRAS'''
    #sanity plot
    # processed_file = os.getcwd() + '/data/processed_66/1400_UTC/decimated_2026-06-06_13.55.01_UTC.h5'
    # raw_file = os.getcwd() + '/data/raw_66/1400_UTC/decimator_2026-06-06_13.55.01_UTC_011456.h5'
    # sanity_plot(processed_file, raw_file, initial_frequency, decimated_frequency, channel=402)





