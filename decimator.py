from scipy.signal import decimate
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from FRF_DAS_processing import read_dasFile, sequence_number, time2dateTime, read_Json


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


def decimate_data(file_path, save_dir):

    rawdata = read_dasFile(file_path)

    raw_signal = rawdata['RawData']
    raw_time = rawdata['RawDataTime']

    #decimate the raw signal - from 500Hz to 2Hz is a factor of 250 (q). Decimate in three steps (10, 5, 5) to avoid instabilities using an 'iir' filter - recommended by documentation
    #step 1: factor of 10 - returns 50 Hz signal
    step1 = decimate(raw_signal, 10, ftype='iir', axis=0, zero_phase=True) #zero_phase = True == no phase shift on the data - We are measuring phase shifts this is important to NOT change

    #step 2: factor of 5 - returns 10 Hz signal
    step2 = decimate(step1, 5, ftype='iir', axis=0, zero_phase=True)

    #step 3: factor of 5 - returns 2 Hz signal
    step3 = decimate(step2, 5, ftype='iir', axis=0, zero_phase=True)

    downsampled_signal = step3 #rename

    #downsample time to match the signal - factor of 250
    downsampled_time = raw_time[::250]  #TODO: make the factor an argument - no hard code

    #write to hdf5 file
    write_to_hdf5(
        downsampled_signal,
        downsampled_time,
        save_dir
    )

def sanity_plot(file_path, raw_file_path, channel=1000):

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
        label='Raw (500 Hz)'
    )

    # Downsampled signal
    plt.plot(
        time,
        signal[:, channel],
        color='C1',
        markersize=3,
        linewidth=1,
        label='Decimated (2 Hz)'
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Phase Shift (rad)")
    plt.title(f"Channel {channel}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close('all')


def print_all_attributes(name, obj):
    if obj.attrs:
        print(f"\nAttributes for [{name}]:")
        for key, value in obj.attrs.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":

    datafolder = '/mnt/d/For_OSU_raw/20260601/'
    save_dir = os.getcwd() + '/data/for_OSU/20260601'
    all_files = sorted(glob.glob(os.path.join(datafolder, '*.h5')), key=sequence_number)

    # loop through file list
    for file in tqdm(all_files):
        decimate_data(file, save_dir)





    '''EXTRAS'''  #TODO: maybe new script at some point for these
    #sanity plot
    # processed_file = os.getcwd() + '/data/processed_66/1400_UTC/decimated_2026-06-06_13.55.01_UTC.h5'
    # raw_file = os.getcwd() + '/data/raw_66/1400_UTC/decimator_2026-06-06_13.55.01_UTC_011456.h5'
    # sanity_plot(processed_file, raw_file, channel=400)


    # #read_Json Files - DAS configuration and engineering chain
    # optical_setup = './documentation/FRF_Fiber1Mode8_Fiber2Mode10_1.6mGL_opticalSetUp.json'
    # engineering_chain = './documentation/ONYX-0564 Fiber 1__SimpleLinearAssetChainFiber1_V2_v50_engineeringChain.json'
    # read_Json(optical_setup)

    #read attributes
    # with h5py.File(raw_file, 'r') as f:
    #     # First, print root attributes
    #     print_all_attributes(raw_file, f)
    #     # Then, recursively visit all objects inside the file
    #     f.visititems(print_all_attributes)



