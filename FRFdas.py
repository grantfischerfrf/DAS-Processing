import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, filtfilt, lfilter
from scipy.ndimage import uniform_filter1d
from collections import deque
import os
import h5py
import glob
from tqdm import tqdm
from datetime import datetime, timezone


def read_dasFile(file_path:str):

    data_vars = {}
    with h5py.File(file_path) as f:

        # keys = list(f.keys()) #list all keys
        Acquisition = f['Acquisition']  #headers in h5 structure  #TODO: check the last file (this may not be true for all files), or add a safety check to make sure the key exists
        raw = Acquisition['Raw[0]']

        #read through each key and save as a variable to the dictionary
        for key in raw.keys():
            data_vars[key] = np.array(raw[key])

    return data_vars


def phase2strain(phase_shifts, gauge_length):
    #use equations from Hartog 2017. Use linear theory to convert phase shifts to axial strain (Exx(t, x))
    L = 1550e-9  #wavelength of light in meters
    psi = 0.79  #pockel's coefficient for single mode glass fiber
    nc = 1.46  #index of refraction for the optical fiber

    strain = (L / (4 * np.pi * nc * gauge_length * psi)) * phase_shifts

    return strain


def plot_timeSeries(data, channel:int):

    #plot a time series at a particular channel location
    phase_shifts = data['data']
    time = data['time']

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(time[:], phase_shifts[:, channel])
    ax.set_xlabel('Time (unix stamp)')
    ax.set_ylabel('Phase Shift (radians)')
    ax.set_title(f'Time Series at Channel {channel}')
    plt.show()
    plt.close('all')


def time2dateTime(time_array):

    #convert array of unix timestamps to datetime objects
    dateTime_array = [datetime.fromtimestamp(t, tz=timezone.utc) for t in time_array]
    return dateTime_array


class DASProcessor:

    # process the raw strain data - the order of operations will be as follows:
    # 1: temporal detrend with a 10-minute moving window (in each isolated channel)
    # - this should remove any low frequency noise (temperature, laser drift, etc.) and leave the wave band
    # 2: common - mode removal (spatial detrend)
    # - detrend across channels - removes DC offsets, biases, temp, lase drift etc... also should leave just the wave band
    # 3: low pass filter to remove high frequency noise
    # make sure to low pass filter at least close to the nyquist frequency, there is likely almost no energy between 1 - ~5Hz that could alias
    # 4: downsample to a chosen frequency
    # finishes decimating the data. DAS has extremely high temporal resolution. Not all of this is needed.

    def __init__(self, gauge_length:float, sampling_frequency:int, temporal_window:int, save_path:str):
        #gauge lenth in meters, sampling frequency in Hz, temporal window in minutes, save path for processed data

        self.gauge_length = gauge_length
        self.sampling_frequency = sampling_frequency
        self.temporal_window = temporal_window
        self.window_samples = int(temporal_window * 60 * sampling_frequency) #number of samples in the temporal window

        self.f = h5py.File(save_path, "w")  # open (create/overwrite) the output HDF5 file for incremental writing
        self.strain_ds = None  # will hold the resizable strain dataset once initialized
        self.time_ds = None  # will hold the resizable time dataset once initialized

    def process_file(self, rawData, unix_time, start_clip, stop_clip, cutoff_frequency=5, downsample_frequency=2):

        # convert unix time to datetime object
        time_seconds = unix_time / 1e6  # unix time stamp is stored in microseconds - convert to seconds

        #run data through processing pipeline
        rawStrain = phase2strain(rawData, self.gauge_length)  # convert phase shifts to strain using hartog, 2017 linear method

        #initialize hdf5 file if not already done
        n_channels = rawStrain.shape[1]
        if self.strain_ds is None:
            self._init_h5(n_channels)

        strain_temporal_detrend = self.temporal_detrend(rawStrain)  # temporal detrend with a 10-minute moving window (in each isolated channel)

        strain_common_mode_removed = self.common_mode_removal(strain_temporal_detrend)  # common - mode removal (spatial detrend)

        strain_low_pass_filtered = self.low_pass_filter(strain_common_mode_removed, cutoff_frequency=cutoff_frequency)  # low pass filter to remove high frequency noise

        #downsample the data
        strain_low_pass_filtered = strain_low_pass_filtered[start_clip:stop_clip]  #clip strain and time arrays
        time_seconds = time_seconds[start_clip:stop_clip]

        #downsample
        strain_downsampled, time_downsampled = processor.downsample(strain_low_pass_filtered, time_seconds, downsample_frequency=2)

        #write to hdf5 file
        processor._write(strain_downsampled, time_downsampled)

        # return strain_downsampled, time_downsampled


    def temporal_detrend(self, rawStrain):

        # 1: temporal detrend with a 10-minute moving window (in each isolated channel)
        # - this should remove any low frequency noise (temperature, laser drift, etc.) and leave the wave band

        baseline = uniform_filter1d(rawStrain, size=self.window_samples, axis=0, mode='nearest')

        return rawStrain - baseline

    def common_mode_removal(self, strain_temporal_detrend):

        # 2: common - mode removal (spatial detrend)
        # - detrend across channels - removes DC offsets, biases, temp, lase drift etc... also should leave just the wave band

        baseline = np.median(strain_temporal_detrend, axis=1, keepdims=True)

        return strain_temporal_detrend - baseline

    def low_pass_filter(self, strain_common_mode_removed, cutoff_frequency=5, butter_order=4):

        # 3: low pass filter to remove high frequency noise
        # make sure to low pass filter at least close to the nyquist frequency, there is likely almost no energy between 1 - ~5Hz that could alias

        b, a = butter(butter_order, cutoff_frequency, btype='lowpass', fs=self.sampling_frequency, analog=False, output='ba') #returns numerator and denominator coefficients of the filter transfer function
        # print(len(output))
        # b, a = output[0], output[1] #unpack output

        filtered_strain = filtfilt(b, a, strain_common_mode_removed, axis=0)  # apply filter along the time axis - zero phase filtering

        return filtered_strain

    def downsample(self, strain_low_passed, time, downsample_frequency=2):

        # 4: downsample to a chosen frequency
        # finishes decimating the data. DAS has extremely high temporal resolution. Not all of this is needed.

        sampling_interval = int(self.sampling_frequency / downsample_frequency)
        strain_downsampled = strain_low_passed[::sampling_interval, :]
        time_downsampled = time[::sampling_interval]

        return strain_downsampled, time_downsampled


    def _init_h5(self, n_ch, chunk_size=5000):
        #initialize the HDF5 file with datasets for strain and time
        self.strain_ds = self.f.create_dataset(   #create a strain dataset in the hdf5 file
            "strain",  #name of dataset
            shape=(0, n_ch), #initial shape
            maxshape=(None, n_ch), #maximum shape - allow for unlimited growth in the time dimension
            dtype=np.float32, #datatype, keep storage light bc DAS data is dense
            chunks=(chunk_size, n_ch), #store data in chunks of (chunk_size) time samples to make writing faster
            compression="gzip",  #compress the data to save space
        )

        self.time_ds = self.f.create_dataset(   #create a time dataset in the hdf5 file
            "time", #name of dataset
            shape=(0,), #initial shape - no channels for a time array
            maxshape=(None,), #allow infinite growth
            dtype=np.float64,
            chunks=(chunk_size,), #store data in chunks of (chunk_size) time samples to make writing faster
            compression="gzip",  #compress the data to save space
        )

    def _write(self, strain, time):
        #write data to hdf5 file
        old = self.strain_ds.shape[0]  #old shape of the dataset - num of time samples already written
        new = strain.shape[0]  #new shape of the dataset - num of time samples to be written

        self.strain_ds.resize(old + new, axis=0) #resize the dataset to accommodate the new data
        self.strain_ds[old:] = strain.astype(np.float32) #write the new data to the dataset

        self.time_ds.resize(old + new, axis=0) #resize the dataset to accommodate the new data
        self.time_ds[old:] = time.astype(np.float64) #write the new data to the dataset

    def close(self):
        self.f.close() #close the HDF5 file when done



if __name__ == "__main__":

    datafolder = os.getcwd() + '/data/raw_614'
    savefolder = os.getcwd() + '/data/processed_614'

    #define gauge length and sampling frequency
    gauge_length = 1.6 #meters
    sampling_frequency = 500 #hz
    #1592 channels?? #TODO: find the channel length in meters?


    #instantiate the DASProcessor class
    processor = DASProcessor(
        gauge_length=gauge_length,
        sampling_frequency=sampling_frequency,
        temporal_window=10, #minutes - similar to the moving window used in Glover et al., 2024
        save_path=os.path.join(savefolder, 'processed_data.h5')
    )


    '''all files'''
    all_files = glob.glob(os.path.join(datafolder, '*.h5'))
    for i in tqdm(range(1, len(all_files) - 1)):  #start from index one to get current previous and next file

        previous_file = read_dasFile(all_files[i - 1])
        current_file = read_dasFile(all_files[i]) #raw data is of shape (time, N_channels)
        next_file = read_dasFile(all_files[i + 1])

        current_file_path = all_files[i]
        print(f'Processing file: {current_file_path}')

        rawData = np.vstack([   #Stack data into one array
            previous_file['RawData'],
            current_file['RawData'],
            next_file['RawData']
        ])

        time = np.concatenate([    #stack time into one array
            previous_file['RawDataTime'],
            current_file['RawDataTime'],
            next_file['RawDataTime']
        ])

        #we only want to save the processed_data from the current file - therefore we must clip the data to remove the buffers from the previous and next files
        n = current_file['RawData'].shape[0]  #N of samples in the current file
        start_clip = previous_file['RawData'].shape[0] #indices where to start and stop a clip to get rid of the buffers
        stop_clip = start_clip + n

        #process raw strain data
        processor.process_file(rawData, time, start_clip, stop_clip, cutoff_frequency=5, downsample_frequency=2)  #TODO: can return the last processed data to plot if wanted in the future.

    processor.close()


    # processed_data_path = './data/processed_614/processed_data.h5'
    # with h5py.File(processed_data_path, 'r') as f:
    #     strain = f['strain'][:]  # shape (time, N_channels)
    #     time = f['time'][:]  # shape (time,)
    #
    # normalize_strain = strain / (np.std(strain, axis=0, keepdims=True) + 1e-12)

    # datetime_array = time2dateTime(time)  #FIXME: clip to seconds only for visualizations
    # channel = 700
    #
    # fig, ax = plt.subplots()
    # channels = np.arange(strain.shape[1])  # create an array of channel indices
    # # ax.plot(datetime, strain[:, 700])
    # v = np.nanpercentile(np.abs(strain), 99)
    # pcm = ax.pcolormesh(datetime_array, channels, strain.T, shading='auto', vmin=-v, vmax=v, cmap='RdBu')
    # t0 = datetime(2026, 6, 14, 8, 30, 0, tzinfo=timezone.utc)
    # t1 = datetime(2026, 6, 14, 8, 31, 0, tzinfo=timezone.utc)
    # ax.set_xlim(t0, t1)
    # ax.set_ylim(400, 460)
    # cbar = plt.colorbar(pcm, ax=ax)
    # cbar.set_label("Strain")
    #
    #
    # plt.savefig('./figures/waterfall.png')
    # plt.close('all')





    #TODO: at some point it may be good to output all intermediate products like is done in the TapTest.ipynb

    #TODO: Create the option to save the files as 'x' minute sections or One large hdf5 file




