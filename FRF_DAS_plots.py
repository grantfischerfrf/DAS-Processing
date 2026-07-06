import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime, timezone
import h5py
import os
import numpy as np
import glob
from tqdm import tqdm
from FRF_DAS_processing import read_dasFile, sequence_number
from scipy.signal import welch, spectrogram


def time2dateTime(time_array):

    #convert array of unix timestamps to datetime objects
    dateTime_array = np.array([datetime.fromtimestamp(t, tz=timezone.utc) for t in time_array])
    return dateTime_array


def load_processedData(processed_data_path):

    with h5py.File(processed_data_path, 'r') as f:  #open hdf5 file and read the strain and time datasets
        strain = f['strain'][:]  # shape (time, N_channels)
        time = f['time'][:]  # shape (time,)

    return strain, time


def plot_timeSeries(data, time, channel:int):

    #plot a time series at a particular channel location

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(time[:], data[:, channel])
    ax.set_xlabel('Time (unix stamp)')
    # ax.set_ylabel('Phase Shift (radians)')
    ax.set_title(f'Time Series at Channel {channel}')
    plt.show()
    plt.close('all')


def waterFall(data, time, starttime:datetime, endtime:datetime, channel_start:int, channel_end:int, savepath:str):

    #plot a waterfall plot of the strain data between starttime and endtime
    fig, ax = plt.subplots(figsize=(10, 8))
    channels = np.arange(channel_start, channel_end)  # create an array of channel indices
    v = np.nanpercentile(np.abs(data), 99) # get the 99th percentile of the absolute value of the data to set the color scale limits
    pcm = ax.pcolormesh(time, channels, data.T, shading='auto', vmin=-v, vmax=v, cmap='RdBu')
    ax.set_xlim(starttime, endtime)
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Channel')
    ax.set_title(f'Waterfall Plot of Strain Data from Channel {channel_start} to {channel_end}')
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label("Strain")
    plt.savefig(savepath + starttime.strftime('_%Y-%m-%d_%H-%M-%S') + '_to_' + endtime.strftime('%Y-%m-%d_%H-%M-%S') + '.png')
    plt.show()
    plt.close('all')


def build_spectrogram_data(all_files, channel=None, sampling_frequency=500,
                             nperseg=512, noverlap=400):
    """
    Concatenate raw strain across files and compute a spectrogram.
    If channel is None, averages across all channels before computing the spectrogram
    """
    strain_list = []
    time_list = []

    for file_path in tqdm(all_files):
        raw_data = read_dasFile(file_path)
        raw_strain = raw_data['RawData']  # shape (n_samples, n_channels)
        file_time = raw_data['RawDataTime'] / 1e6  # microseconds to seconds

        if channel is not None:
            strain_list.append(raw_strain[:, channel]) # select a specific channel
        else:
            strain_list.append(raw_strain.mean(axis=1))  # average across channels

        time_list.append(file_time)

    strain_concat = np.concatenate(strain_list)   # shape (total_samples,)
    time_concat = np.concatenate(time_list)       # shape (total_samples,)

    freqs, times, Sxx = spectrogram(
        strain_concat,
        fs=sampling_frequency,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'  # gives PSD units (strain^2/Hz)
    )
    # Sxx shape: (n_freqs, n_times)

    # scipy's spectrogram returns `times` as seconds-from-start, not absolute unix time
    # convert those to absolute unix time, then to datetime for plotting
    absolute_times = time_concat[0] + times
    datetime_array = time2dateTime(absolute_times)

    return freqs, datetime_array, Sxx


def spectrogram_plot(PSD, frequencies, time, savepath:str, channel=None, dB=False):

    #create a spectrogram of frequency data over time.
    fig, ax = plt.subplots(figsize=(10, 8))

    if dB:
        dB_PSD = 10 * np.log10(PSD + 1e-20) #convert to dB avoiding log(0)
        pcm = ax.pcolormesh(time, frequencies, dB_PSD[:-1, :-1], shading='auto', cmap='viridis')
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label("Power Spectral Density (dB)")
    else:
        pcm = ax.pcolormesh(time, frequencies, PSD[:-1, :-1], norm=colors.LogNorm(), shading='auto', cmap='viridis')
        #log scale
        # ax.set_yscale('log')
        #linear scale
        ax.set_ylim(0, 20)
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label("Power Spectral Density (Strain**2/Hz)")

    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Frequency (Hz)')

    if channel is not None:
        ax.set_title(f'Spectrogram of Power Spectral Density at Channel {channel}')
        plt.savefig(savepath + '_' + str(channel) + '_' + time[0].strftime('_%Y-%m-%d_%H-%M-%S') + '_to_' + time[-1].strftime(
            '%Y-%m-%d_%H-%M-%S') + '.png')
    else:
        ax.set_title('Spectrogram of Power Spectral Density')
        plt.savefig(savepath + time[0].strftime('_%Y-%m-%d_%H-%M-%S') + '_to_' + time[-1].strftime(
            '%Y-%m-%d_%H-%M-%S') + '.png')

    plt.show()
    plt.close('all')


if __name__ == "__main__":

    #processed data
    processed_data_path = os.getcwd() + '/data/processed_614/processed_data.h5'
    strain, time = load_processedData(processed_data_path) #pull processed strain and time data

    datetime_array = time2dateTime(time) #convert unix to datetime for visualizations

    #define starttime and endtime for plotting
    starttime = datetime(2026, 6, 14, 7, 50, 0, tzinfo=timezone.utc)
    endtime = datetime(2026, 6, 14, 8, 56, 0, tzinfo=timezone.utc)

    '''CREATE WATERFALL PLOT'''
    # savepath = './figures/waterfall'
    # channel_start = 0  #define start and end channels for plotting
    # channel_end = strain.shape[1]
    # waterFall(strain, datetime_array, starttime, endtime, channel_start, channel_end, savepath)

    '''CREATE SPECTROGRAM'''
    savepath = './figures/spectrogram'

    #raw data
    raw_data_path = os.getcwd() + '/data/raw_614/'

    all_files = sorted(glob.glob(os.path.join(raw_data_path, '*.h5')), key=sequence_number)

    channel = 400  #User input here

    freqs, datetime_array_, Sxx = build_spectrogram_data(
        all_files, channel=channel, sampling_frequency=500, nperseg=512, noverlap=400
    )

    spectrogram_plot(Sxx, freqs, datetime_array_, savepath, channel=channel, dB=True)











    #TODO: create a PSD spectrogram for any given channel so we can check the frequency content and see if aliasing is actually going to be a problem.

    #TODO: This script would be good as a jupyter notebook if other people are to use it at some point
