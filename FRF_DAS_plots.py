import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
import h5py
import os
import numpy as np
import glob
from tqdm import tqdm
from FRF_DAS_processing import read_dasFile, sequence_number, time2dateTime
from scipy.signal import welch, spectrogram

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


def waterFall(data, time, channel_spacing, channel_start:int, channel_end:int, savepath:str, starttime:datetime=None, endtime:datetime=None):

    #plot a waterfall plot of the strain data between starttime and endtime
    fig, ax = plt.subplots(figsize=(10, 8))
    channels = np.arange(channel_start, channel_end)  # create an array of channel indices

    #convert channels to cross-shore distance - We need to know the exact fiber orientation in order to do this. It may not be perpendicular to shore anymore
    along_cable_distance = channels * channel_spacing  # convert channel indices to distance along the cable

    v = np.nanpercentile(np.abs(data), 99) # get the 99th percentile of the absolute value of the data to set the color scale limits - limit outliers
    pcm = ax.pcolormesh(time, along_cable_distance, data.T, shading='auto', vmin=-v, vmax=v, cmap='RdBu')
    ax.set_xlabel('Time (UTC)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.set_ylabel('Along-Cable Distance (m)')
    if starttime is not None:
        ax.set_title('Waterfall Plot' + starttime.strftime('_%Y_%m_%d_%H:%M') + '_to_' + endtime.strftime('%Y_%m_%d_%H:%M'))
        ax.set_xlim(starttime, endtime)
    else:
        ax.set_title('Waterfall Plot' + time[0].strftime('_%Y_%m_%d_%H:%M') + '_to_' + time[-1].strftime('%Y_%m_%d_%H:%M'))
        ax.set_xlim(time[0], time[-1])
    # ax.set_ylim(600, 1500)
    ax.tick_params(axis='x', rotation=45)
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label("Strain")
    plt.tight_layout()
    plt.savefig(savepath + time[0].strftime('_%Y_%m_%d_%H_%M_%S') + '_to_' + time[-1].strftime('%Y_%m_%d_%H_%M_%S') + '.png')
    # plt.show()
    plt.close('all')


def build_spectrogram_data(all_files, channel:int, sampling_frequency=500,
                             nperseg=512, noverlap=400):
    """
    Concatenate raw strain across files and compute a spectrogram.
    If channel is None, averages across all channels before computing the spectrogram
    """
    strain_list = []
    time_list = []  #FIXME: careful with the memory here

    for file_path in tqdm(all_files):  #TODO: this will need to handle either one large file or a list of files.
        raw_data = read_dasFile(file_path)
        # strain, time = load_processedData(file_path)
        strain = raw_data['RawData']  # shape (n_samples, n_channels)
        time = raw_data['RawDataTime'] / 1e6  # microseconds to seconds


        strain_list.append(strain[:, channel]) # select a specific channel

        time_list.append(time)

    strain_concat = np.concatenate(strain_list)   # shape (total_samples,)
    time_concat = np.concatenate(time_list)       # shape (total_samples,)

    freqs, times, Sxx = spectrogram(   #TODO: how big are the frequency bins that this creates??
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


def spectrogram_plot(PSD, frequencies, time, savepath:str, channel:int, dB=False):

    #create a spectrogram of frequency data over time.
    fig, ax = plt.subplots(figsize=(10, 8))

    if dB:
        tail='dB'
        dB_PSD = 10 * np.log10(PSD + 1e-20) #convert to dB avoiding log(0)
        vmin = np.percentile(dB_PSD, 5) # set the colorbar floor to a value below the 5th percentile of the data to avoid outliers dominating the color scale
        vmax = np.percentile(dB_PSD, 99) # set the colorbar ceiling to a value above the 99th percentile of the data to avoid outliers dominating the color scale
        ax.set_ylim(0, 5)
        pcm = ax.pcolormesh(time, frequencies, dB_PSD[:-1, :-1], shading='auto', vmin=vmin, vmax=vmax, cmap='viridis')
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label("Power Spectral Density (dB)")
    else:
        tail=''
        vmin = np.percentile(PSD,5)  # set the colorbar floor to a value below the 5th percentile of the data to avoid outliers dominating the color scale
        vmax = np.percentile(PSD,99.9)  # set the colorbar ceiling to a value above the 99th percentile of the data to avoid outliers dominating the color scale
        pcm = ax.pcolormesh(time, frequencies, PSD[:-1, :-1], shading='auto', vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_ylim(0, 5)
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label("Power Spectral Density (Strain**2/Hz)")

    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Frequency (Hz)')


    ax.set_title(f'Spectrogram of Power Spectral Density at Channel {channel}')
    plt.tight_layout()
    plt.savefig(savepath + '_' + str(channel) + '_' + time[0].strftime('_%Y-%m-%d_%H-%M-%S') + '_to_' + time[-1].strftime(
        '%Y-%m-%d_%H-%M-%S') + tail + '.png')

    plt.show()
    plt.close('all')


if __name__ == "__main__":

    #processed data
    processed_data_path = os.getcwd() + '/data/processed_66/1300_UTC/Processed_data_all_files.h5'
    strain, time = load_processedData(processed_data_path) #pull processed strain and time data

    datetime_array = time2dateTime(time) #convert unix to datetime for visualizations

    #define starttime and endtime for plotting
    starttime = datetime(2026, 6, 6, 0, 0, 0, tzinfo=timezone.utc)
    endtime = datetime(2026, 6, 6, 13, 5, 0, tzinfo=timezone.utc)

    '''CREATE WATERFALL PLOT'''
    # savepath = './figures/20260601'
    channel_start = 0  #define start and end channels for plotting
    # channel_end = strain.shape[1]
    channel_spacing=1.6 #meters
    # waterFall(strain, datetime_array, channel_spacing, starttime, endtime, channel_start, channel_end, savepath)

    filepaths = sorted(glob.glob('./data/for_OSU_processed/20260601/*.h5'), key=sequence_number)
    count = 0
    for file in tqdm(filepaths):
        strain, time = load_processedData(file)

        channel_end = strain.shape[1]

        savepath = f'./figures/20260601/Waterfall_'

        waterFall(strain, time2dateTime(time), channel_spacing, channel_start, channel_end, savepath)



    '''CREATE SPECTROGRAM'''
    # savepath = './figures/spectrogram'
    #
    # #raw data
    # raw_data_path = os.getcwd() + '/data/raw_614/'
    # # raw_data_path = os.getcwd() + '/data/processed_614/detrend/'
    #
    # all_files = sorted(glob.glob(os.path.join(raw_data_path, '*.h5')), key=sequence_number)
    #
    # channel = 400  #User input here
    #
    # freqs, datetime_array_, Sxx = build_spectrogram_data(
    #     all_files, channel=channel, sampling_frequency=500, nperseg=4096, noverlap=3000
    # )
    #
    # spectrogram_plot(Sxx, freqs, datetime_array_, savepath, channel=channel, dB=False)

    '''VISUALIZE LARC SURVEY - ORIGINAL DEPLOYMENT DATE 7/30'''


    #TODO: normalizing the strain by the distance from shore may help with fading in the middle. Not sure if you can do this, decay in strain signal may be related nonlinearly with distance
