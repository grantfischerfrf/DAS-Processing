import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import maximum_filter1d
import h5py
import os
import re
import numpy as np
import glob
from tqdm import tqdm
from FRF_DAS_processing import read_dasFile, sequence_number, time2dateTime, strain2phase
from scipy.signal import welch, spectrogram


def load_processedData(processed_data_path):

    with h5py.File(processed_data_path, 'r') as f:  #open hdf5 file and read the strain and time datasets
        strain = f['strain'][:]  # shape (time, N_channels)
        time = f['time'][:]  # shape (time,)

    return strain, time


def plot_timeSeries(file_paths, channel:int, save_path:str, t_start=None, t_end=None, envelope=False):

    #plot a timeseries given a list of strain file paths.
    fig, axes = plt.subplots(1, 1, figsize=(20, 8))

    tqdm.write(f'Plotting {len(file_paths)} files...')
    for file in tqdm(file_paths, colour='#ff5733'): #loop through all file paths

        match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})',
                          file)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
            timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
            if timestamp_utc < t_start or timestamp_utc > t_end:
                continue
        else:
            print(f"No timestamp found in file name: {file}")  # break the loop and inspect file names instead
            break

        strain, time = load_processedData(file)

        #convert strain to phase shifts
        phase = strain2phase(strain)

        datetime_array = time2dateTime(time)  # convert unix to datetime for visualizations
        if not envelope:
            axes.plot(datetime_array[::100], phase[::100, channel], color='blue')
        else:
            #find phase envelope
            phase_envelope = np.abs(hilbert(phase[:, channel]))
            # find peaks in envelope
            peaks, _ = find_peaks(phase_envelope, distance=10, prominence=0.0001)

            # plot only the top envelope points
            axes.plot(
                datetime_array[peaks][::100],
                phase_envelope[peaks][::100],
                color='red'
            )

    axes.set_title(f'Channel {channel}: Phase Shift v Time')
    axes.set_ylabel('Phase Shift (rad)')
    axes.set_xlabel('Time (UTC)')
    if t_start is not None and t_end is not None:
        axes.set_xlim(t_start, t_end)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{starttime.strftime("%Y_%m_%d_%H_%M_%S")}_timeseries_{channel}.png')
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
    plt.show()
    plt.close('all')


def build_spectrogram_data(all_files, channel:int, sampling_frequency=500,
                             nperseg=512, noverlap=400):
    """
    Concatenate raw strain across files and compute a spectrogram.
    If channel is None, averages across all channels before computing the spectrogram
    """
    strain_list = []
    time_list = []  #FIXME: careful with the memory here

    for file_path in tqdm(all_files):
        raw_data = read_dasFile(file_path)
        # strain, time = load_processedData(file_path)
        strain = raw_data['RawData']  # shape (n_samples, n_channels)
        time = raw_data['RawDataTime'] / 1e6  # microseconds to seconds


        strain_list.append(strain[:, channel]) # select a specific channel

        time_list.append(time)

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


def get_signal_amplitude(strain, channel:int, sampling_frequency=2):

    # use the strain timeseries and welches method to get the PSD of the strain signal across frequency

    # nperseg = number of samples per segment for Welch's method.
    # Welch splits the signal into overlapping segments of this length,
    # computes an FFT-based periodogram for each, and averages them
    # to get a smoother, lower-variance PSD estimate.

    #   larger nperseg  -> finer frequency resolution (df = fs/nperseg), but fewer
    #                      segments to average -> noisier PSD
    #   smaller nperseg -> coarser frequency resolution, but more segments
    #                      averaged -> smoother, more stable PSD

    freqs, psd = welch(strain[:, channel], fs=sampling_frequency,
                       nperseg=256)  # return psd (strain**2/Hz) and frequency bins (Hz) - iterate over time, if chosing a channel this happens by default

    wave_band = (freqs >= 0.05) & (
                freqs <= 1)  # boolean mask of incident wave frequency band outlined in the DAS paper

    Amplitude = np.sqrt(np.trapezoid(psd[wave_band], freqs[
        wave_band]))  # integrate across the wave band to get the rms strain amplitude at the particular channel

    return Amplitude


def get_waveNumber(strain, channel_start=250, channel_end=300, dx = 1.6, sampling_frequency=2, peak=True, debug=False):
    #use real fft to pull the peak wave number from the signal. Need to preserve phase information so no PSD.
    # real fft because we only need the positive values of the frequency spectrum (bc we only need frequencies in the wave band) - hermitian symmetry
    strain = strain[:, channel_start:channel_end] #defaults to the cross-shore section of the cable

    n_samples, n_channels = np.shape(strain) #shape of the strain array

    channel_positions = dx * np.arange(n_channels) #builds the x position (in meters) of the channels

    # exposed_mask = (channel_positions < 90) | (channel_positions > 160) #exposed portion of the channel

    X = np.fft.rfft(strain, axis=0) #compute the real fft of the strain data along the time axis
    freqs = np.fft.rfftfreq(n_samples, d=1/sampling_frequency) #return the discrete fourier transform sampling frequencies

    wave_band = (freqs >= 0.05) & (freqs <= 1)  # boolean mask of incident wave frequency band outlined in the DAS paper

    band_idx = np.where(wave_band)[0] #actual rows of X that correspond to the wave band

    power_in_band = np.sum(np.abs(X[band_idx, :])**2, axis=1) #for each frequency bin within the wave band, sum up the power |X|**2 across all channels. How much energy at a particular frequency over the entire array

    if debug:
        #debugging - sanity visualization of the power spectrum
        plt.plot(freqs[band_idx], power_in_band)
        plt.show()
        plt.close('all')

    if peak:
        freq_index = band_idx[np.argmax(power_in_band)] #find the frequency bin index (in the wave band) with the most energy

        phi = np.angle(X[freq_index, :])  # get the phase at the one dominant frequency across all channels (phi(x))

        phi_unwrapped = np.unwrap(
            phi)  # because np.angle only returns (-pi, pi] if the phase goes past pi it jumps to -pi. This makes the sequence continuous instead

        k, phi_o = np.polyfit(channel_positions, phi_unwrapped,
                              1)  # fits a straight line -> phi(x) = kx + phi_o. Through the position (x) and unwrapped phase points (phi(x)) using least squares,

    else: #take a power weighted average of all wave numbers across every frequency bin in the wave band. helps avoid problems generated by a noisy or bimodal spectrum
        k_per_freq = []  #empty array for k's across the entire wave band
        power_per_freq = [] #empty array for power across the entire wave band
        for idx in band_idx:  #for frequency bins inside the wave band...
            phi_unwrapped = np.unwrap(np.angle(X[idx, :])) # because np.angle only returns (-pi, pi] if the phase goes past pi it jumps to -pi. This makes the sequence continuous instead
            k, phi_o = np.polyfit(channel_positions, phi_unwrapped, 1) # fits a straight line (deg=1) -> phi(x) = kx + phi_o. Through the position (x) and unwrapped phase points (phi(x)) using least squares. k in units of rad/m
            k_per_freq.append(k) #append k to the list
            power_per_freq.append(np.sum(np.abs(X[idx, :]) ** 2)) #append the power at the frequency bin to the list.

        k_per_freq = np.array(k_per_freq)
        power_per_freq = np.array(power_per_freq)

        k = np.average(k_per_freq, weights=power_per_freq) #take the power weighted average k

    if debug:
        # residuals = actual measured phase (phi_unwrapped, from the FFT) minus the
        # predicted phase from the linear fit (k*x + phi_o), evaluated at each channel
        # position. Large/systematic residuals indicate the true phase isn't well
        # described by a single constant k over this span (e.g. shoaling-driven
        # spatial variation in k).
        residuals = phi_unwrapped - np.polyval([k, phi_o], channel_positions) #pulls the phase (phi(x)) evaluated at each channel position, then subtract the phase from the fourier transform
        plt.figure(figsize=(8, 4))
        plt.scatter(channel_positions, residuals, s=15)
        plt.axhline(0, color='k', linewidth=0.8)
        plt.xlabel('Along Cable Distance (m)')
        plt.ylabel('Phase residual (rad)')
        plt.title('Residuals of linear phase fit vs. position')
        plt.show()

    # returns only the slope (Local wavenumber k). phi_o would be the y intercept.
    #  k is estimated from a single least-squares fit across all channels at the
    #  peak frequency (or power weighted average), rather than from point-to-point phase differences, making it
    #  more robust to per-channel noise in the phase measurements.
    return k


def PSD_One_Channel(file_paths, channel, nperseg, noverlap, sampling_frequency=2):

    #TODO: add time mask component here at some point. also make it so multiple channels can be plot on the same graph

    tqdm.write(f'Plotting {len(file_paths)} files...')
    for file in tqdm(file_paths, colour='#ff5733'):  # loop through all file paths
        strain, time = load_processedData(file)

        datetime_array = time2dateTime(time)
        save_time = datetime_array[0]

        #obtain the power spectral density (and sample frequencies) of strain array
        freqs, psd = welch(strain[:, channel], fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap)

        fig, axes = plt.subplots(1, 1, figsize=(12, 8))

        axes.plot(freqs, psd, color='red', linewidth=1)
        axes.set_xlabel('Frequency (Hz)')
        axes.set_ylabel('PSD (strain**2/Hz)')
        axes.set_title(f'Power Spectral Density at Channel {channel}')
        axes.grid()

        plt.tight_layout()
        plt.savefig(f'./figures/PSD_one_channel/{save_time.strftime("_%Y-%m-%d_%H-%M-%S")}_PSD_at_Channel_{channel}.png')
        # plt.show()
        plt.close('all')



def k_and_amp_timeseries(file_paths, save_path, channel_start:int, channel_end:int, channel_spacing=1.6, sampling_frequency=2):

    #for each file, generate a wavenumber and amplitude and plot a timeseries of values
    fig, axes = plt.subplots(2, 1, figsize=(20, 8))

    channel = int((channel_start + channel_end) / 2)

    times = [] #empty lists to accumulate data
    amps = []
    ks = []

    tqdm.write(f'Plotting {len(file_paths)} files...')
    for file in tqdm(file_paths, colour='#ff5733'):  # loop through all file paths
        strain, time = load_processedData(file)  #pull strain and time data

        microstrain = strain * 1e6 #convert to microstrain for scaling

        datetime_array = time2dateTime(time)  # convert unix to datetime for visualizations

        times.append(datetime_array[0])
        amps.append(get_signal_amplitude(microstrain, channel, sampling_frequency))
        ks.append(get_waveNumber(strain, channel_start, channel_end,
                                 channel_spacing, sampling_frequency))

    axes[0].plot(times, amps, color='red', linewidth=1)
    axes[1].plot(times, ks, color='blue', linewidth=1)

    axes[0].set_xlabel('Time (UTC)')
    axes[0].set_ylabel('Amplitude (microstrain)')
    axes[0].set_title(f'RMS Strain Amplitude at Channel {channel}')
    axes[0].grid()

    axes[1].set_xlabel('Time (UTC)')
    axes[1].set_ylabel('wavenumber (rad/m)')
    axes[1].set_title(f'Local Wavenumber Channel {channel_start} to Channel {channel_end}')
    axes[1].grid()

    plt.tight_layout()
    plt.savefig(f'{save_path}/amp_and_k_timeseries_{channel}.png')
    plt.show()
    plt.close('all')

def PSD_across_channels(file_paths, nperseg, noverlap, sampling_frequency=2, channel_spacing=1.6, averaged=False,
                        dB=False):

    tqdm.write(f'Plotting {len(file_paths)} files...')
    psd_to_avg = []  # accumulate psd to average over

    # sample file to get the shape of the strain array
    sample_strain, sample_time = load_processedData(file_paths[0])

    N_samples, N_channels = sample_strain.shape  # get shape of the strain array

    # set up grid of cross shore channels
    channel_positions = channel_spacing * np.arange(N_channels)

    for file in tqdm(file_paths, colour='#ff5733'):  # loop through all file paths

        strain, time = load_processedData(file)  # pull data

        # strain array of shape (t, x)
        freqs, psd = welch(strain, fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap,
                           axis=0)  # iterate through time, return PSD of shape (f, x)

        if averaged:
            psd_to_avg.append(psd)

        if dB:
            psd = 10 * np.log10(psd + 1e-20)  # convert to dB avoiding log(0) - if you want the averaged plot in dB

        vmin = np.percentile(psd,
                             5)  # set the colorbar floor to a value below the 5th percentile of the data to avoid outliers dominating the color scale
        vmax = np.percentile(psd,
                             99.9)  # set the colorbar ceiling to a value above the 99th percentile of the data to avoid outliers dominating the color scale

        if not averaged:
            fig, axes = plt.subplots(1, 1, figsize=(12, 8))
            pcm = axes.pcolormesh(channel_positions, freqs, psd, shading='auto', cmap='viridis', vmin=vmin,
                                  vmax=vmax)  # plot the average frequency content from the 5-minute file.
            time_array = time2dateTime(time)
            save_time = time_array[0]

            cbar = fig.colorbar(pcm, ax=axes)
            cbar.set_label("Power Spectral Density (Strain**2/Hz)")

            axes.set_xlabel('Along Cable Distance (m)')
            axes.set_ylabel('Frequency (Hz)')
            axes.set_title(f'PSD v Channel Distance {save_time.strftime("_%Y-%m-%d_%H:%M:%S")} UTC')

            plt.tight_layout()
            plt.savefig(f'./figures/freq_dist_plots/freq_dist_{save_time.strftime("_%Y-%m-%d_%H-%M-%S")}.png')
            # plt.show()
            plt.close('all')

    if averaged:
        averaged_psd = np.average(psd_to_avg, axis=0)

        vmin = np.percentile(averaged_psd,
                             5)  # set the colorbar floor to a value below the 5th percentile of the data to avoid outliers dominating the color scale
        vmax = np.percentile(averaged_psd,
                             99.9)  # set the colorbar ceiling to a value above the 99th percentile of the data to avoid outliers dominating the color scale

        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        pcm = axes.pcolormesh(channel_positions, freqs, averaged_psd, shading='auto', cmap='viridis', vmin=vmin,
                              vmax=vmax)

        cbar = fig.colorbar(pcm, ax=axes)
        cbar.set_label("Power Spectral Density (Strain**2/Hz)")

        axes.set_xlabel('Along Cable Distance (m)')
        axes.set_ylabel('Frequency (Hz)')
        axes.set_title(f'PSD v Channel Distance Averaged')

        plt.tight_layout()
        plt.savefig(f'./figures/freq_dist_plots/Averaged_freq_dist.png')
        plt.show()
        plt.close('all')

#TODO: save path and time filter here

if __name__ == "__main__":

    #processed data
    processed_data_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)
    # strain, time = load_processedData(processed_data_paths) #pull processed strain and time data

    # datetime_array = time2dateTime(time) #convert unix to datetime for visualizations

    # define starttime and endtime for plotting
    starttime = datetime(2026, 6, 6, 0, 0, 0, tzinfo=timezone.utc)
    endtime = datetime(2026, 6, 6, 23, 59, 59, tzinfo=timezone.utc)

    '''CREATE WATERFALL PLOT'''
    # # savepath = './figures/20260601'
    # channel_start = 0  #define start and end channels for plotting
    # # channel_end = strain.shape[1]
    # channel_spacing=1.6 #meters
    # # waterFall(strain, datetime_array, channel_spacing, starttime, endtime, channel_start, channel_end, savepath)
    #
    # filepaths = sorted(glob.glob('./data/for_OSU_processed/20260601/*.h5'), key=sequence_number)
    # count = 0
    # for file in tqdm(filepaths):
    #     strain, time = load_processedData(file)
    #
    #     channel_end = strain.shape[1]
    #
    #     savepath = f'./figures/20260601/Waterfall_'
    #
    #     waterFall(strain, time2dateTime(time), channel_spacing, channel_start, channel_end, savepath)


    '''CREATE SPECTROGRAM'''
    # savepath = './figures/spectrogram'
    #
    # #raw data
    # raw_data_path = os.getcwd() + '/data/raw_614/'
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

    '''TIMESERIES PLOTS'''
    plot_timeSeries(processed_data_paths, channel=700, save_path='./figures', t_start=starttime, t_end=endtime, envelope=False)

    '''WAVENUMBER AND AMPLITUDE TIMESERIES AT A PARTICULAR CHANNEL'''
    # k_and_amp_timeseries(processed_data_paths, save_path='./figures', channel_start=600, channel_end=620, channel_spacing=1.6)

    '''PSD ACROSS CHANNELS'''
    # nperseg = 256
    # noverlap = nperseg // 2
    # # PSD_across_channels(processed_data_paths, nperseg, noverlap, averaged=False, dB=False)

    '''PSD AT ONE CHANNEL'''
    # channel = 310
    # PSD_One_Channel(processed_data_paths, channel, nperseg, noverlap)


    #TODO: normalizing the strain by the distance from shore may help with fading in the middle. Not sure if you can do this, decay in strain signal may be related nonlinearly with distance
    #TODO: normalize by z score???

    #TODO: look at the Radar data.

    #TODO: play with the sampling window for the power spectral density calculation - some data is fuzzy, maybe use log scale for IG signal?

    #TODO: need a clipping function for time
