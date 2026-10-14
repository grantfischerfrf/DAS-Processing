import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import maximum_filter1d
import h5py
import os
import re
import numpy as np
import glob
from tqdm import tqdm
from scipy.signal import welch, spectrogram
from utils import clip_by_time, sequence_number, read_dasFile, time2dateTime, load_processedData, get_DAS_PSD_burst, \
    get_orientation, strain2phase, get_burst_files, get_AWAC_data, smooth_spectrum, integrate_spectrum, compute_bulk_statistics, \
    rebin_psd
from Calibration import wavenumber



def plot_timeSeries(file_paths, channel:int, save_path:str, t_start=None, t_end=None, envelope=False):

    #plot a timeseries given a list of strain file paths.
    fig, axes = plt.subplots(1, 1, figsize=(20, 8))

    #clip filepaths by time stamps
    file_paths = clip_by_time(file_paths, t_start, t_end)

    tqdm.write(f'Plotting {len(file_paths)} files...')
    for file in tqdm(file_paths, colour='#ff5733'): #loop through all file paths

        # match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})',
        #                   file)
        # if match:
        #     timestamp_str = match.group(1)
        #     timestamp = datetime.strptime(timestamp_str, "%Y_%m_%d_%H_%M_%S")
        #     timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
        #     if timestamp_utc < t_start or timestamp_utc > t_end:
        #         continue
        # else:
        #     print(f"No timestamp found in file name: {file}")  # break the loop and inspect file names instead
        #     break

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
    plt.savefig(f'{save_path}/{t_start.strftime("%Y_%m_%d_%H_%M_%S")}_timeseries_{channel}.png')
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


def build_spectrogram_data(all_files, channel:int, start_time:datetime, end_time:datetime, sampling_frequency=500,
                             nperseg=512, noverlap=400):
    """
    Concatenate raw strain across files and compute a spectrogram.
    If channel is None, averages across all channels before computing the spectrogram
    """
    strain_list = []
    time_list = []  #FIXME: careful with the memory here

    all_files = clip_by_time(all_files, start_time, end_time) #clip file paths by timestamps

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
            # because np.angle only returns (-pi, pi] if the phase goes past pi it jumps to -pi. This makes the sequence continuous instead
            phi_unwrapped = np.unwrap(np.angle(X[idx, :]))

            # fits a straight line (deg=1) -> phi(x) = kx + phi_o. Through the position (x) and unwrapped phase points (phi(x)) using least squares. k in units of rad/m
            k, phi_o = np.polyfit(channel_positions, phi_unwrapped, 1)

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


def PSD_One_Channel(file_paths, channel:list, nperseg, noverlap, start_time:datetime, end_time:datetime, sampling_frequency=2):

    file_paths = clip_by_time(file_paths, start_time, end_time)

    tqdm.write(f'Plotting {len(file_paths)} files...')

    for file in tqdm(file_paths, colour='#ff5733'):  # loop through all file paths
        strain, time = load_processedData(file)

        datetime_array = time2dateTime(time)
        save_time = datetime_array[0]

        for ch in channel:
            #obtain the power spectral density (and sample frequencies) of strain array
            freqs, psd = welch(strain[:, ch], fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap)

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


def k_and_amp_timeseries(file_paths, save_path, channel_start:int, channel_end:int, start_time:datetime, end_time:datetime, channel_spacing=1.6, sampling_frequency=2):

    #for each file, generate a wavenumber and amplitude and plot a timeseries of values
    fig, axes = plt.subplots(2, 1, figsize=(20, 8))

    channel = int((channel_start + channel_end) / 2)

    times = [] #empty lists to accumulate data
    amps = []
    ks = []

    file_paths = clip_by_time(file_paths, start_time, end_time)

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


def PSD_across_channels(file_paths, nperseg, noverlap, start_time:datetime, end_time:datetime, sampling_frequency=2, channel_spacing=1.6, averaged=False,
                        dB=False, savepath='./figures/freq_dist_plots'):

    tqdm.write(f'Plotting {len(file_paths)} files...')
    psd_to_avg = []  # accumulate psd to average over

    # sample file to get the shape of the strain array
    sample_strain, sample_time = load_processedData(file_paths[0])

    N_samples, N_channels = sample_strain.shape  # get shape of the strain array

    # set up grid of cross shore channels
    channel_positions = channel_spacing * np.arange(N_channels)

    file_paths = clip_by_time(file_paths, start_time, end_time)

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
            plt.savefig(f'{savepath}/freq_dist_{save_time.strftime("_%Y-%m-%d_%H-%M-%S")}.png')
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
        plt.savefig(f'{savepath}/Averaged_freq_dist.png')
        plt.show()
        plt.close('all')


def Calibrated_DAS_v_Model(das_data_paths, model_data_dir, save_path, target_time:datetime, channel:int, wave_band_min=0.05, wave_band_max=0.3, sampling_frequency=2):

    orientation = get_orientation(channel) #get the orientation of the selected DAS channel

    burst_files = get_burst_files(das_data_paths, target_time, 35) #obtain all files inside a particular burst at the chosen target time. careful of hard coded burst duration

    # pull AWAC spectrum
    model_data_path = os.path.join(model_data_dir, f'Calib_{target_time.strftime("%Y-%m-%d_%H_%M_%S")}.npz') #define awac data path
    model_data = np.load(model_data_path, allow_pickle=True) #read the data
    model_data = model_data['arr_0'].item() #open the dictionary and get past the initial format

    C = model_data['calibration_coefficient'] #calibration coefficient generated through spectral area matching. Using the script run_Emprirical_Calibration.py
    # model_Sf = model_data['awac_corrected_frequency_spectrum']#awac frequency spectrum
    model_Sf = model_data['awac_cable_weighted_frequency_spectrum'] #awac frequency spectrum weighted by the incident wave direction - energy projected along the cable
    model_freqs = model_data['awac_frequency_bins'] #awac frequency bins
    wave_band = (model_freqs >= wave_band_min) & (model_freqs <= wave_band_max)  # define the wave band, same as das but has different bin widths so needs a different variable.

    h = model_data['water_depth'] #water depth along the cable
    calibration_flag = model_data['calibration_flag']  # flag for whether the calibration worked or failed a check, 1 is good, 0 means fail

    nperseg = round(sampling_frequency / np.diff(model_freqs)[0]) #assumes constant spacing of the model frequency bins, computes the nperseg such that PSD bins equal those of the model)

    das_freqs, das_PSD_raw = get_DAS_PSD_burst(burst_files, nperseg=nperseg, sampling_frequency=sampling_frequency) #compute the power spectral density over the burst using welch's method
    # rebin onto the full AWAC grid, then clip both with the same mask
    das_PSD_m = rebin_psd(das_freqs, das_PSD_raw[:, channel], model_freqs)
    das_PSD = das_PSD_m[wave_band] #clip the power spectral density over the wave band

    # define constants
    rho = 1025 #kg/m^3 density of seawater
    g = 9.81 #m/s^2 gravitational acceleration
    h_local = h[channel] #water depth at chosen DAS channel

    model_T = 1 / model_freqs  #convert awac frequency bins to period
    model_k = wavenumber(h_local, model_T)  # awac wavenumber at each period / frequency bin
    model_Kp = 1 / np.cosh(model_k * h_local)  # pressure response factor at each awac wavenumber

    model_freqs = model_freqs[wave_band]  # constrain the awac frequencies over the wave band
    model_Kp = model_Kp[wave_band] #constrain the awac pressure response factor over the wave band

    Seta_f_model = model_Sf[channel, wave_band].T

    # convert awac SSE spectrum to pressure spectrum.
    Sp_f_model = (Seta_f_model * (rho * g * model_Kp) ** 2) / 1e8  # convert pascals to decibars, (must square for units to match)
    # correct DAS spectrum to convert from phase_shifts to pressure
    Sp_f_das = (das_PSD * C[channel]) / 1e8 # the calibration coefficient is for a pressure spectra - convert from pascals to dbars

    # integrate under the spectra - should yield essentially the same value if calibration was done correctly
    Prms_das = np.sqrt(np.trapezoid(Sp_f_das, model_freqs))  #compute rms pressure value from the das pressure spectrum
    Prms_model = np.sqrt(np.trapezoid(Sp_f_model, model_freqs)) #compute rms pressure value from the awac pressure spectrum

    #print out metrics
    print(f'Channel: {channel}, {orientation}')
    print(f'PRESSURE RMS VALUES: \n DAS: {Prms_das} \n AWAC: {Prms_model}')
    print("Calibration Flag: Pass" if calibration_flag[channel] else "Calibration Flag: Fail")

    # plot the spectra on the same plot
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))

    line1 = axes.plot(model_freqs, Sp_f_model, color='blue', label='Modeled Pressure PSD (AWAC)')
    line2 = axes.plot(model_freqs, Sp_f_das, color='red', label='DAS Pressure PSD (Calibrated)')
    axes1 = axes.twinx()
    line3 = axes1.plot(model_freqs, das_PSD, color='green', linestyle='--', label='DAS Phase Shift PSD')
    axes1.set_ylabel('Phase Shift PSD (Rad²/Hz)', color='green')
    axes1.tick_params(axis='y', color='green', labelcolor='green', which='both')
    axes.set_xlabel('Frequency (Hz)')
    axes.set_ylabel('Pressure PSD (dbar²/Hz)')
    axes.set_title(f'Freq. Independent Spectral Area Matching; Channel: {channel}, Time: {target_time.strftime("%Y-%m-%d %H:%M:%S")}')

    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]

    axes.legend(lines, labels, loc='best')

    plt.tight_layout()
    plt.savefig(f'{save_path}/channel_{channel}_time_{target_time}_area.png')
    plt.show()
    plt.close('all')


def Calibrated_DAS_v_Model_allBursts(das_data_paths, model_data_paths, save_path, wave_band_min=0.05, wave_band_max=0.3):

    #calculate the rms pressure from the das and modeled awac for all VALID channels for all bursts
    n_hours = len(model_data_paths) #one burst per hour, files are saved by the hour.
    model_data_dir = os.path.dirname(model_data_paths[0]) #model directory name

    #initialize the figure for plotting
    fig, axes = plt.subplots(1, 1, figsize=(12, 12))
    axes.axline((0, 0), slope=1, color='black', linestyle='--') #diagonal line for one to one plot
    axes.set_xlabel('DAS RMS Pressure (dbar)')
    axes.set_ylabel('Modeled AWAC RMS Pressure (dbar)')

    sum_squared_error = 0.0 #error placeholder to fill incrementally during the loops
    n_valid = 0
    for hour in tqdm(range(n_hours), colour='yellow'):
        target_time = sequence_number(model_data_paths[hour])
        burst_files = get_burst_files(das_data_paths, target_time, 35)  # obtain all files inside a particular burst at the chosen target time. careful of hard coded burst duration

        das_freqs, das_PSD = get_DAS_PSD_burst(burst_files, nperseg=256, sampling_frequency=2)  # compute the power spectral density over the burst using welch's method
        das_wave_band = (das_freqs >= wave_band_min) & (das_freqs <= wave_band_max)  # define the wave frequency band for the das
        das_PSD = das_PSD[das_wave_band, :]  # clip the power spectral density over the wave band
        das_freqs = das_freqs[das_wave_band]  # clip the frequency bins by the wave band

        # pull modeled AWAC spectrum
        model_data_path = os.path.join(model_data_dir, f'Calib_{target_time.strftime("%Y-%m-%d_%H_%M_%S")}.npz')  # define modeled awac data path
        model_data = np.load(model_data_path, allow_pickle=True)  # read the data
        model_data = model_data['arr_0'].item()  # open the dictionary and get past the initial format

        C = model_data['calibration_coefficient']  # calibration coefficient generated through spectral area matching. Using the script run_Emprirical_Calibration.py
        model_Sf = model_data['awac_corrected_frequency_spectrum']  # modeled awac directional spectrum
        model_freqs = model_data['awac_frequency_bins']  # modeled awac frequency bins
        model_wave_band = (model_freqs >= 0.05) & (model_freqs <= 0.3)  # define the wave band, same as das but has different bin widths so needs a different variable.
        h = model_data['water_depth']  # water depth along the cable
        calibration_flag = model_data['calibration_flag']  # flag for whether the calibration worked or failed a check, 1 is good, 0 means fail

        # define constants
        rho = 1025  # kg/m^3 density of seawater
        g = 9.81  # m/s^2 gravitational acceleration
        n_channels = C.shape[0]
        for ch in range(n_channels):
            if not calibration_flag[ch]:
                continue

            h_local = h[ch]  # water depth at chosen DAS channel

            model_T = 1 / model_freqs  # convert modeled awac frequency bins to period
            model_k = wavenumber(h_local, model_T)  # modeled awac wavenumber at each period / frequency bin
            model_Kp = 1 / np.cosh(model_k * h_local)  # pressure response factor at each modeled awac wavenumber

            # integrate modeled awac directional spectrum over directional bins
            model_freqs_band = model_freqs[model_wave_band]  # constrain the modeled awac frequencies over the wave band
            model_Kp_band = model_Kp[model_wave_band]  # constrain the modeled awac pressure response factor over the wave band

            Seta_f_model = model_Sf[ch, model_wave_band].T

            # convert modeled awac SSE spectrum to pressure spectrum.
            Sp_f_model = (Seta_f_model * (rho * g * model_Kp_band) ** 2) / 1e8  # convert pascals to decibars, (must square for units to match)
            # correct DAS spectrum to convert from phase_shifts to pressure
            Sp_f_das = (das_PSD[:, ch] * C[ch]) / 1e8  # the calibration coefficient is for a pressure spectra - convert from pascals to dbars

            # integrate under the spectra - should yield essentially the same value if calibration was done correctly
            Prms_das = np.sqrt(np.trapezoid(Sp_f_das, das_freqs))  # compute rms pressure value from the das pressure spectrum
            Prms_model = np.sqrt(np.trapezoid(Sp_f_model, model_freqs_band))  # compute rms pressure value from the modeled awac pressure spectrum

            axes.scatter(Prms_das, Prms_model, color='blue', s=10, marker='P', alpha=0.5)

            error = Prms_das - Prms_model
            sum_squared_error += error ** 2
            n_valid += 1

    rmse = np.sqrt(sum_squared_error / n_valid)
    print(f"RMSE = {rmse:.6f} dbar")

    axes.set_title(f'DAS v Modeled AWAC Spectral Area Matching Calibration, RMSE:{rmse:.6f} (dbar)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'RMSE_plot.png'))
    plt.show()
    plt.close('all')


def AWAC_modelVobs(awac_data_dir, model_data_dir):

    '''compare the bulk wave parameters produced by the model and the in situ AWACs at xFRF 300m and xFRF 600m'''

    #model data
    model_data_paths = sorted(glob.glob(os.path.join(model_data_dir, '*.npz')), key=sequence_number)
    n_hours = len(model_data_paths) #model data is saved one file every hour so n_hours is just the number of file paths

    awac_300_path, = glob.glob(os.path.join(awac_data_dir, '*300*.nc'))  #find the 300m awac path and unpack it from the list
    awac_600_path, = glob.glob(os.path.join(awac_data_dir, '*600*.nc')) #find the 600m awac path and unpack it from the list

    #channels/xlocations in the model grid that correspond to the 300m and 600m awacs
    awac_300_channel = 208
    awac_600_channel = 402

    # awac observed data
    S_f_t, freqs, dirs, time, dtheta, df, T, gaugeDepth, gaugeElevation, bottomElevation = get_AWAC_data(awac_300_path)  #shape (N_samples/time, n_freqs, n_dirs)
    wave_band = (freqs >= 0.05) & (freqs <= 0.3)

    # integrate the directional spectrum over the directional bins to get the frequency spectrum, shape (N_samples/time, n_freqs)
    S_f = integrate_spectrum(S_f_t[:, wave_band, :], dirs, axis=-1, integrate_over='direction')

    # integrate the directional spectrum over the frequency bins to get the 1D directional spectrum, shape (N_samples/time, n_dirs)
    S_t = integrate_spectrum(S_f_t[:, wave_band, :], freqs[wave_band], axis=1, integrate_over='frequency')

    #compute HS before smoothing
    awac_Hs_all, _, _ = compute_bulk_statistics(S_f, S_t, freqs, dirs, wave_band)

    #smooth the AWAC spectra
    S_t = smooth_spectrum(S_t, window=5, filter_type='gaussian', mode='wrap')
    S_f = smooth_spectrum(S_f, window=5, filter_type='gaussian', mode='wrap')

    #bulk wave statistics of the observed awac data (N_samples/time,)
    _, awac_Tp_all, awac_thetap_all = compute_bulk_statistics(S_f, S_t, freqs, dirs, wave_band)

    #initialize plot for visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    axes[0].set_title('Significant Wave Height (Hs) Comparison')
    axes[0].set_xlabel('Observed AWAC Hs (m)')
    axes[0].set_ylabel('Modeled AWAC Hs (m)')
    axes[0].axline((0, 0), slope=1, color='black', linestyle='--')  # diagonal line for one to one plot

    axes[1].set_title('Peak Period (Tp) Comparison')
    axes[1].set_xlabel('Observed AWAC Tp (s)')
    axes[1].set_ylabel('Modeled AWAC Tp (s)')
    axes[1].axline((0, 0), slope=1, color='black', linestyle='--')  # diagonal line for one to one plot

    axes[2].set_title('Peak Direction (θp) Comparison')
    axes[2].set_xlabel('Observed AWAC θp (degrees)')
    axes[2].set_ylabel('Modeled AWAC θp (degrees)')
    axes[2].axline((0, 0), slope=1, color='black', linestyle='--')  # diagonal line for one to one plot

    fig.suptitle('Modeled vs Observed AWAC (300m xFRF) Bulk Wave Parameter Comparison')

    #error variables
    Hs_sumSquared_error = 0.0
    Tp_sumSquared_error = 0.0
    thetap_sumSquared_error = 0.0

    count = 0
    for hour in tqdm(range(n_hours), colour='yellow'):
        target_time = sequence_number(model_data_paths[hour]) #target time to match the model and awac to

        idx = np.argmin(np.abs(np.datetime64(target_time) - time)) #find the index of the closest time in the awac observational data to the target time

        model_data_path = os.path.join(model_data_dir, f'Calib_{target_time.strftime("%Y-%m-%d_%H_%M_%S")}.npz')  # define modeled awac data path
        model_data = np.load(model_data_path, allow_pickle=True)  # read the data
        model_data = model_data['arr_0'].item()  # open the dictionary and get past the initial format

        model_S_f_t = model_data['awac_corrected_directional_spectrum'] #2D wave spectrum (n_channels, n_freqs, n_dirs)

        # integrate the directional spectrum over the directional bins to get the 1D frequency spectrum, shape (n_channels, n_freqs)
        model_S_f = integrate_spectrum(model_S_f_t[:, wave_band, :], bins=dirs, axis=-1, integrate_over='direction')

        # integrate the directional spectrum over the frequency bins to get the 1D directional spectrum, shape (n_channels, n_dirs)
        model_S_t = integrate_spectrum(model_S_f_t[:, wave_band, :], bins=freqs[wave_band], axis=1, integrate_over='frequency')

        #compute Hs before smoothing
        model_Hs_all, _, _ = compute_bulk_statistics(model_S_f, model_S_t, freqs, dirs, wave_band)

        model_S_t = smooth_spectrum(model_S_t, window=5, filter_type='gaussian', mode='wrap') #smooth the DAS spectra
        model_S_f = smooth_spectrum(model_S_f, window=5, filter_type='gaussian', mode='wrap')

        #sanity check - plot the frequency/direction spectrum of the awac measurement and the model for the same time step
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(dirs, S_t[idx, :], color='blue', label='300m awac measurement')
        ax.plot(dirs, model_S_t[awac_300_channel, :], color='red', label='model eval at 300mAWAC')
        ax.legend()
        # plt.show()
        plt.savefig(f'./figures/modelvawac/direction/peakDir_{count}.png')
        plt.close()
        count+=1

        # bulk wave statistics from the model shape: (n_channels,)
        _, model_Tp_all, model_thetap_all = compute_bulk_statistics(model_S_f, model_S_t, freqs, dirs, frequency_band=wave_band)

        #BULK STATISTICS
        awac_Hs = awac_Hs_all[idx] #300m awac Hs at the closest time to the model data
        model_Hs = model_Hs_all[awac_300_channel] #model Hs at the channel corresponding to the 300m awac
        Hs_error = awac_Hs - model_Hs
        Hs_sumSquared_error += Hs_error ** 2
        axes[0].scatter(awac_Hs, model_Hs, color='blue', alpha=0.5)

        awac_Tp = awac_Tp_all[idx] #300m awac Tp at the closest time to the model data
        model_Tp = model_Tp_all[awac_300_channel] #model Tp at the channel corresponding to the 300m awac
        Tp_error = awac_Tp - model_Tp
        Tp_sumSquared_error += Tp_error ** 2
        axes[1].scatter(awac_Tp, model_Tp, color='green', alpha=0.5)

        awac_thetap = awac_thetap_all[idx] #300m awac thetap at the closest time to the model data
        model_thetap = model_thetap_all[awac_300_channel] #model thetap at the channel corresponding to the 300m awac
        thetap_error = (awac_thetap - model_thetap + 180) % 360 - 180  #force range into -180 to 180 degrees to avoid issues with circularity
        thetap_sumSquared_error += thetap_error ** 2
        axes[2].scatter(awac_thetap, model_thetap, color='red', alpha=0.5)

    #calculate RMSE for each bulk statistic
    Hs_RMSE = np.sqrt(Hs_sumSquared_error / n_hours)
    Tp_RMSE = np.sqrt(Tp_sumSquared_error / n_hours)
    thetap_RMSE = np.sqrt(thetap_sumSquared_error / n_hours)
    axes[0].annotate(f'RMSE = {Hs_RMSE:.2f} m', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='blue')
    axes[1].annotate(f'RMSE = {Tp_RMSE:.2f} s', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='green')
    axes[2].annotate(f'RMSE = {thetap_RMSE:.2f} degrees', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='red')
    plt.tight_layout()
    plt.savefig('./figures/model_v_300mAWAC_bulkStats.png')
    plt.show()
    plt.close('all')


def AWAC300_v_AWAC600(awac_data_dir, save_dir, start_time=None, end_time=None):

    if start_time is None and end_time is None:
        start_time = np.datetime64(datetime(2026, 6, 1, 0, 0, 0))
        end_time = np.datetime64(datetime(2026, 6, 14, 8, 0, 0))

    # #get the number of hours to iterate over - the awacs collect on 34 minute bursts on the top of each hour
    # n_hours = int((end_time - start_time).total_seconds() / 3600)

    awac_300_path, = glob.glob(os.path.join(awac_data_dir, '*300*.nc'))  # find the 300m awac path and unpack it from the list
    awac_600_path, = glob.glob(os.path.join(awac_data_dir, '*600*.nc'))  # find the 600m awac path and unpack it from the list

    # Load 300 m AWAC data
    S300_f_t, freqs300, dirs300, time300, dtheta300, df300, T300, \
        gaugeDepth300, gaugeElevation300, bottomElevation300 = get_AWAC_data(awac_300_path)

    # Load 600 m AWAC data
    S600_f_t, freqs600, dirs600, time600, dtheta600, df600, T600, \
        gaugeDepth600, gaugeElevation600, bottomElevation600 = get_AWAC_data(awac_600_path)

    wave_band = (freqs300 >= 0.05) & (freqs300 <= 0.3)  # frequency band for wave statistics - same bins for both awacs

    #Bulk Statistics for 300m AWAC
    S300_f = integrate_spectrum(S300_f_t[:, wave_band, :], dirs300, axis=-1, integrate_over='direction') #1D frequency spectrum (n_samples, n_freqs)
    S300_t = integrate_spectrum(S300_f_t[:, wave_band, :], freqs300[wave_band], axis=1, integrate_over='frequency') #1D direction spectrum (n_samples, n_dirs)

    #compute Hs before smoothing
    Hs300_all, _, _ = compute_bulk_statistics(S300_f, S300_t, freqs300, dirs300, frequency_band=wave_band)

    #smooth the AWAC spectra
    S300_t = smooth_spectrum(S300_t, window=5, filter_type='gaussian', mode='wrap')
    S300_f = smooth_spectrum(S300_f, window=5, filter_type='gaussian', mode='wrap')

    #bulk wave statistics of the observed 300m awac data, shape (N_samples/time,)
    _, Tp300_all, thetap300_all = compute_bulk_statistics(S300_f, S300_t, freqs300, dirs300, frequency_band=wave_band)

    #Bulk Statistics for 600m AWAC
    S600_f = integrate_spectrum(S600_f_t[:, wave_band, :], dirs600, axis=-1, integrate_over='direction') #1D frequency spectrum (n_samples, n_freqs)
    S600_t = integrate_spectrum(S600_f_t[:, wave_band, :], freqs600[wave_band], axis=1, integrate_over='frequency') #1D direction spectrum (n_samples, n_dirs)

    #Compute Hs before smoothing
    Hs600_all, _, _ = compute_bulk_statistics(S600_f, S600_t, freqs600, dirs600, frequency_band=wave_band)

    #smooth the AWAC spectra
    S600_t = smooth_spectrum(S600_t, window=5, filter_type='gaussian', mode='wrap')
    S600_f = smooth_spectrum(S600_f, window=5, filter_type='gaussian', mode='wrap')

    # #sanity plot - awac300 v awac 600 frequency spectrum
    # count = 0
    # for n in range(len(time300)):
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(freqs300[wave_band], S300_f[n, :], label='AWAC 300m', color='blue')
    #     plt.plot(freqs600[wave_band], S600_f[n, :], label='AWAC 600m', color='red')
    #     plt.title(f'Frequency Spectrum Comparison - between awac 300 and awac 600')
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Spectral Density (m^2/Hz)')
    #     plt.legend()
    #     plt.savefig(f'./figures/awac300vawac600/awac300Vawac600_{count}.png')
    #     plt.close('all')
    #     count+=1

    #bulk wave statistics of the observed 600m awac data, shape (N_samples/time,)
    _, Tp600_all, thetap600_all = compute_bulk_statistics(S600_f, S600_t, freqs600, dirs600, frequency_band=wave_band)

    #define a time mask to clip the full arrays
    time_mask300 = (time300 >= start_time) & (time300 <= end_time)
    time_mask600 = (time600 >= start_time) & (time600 <= end_time)

    tol = np.timedelta64(30, 's')  # generous buffer for clock sampling differences
    t300 = time300[time_mask300]
    t600 = time600[time_mask600]

    assert len(t300) == len(t600), "mask lengths differ"
    max_offset = np.max(np.abs(t300 - t600))
    assert max_offset < tol, f"max offset {max_offset} exceeds tolerance"  #assert that the times match

    Hs300 = Hs300_all[time_mask300]
    Tp300 = Tp300_all[time_mask300]
    thetap300 = thetap300_all[time_mask300]

    Hs600 = Hs600_all[time_mask600]
    Tp600 = Tp600_all[time_mask600]
    thetap600 = thetap600_all[time_mask600]

    #compute error statistics
    Hs_RMSE = np.sqrt(np.nanmean((Hs300 - Hs600) ** 2))
    Tp_RMSE = np.sqrt(np.nanmean((Tp300 - Tp600) ** 2))
    thetap_error = (thetap300 - thetap600 + 180) % 360 - 180
    thetap_RMSE = np.sqrt(np.nanmean(thetap_error ** 2))  #avoid circularity issues by wrapping the error to [-180, 180]

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    axes[0].scatter(Hs300, Hs600, color='blue', alpha=0.5)
    axes[0].set_xlabel('Hs 300m (m)')
    axes[0].set_ylabel('Hs 600m (m)')
    axes[0].set_title('Significant Wave Height (Hs) Comparison')
    axes[0].axline((0, 0), slope=1, color='black', linestyle='--')  # diagonal line for one to one plot
    axes[0].annotate(f'RMSE: {Hs_RMSE:.2f} m', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='blue')

    axes[1].scatter(Tp300, Tp600, color='green', alpha=0.5)
    axes[1].set_xlabel('Tp 300m (s)')
    axes[1].set_ylabel('Tp 600m (s)')
    axes[1].set_title('Peak Period (Tp) Comparison')
    axes[1].axline((0, 0), slope=1, color='black', linestyle='--')  # diagonal line for one to one plot
    axes[1].annotate(f'RMSE: {Tp_RMSE:.2f} s', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='green')

    axes[2].scatter(thetap300, thetap600, color='red', alpha=0.5)
    axes[2].set_xlabel('Peak Direction 300m (degrees)')
    axes[2].set_ylabel('Peak Direction 600m (degrees)')
    axes[2].set_title('Peak Direction (θp) Comparison')
    axes[2].axline((0, 0), slope=1, color='black', linestyle='--')  # diagonal line for one to one plot
    axes[2].annotate(f'RMSE: {thetap_RMSE:.2f} degrees', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12, color='red')

    fig.suptitle('AWAC 300m vs 600m Bulk Wave Parameter Comparison')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'awac_bulk_stats_comparison.png'), dpi=300)
    plt.show()
    plt.close('all')


def Calibrated_DAS_freq_dependent_v_Model(das_data_paths, model_data_dir, calibCoeff_path, save_path, target_time, channel, wave_band_min=0.05, wave_band_max=0.3, sampling_frequency=2):
    #compare the calibrated DAS PSD with the frequency dependent calibration factor to the modeled AWAC PSD

    #average C(f) across all bursts - Frequency dependent calibration does not preserve DAS PSD shape for a single burst, it will make it equivalent to the model AWAC PSD
    #in order to make a calibration factor useful take the temporal median of C(t, x, f) across bursts

    day = target_time.day #get the day of the collect

    if type(calibCoeff_path) == str:
        #open the calibration file to get C(x, f) - temporal avg over entire dataset
        C_f = np.load(calibCoeff_path) #shape (n_channels, n_freqs)

    if type(calibCoeff_path) == list:
        #find the matching file to day to get the temporal avg over that day of data
        match = next(  #next works with the iterator (path for path in ...) to return the first value that satisfies the condition put forth
            path for path in calibCoeff_path
            if sequence_number(path) == day
        )
        C_f = np.load(match)

    orientation = get_orientation(channel)  # get the orientation of the selected DAS channel

    # pull Modeled AWAC spectrum
    model_data_path = os.path.join(model_data_dir, f'Calib_{target_time.strftime("%Y-%m-%d_%H_%M_%S")}.npz')  # define awac data path
    model_data = np.load(model_data_path, allow_pickle=True)  # read the data
    model_data = model_data['arr_0'].item()  # open the dictionary and get past the initial format

    # model_Sf = model_data['awac_corrected_frequency_spectrum']  # awac frequency spectrum
    model_Sf = model_data['awac_cable_weighted_frequency_spectrum'] #awac frequency spectrum weighted by the incident wave direction - energy projected along the cable
    model_freqs = model_data['awac_frequency_bins']  # awac frequency bins
    wave_band = (model_freqs >= wave_band_min) & (model_freqs <= wave_band_max)  # define the wave band, same as das but has different bin widths so needs a different variable.

    h = model_data['water_depth']  # water depth along the cable
    calibration_flag = model_data['calibration_flag']  # flag for whether the calibration worked or failed a check, 1 is good, 0 means fail

    burst_files = get_burst_files(das_data_paths, target_time, 35)  # obtain all files inside a particular burst at the chosen target time. careful of hard coded burst duration

    nperseg = round(sampling_frequency / np.diff(model_freqs)[0]) #assumes constant spacing of the model frequency bins, computes the nperseg such that PSD bins equal those of the model

    das_freqs, das_PSD_raw = get_DAS_PSD_burst(burst_files, nperseg=nperseg, sampling_frequency=sampling_frequency)  # compute the power spectral density over the burst using welch's method
    # rebin onto the full AWAC grid, then clip both with the same mask
    das_PSD_m = rebin_psd(das_freqs, das_PSD_raw[:, channel], model_freqs)
    das_PSD = das_PSD_m[wave_band]  # clip the power spectral density over the wave band

    # define constants
    rho = 1025  # kg/m^3 density of seawater
    g = 9.81  # m/s^2 gravitational acceleration
    h_local = h[channel]  # water depth at chosen DAS channel

    model_T = 1 / model_freqs  # convert awac frequency bins to period
    model_k = wavenumber(h_local, model_T)  # awac wavenumber at each period / frequency bin
    model_Kp = 1 / np.cosh(model_k * h_local)  # pressure response factor at each awac wavenumber

    model_freqs = model_freqs[wave_band]  # constrain the awac frequencies over the wave band
    model_Kp = model_Kp[wave_band]  # constrain the awac pressure response factor over the wave band

    Seta_f_model = model_Sf[channel, wave_band].T

    # convert awac SSE spectrum to pressure spectrum.
    Sp_f_model = (Seta_f_model * (rho * g * model_Kp) ** 2) / 1e8  # convert pascals to decibars, (must square for units to match)
    # correct DAS spectrum to convert from phase_shifts to pressure
    Sp_f_das = (das_PSD * C_f[channel, :]) / 1e8  # the calibration coefficient is for a pressure spectra - convert from pascals to dbars

    # integrate under the spectra - should yield essentially the same value if calibration was done correctly
    Prms_das = np.sqrt(np.trapezoid(Sp_f_das, model_freqs))  # compute rms pressure value from the das pressure spectrum
    Prms_model = np.sqrt(np.trapezoid(Sp_f_model, model_freqs))  # compute rms pressure value from the awac pressure spectrum

    # print out metrics
    print(f'Channel: {channel}, {orientation}')
    print(f'PRESSURE RMS VALUES: \n DAS: {Prms_das} \n AWAC: {Prms_model}')
    print("Calibration Flag: Pass" if calibration_flag[channel] else "Calibration Flag: Fail")

    # plot the spectra on the same plot
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))

    line1 = axes.plot(model_freqs, Sp_f_model, color='blue', label='Modeled Pressure PSD (AWAC)')
    line2 = axes.plot(model_freqs, Sp_f_das, color='red', label='DAS Pressure PSD (Calibrated)')
    axes1 = axes.twinx() #secondary y axis
    line3 = axes1.plot(model_freqs, das_PSD, color='green',  linestyle='--', label='DAS Phase Shift PSD')
    axes1.set_ylabel('Phase Shift PSD (Rad²/Hz)', color='green')
    axes1.tick_params(axis='y', color='green', labelcolor='green', which='both')
    axes.set_xlabel('Frequency (Hz)')
    axes.set_ylabel('Pressure PSD (dbar²/Hz)')
    axes.set_title(f'Freq. Dependent Spectral Area Matching; Channel: {channel}, Time: {target_time.strftime("%Y-%m-%d %H:%M:%S")}')

    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]

    axes.legend(lines, labels, loc='best')

    plt.tight_layout()
    plt.savefig(f'{save_path}/channel_{channel}_time_{target_time}_area.png')
    plt.show()
    plt.close('all')


def C_f_v_Time(model_data_dir, save_path, channel):

    calib_data_paths = sorted(glob.glob(os.path.join(model_data_dir, '*.npz')), key=sequence_number)  # list of all calibration coefficient files in the directory

    c_all = []  # empty array for calibration coefficients
    time = []  # empty array for time

    fig = plt.figure(figsize=(15, 8))
    axes = fig.add_subplot(111)

    tqdm.write('Loading Calibration Coefficients...')
    for path in tqdm(calib_data_paths, colour='yellow'):
        time.append(sequence_number(path))
        data = np.load(path, allow_pickle=True)  # load data
        data = data['arr_0'].item()  # go through first header

        c = data['calibration_coefficient_freq_dependent']  # pull only the frequency dependent calibration coefficient
        freqs = data['awac_frequency_bins']  # pull the frequency bins

        c_all.append(c)  # append to list

    c_all = np.stack(c_all)  # shape (n_bursts, n_channels, n_freqs)
    C_avg = np.nanmedian(c_all, axis=0)  # take the temporal median of the frequency dependent calibration coefficient

    wave_band = (freqs >= 0.05) & (freqs <= 0.3)  # define the wave band, same as das but has different bin widths so needs a different variable.

    #plot C as a function of frequency for all time
    time_mpl = mdates.date2num(time)  # convert time to matplotlib format
    norm = plt.Normalize(min(time_mpl), max(time_mpl))
    cmap = plt.get_cmap('viridis')  # choose a colormap
    for c, t in zip(c_all, time_mpl):
        axes.plot(freqs[wave_band], c[channel, :], color=plt.cm.viridis(norm(t)), alpha=0.5)
    axes.plot(freqs[wave_band], C_avg[channel, :].T, color='red', linewidth=2, label='Median Calibration Coefficient')
    axes.set_yscale('log')
    axes.set_ylabel('Calibration Coefficient (-/-)')
    axes.set_xlabel('Frequency (Hz)')
    axes.legend()

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes)
    cbar.set_label('Time')
    cbar.ax.yaxis.set_major_formatter(
        mdates.DateFormatter('%m/%d/%Y')
    )

    plt.savefig(f'{save_path}/channel_{channel}_C_f_v_time.png')
    plt.show()
    plt.close('all')


def Compare_Bulk_Wave_Stats(model_data_dir, das_data_paths, calibCoeff_path, save_path, channel, calibration='freq_independent', wave_band_min=0.05, wave_band_max=0.3, sampling_frequency=2):
    #compare the bulk wave statistics, Hs, Te, Thetap of both the model and the calibrated DAS for all timestamps
    assert calibration in ['freq_independent', 'freq_dependent'], "Calibration must be either 'freq_independent' or 'freq_dependent'"

    model_data_paths = sorted(glob.glob(os.path.join(model_data_dir, '*.npz')), key=sequence_number)
    n_hours = len(model_data_paths) # number of hours of data - one hour per burst (burst is the first 35 minutes of DAS data in each hour)

    timestamps = []
    Hs_das_values = []
    Hs_awac_values = []
    Te_das_values = []
    Te_awac_values = []
    Hs_error = 0.0
    Te_error = 0.0

    #preset plot setup
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    fig.suptitle(f'Bulk Wave Statistics Comparison; Channel: {channel}_{calibration}', fontweight='bold', fontsize=16)
    axes[1].set_xlabel('Time (UTC)')
    axes[0].set_ylabel('Significant Wave Height (m)')
    axes[1].set_ylabel('Energy Weighted Period, Te (s)')

    for hour in tqdm(range(n_hours), colour='yellow'):
        #pull the model and awac spectrum for each burst, allow for different types of calibration time averaging.
        timestamp = sequence_number(model_data_paths[hour])  # get the day of the burst\
        day = timestamp.day

        if calibration == 'freq_dependent':

            if type(calibCoeff_path) == str:
                # open the calibration file to get C(x, f) - temporal avg over entire dataset
                C = np.load(calibCoeff_path)  # shape (n_channels, n_freqs)

            if type(calibCoeff_path) == list:
                # find the matching file to day to get the temporal avg over that day of data
                match = next(  # next works with the iterator (path for path in ...) to return the first value that satisfies the condition put forth
                    path for path in calibCoeff_path
                    if sequence_number(path) == day
                )
                C = np.load(match)

        orientation = get_orientation(channel)  # get the orientation of the selected DAS channel

        # pull Modeled AWAC spectrum
        model_data_path = os.path.join(model_data_dir, f'Calib_{timestamp.strftime("%Y-%m-%d_%H_%M_%S")}.npz')  # define awac data path
        model_data = np.load(model_data_path, allow_pickle=True)  # read the data
        model_data = model_data['arr_0'].item()  # open the dictionary and get past the initial format

        model_Sf_full = model_data['awac_corrected_frequency_spectrum']  # awac frequency spectrum
        model_Sf = model_data['awac_cable_weighted_frequency_spectrum']  # awac frequency spectrum weighted by the incident wave direction - energy projected along the cable

        model_freqs = model_data['awac_frequency_bins']  # awac frequency bins
        wave_band = (model_freqs >= wave_band_min) & (model_freqs <= wave_band_max)  # define the wave band, this will lose some information for bulk statistics but is ok as it is applied to all

        if calibration == 'freq_independent':
            C = model_data['calibration_coefficient'] # pull the frequency independent calibration coefficient from the model data

        h = model_data['water_depth']  # water depth along the cable
        calibration_flag = model_data['calibration_flag']  # flag for whether the calibration worked or failed a check, 1 is good, 0 means fail

        burst_files = get_burst_files(das_data_paths, timestamp, 35)  # obtain all files inside a particular burst at the chosen target time. careful of hard coded burst duration

        nperseg = round(sampling_frequency / np.diff(model_freqs)[0])  # assumes constant spacing of the model frequency bins, computes the nperseg such that PSD bins equal those of the model

        das_freqs, das_PSD_raw = get_DAS_PSD_burst(burst_files, nperseg=nperseg, sampling_frequency=sampling_frequency)  # compute the power spectral density over the burst using welch's method
        # rebin onto the full AWAC grid, then clip both with the same mask
        das_PSD_m = rebin_psd(das_freqs, das_PSD_raw[:, channel], model_freqs)
        das_PSD = das_PSD_m[wave_band]  # clip the power spectral density over the wave band

        # define constants
        rho = 1025  # kg/m^3 density of seawater
        g = 9.81  # m/s^2 gravitational acceleration
        h_local = h[channel]  # water depth at chosen DAS channel

        model_T = 1 / model_freqs  # convert awac frequency bins to period
        model_k = wavenumber(h_local, model_T)  # awac wavenumber at each period / frequency bin
        model_Kp = 1 / np.cosh(model_k * h_local)  # pressure response factor at each awac wavenumber

        model_freqs = model_freqs[wave_band]  # constrain the awac frequencies over the wave band
        model_Kp = model_Kp[wave_band]  # constrain the awac pressure response factor over the wave band

        Seta_f_model = model_Sf[channel, wave_band].T #weighted frequency spectrum
        Seta_f_awac_full = model_Sf_full[channel, wave_band].T #full unweighted spectrum
        # get the directional weighting of the spectrum along the cable, this is the ratio of the weighted frequency spectrum to the full spectrum - exact same for the DAS and AWAC. (n_freqs)
        D_eff = Seta_f_model / Seta_f_awac_full

        # correct DAS spectrum to convert from phase_shifts to pressure
        Sp_f_das = (das_PSD * C[channel])   # the calibration coefficient is for a pressure spectra
        Seta_f_das_weighted = Sp_f_das / ((rho * g * model_Kp) ** 2)  # convert to surface elevation spectrum using the pressure response factor - directionally weighted
        Seta_f_das_full = Seta_f_das_weighted / D_eff  # recover the full spectrum by dividing by the directional weight of the spectrum along the cable

        #compute significant wave height for both the DAS and AWAC, don't need frequency band because it is already masked
        Hs_das, _, _ = compute_bulk_statistics(Seta_f_das_full, None, model_freqs, None, None) #significant wave height derived from das spectrum
        Hs_awac, _, _ = compute_bulk_statistics(Seta_f_awac_full, None, model_freqs, None, None) #significant wave height derived from awac spectrum

        #energy weighted period calculation
        df = np.diff(model_freqs)[0]
        m0_das = np.nansum(Seta_f_das_full) * df #0th moment
        m1_das = np.nansum(Seta_f_das_full * model_freqs) * df
        f_e_das = m1_das / m0_das
        Te_das = 1 / f_e_das

        m0_awac = np.nansum(Seta_f_awac_full) * df #0th moment
        m1_awac = np.nansum(Seta_f_awac_full * model_freqs) * df
        f_e_awac= m1_awac / m0_awac
        Te_awac = 1 / f_e_awac

        timestamps.append(timestamp)
        Hs_das_values.append(Hs_das)
        Hs_awac_values.append(Hs_awac)
        Te_das_values.append(Te_das)
        Te_awac_values.append(Te_awac)

        Hs_error += (Hs_awac - Hs_das) ** 2
        Te_error += (Te_awac - Te_das) ** 2

    Hs_rmse = np.sqrt(Hs_error / len(timestamps))
    Te_rmse = np.sqrt(Te_error / len(timestamps))

    #add information to the plots
    axes[0].plot(timestamps, Hs_das_values, 'o-', color='blue', label='DAS Hs')
    axes[0].plot(timestamps, Hs_awac_values, 'o-', color='orange', label='AWAC Hs')

    axes[1].plot(timestamps, Te_das_values, 'o-', color='blue', label='DAS Te')
    axes[1].plot(timestamps, Te_awac_values, 'o-', color='orange', label='AWAC Te')

    axes[0].legend()
    axes[0].grid(True)
    axes[1].legend()
    axes[1].grid(True)

    axes[0].set_title(f'Significant Wave Height (Hs), RMSE={Hs_rmse:.2f} m')
    axes[1].set_title(f'Energy Weighted Period (Te), RMSE={Te_rmse:.2f} s')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'wave_stats_Channel_{channel}_{calibration}.png'))
    plt.show()
    plt.close('all')








if __name__ == "__main__":

    #processed data
    # processed_data_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)
    # strain, time = load_processedData(processed_data_paths) #pull processed strain and time data

    # datetime_array = time2dateTime(time) #convert unix to datetime for visualizations

    # define starttime and endtime for plotting
    # starttime = datetime(2026, 6, 6, 0, 0, 0, tzinfo=timezone.utc)
    # endtime = datetime(2026, 6, 6, 23, 59, 59, tzinfo=timezone.utc)

    # xfrf 300m awac channel location 208. 600m awac is located at channel 402 --> approximate locations based on xFRF location of channels and AWAC


    '''CREATE WATERFALL PLOT'''
    # # savepath = './figures/20260601'
    # channel_start = 0  #define start and end channels for plotting
    # # channel_end = strain.shape[1]
    # channel_spacing=1.6 #meters
    # # waterFall(strain, datetime_array, channel_spacing, starttime, endtime, channel_start, channel_end, savepath)
    #
    # filepaths = sorted(glob.glob('./data/for_OSU_processed/20260601/*.h5'), key=sequence_number)
    # filepaths = clip_by_time(file_paths, starttime, endtime)
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
    # plot_timeSeries(processed_data_paths, channel=700, save_path='./figures', t_start=starttime, t_end=endtime, envelope=False)

    '''WAVENUMBER AND AMPLITUDE TIMESERIES AT A PARTICULAR CHANNEL'''
    # k_and_amp_timeseries(processed_data_paths, save_path='./figures', channel_start=600, channel_end=620, channel_spacing=1.6)

    '''PSD ACROSS CHANNELS'''
    # nperseg = 256
    # noverlap = nperseg // 2
    # # PSD_across_channels(processed_data_paths, nperseg, noverlap, averaged=False, dB=False)

    '''PSD AT ONE CHANNEL'''
    # channel = 310
    # PSD_One_Channel(processed_data_paths, channel, nperseg, noverlap)

    '''DAS CALIBRATION CHECK - ONE CHANNEL - ONE BURST - FREQUENCY INDEPENDENT CALIBRATION'''
    # target_time = datetime(2026, 6, 2, 22, 0, 0) #target time to check spectra
    # model_data_dir = './data/frf_awac/calibration_coefficients_burst_600_weighted'
    # das_data_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)
    # save_path = './figures/area_matching'
    # Calibrated_DAS_v_Model(das_data_paths, model_data_dir, save_path, target_time, channel=402)

    '''DAS CALIBRATION CHECK - RMS PRESSURE - ALL CHANNELS - ALL BURSTS'''
    # model_data_paths = sorted(glob.glob('./data/frf_awac/calibration_coefficients_burst_600/*'), key=sequence_number)
    # das_data_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)
    # save_path = './figures/area_matching'
    # Calibrated_DAS_v_Model_allBursts(das_data_paths, model_data_paths, save_path)

    '''BULK WAVE PARAM COMPARISON BETWEEN MODEL AND AWAC'''
    # awac_data_dir = './data/frf_awac/'
    # model_data_dir = './data/frf_awac/calibration_coefficients_burst_600'
    # AWAC_modelVobs(awac_data_dir, model_data_dir)

    '''BULK WAVE PARAM COMPARISON BETWEEN 300 AND 600 AWACS'''
    # awac_data_dir = './data/frf_awac/'
    # save_dir = './figures'
    # AWAC300_v_AWAC600(awac_data_dir, save_dir)

    '''FREQUENCY DEPENDENT CALIBRATION COEFFICIENT - DAS V MODEL'''
    # target_time = datetime(2026, 6, 2, 22, 0, 0)  # target time to check spectra        #MAKE SURE THIS IS UPDATED BY RUNNING UTILS.PY. OTHERWISE, IT WILL BE AN OLD FILE CAUSING ERRORS
    # das_data_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)
    # model_data_dir = './data/frf_awac/calibration_coefficients_burst_600_weighted'
    # save_path = './figures/area_matching_freq_dependent'
    # # allAvg_C_f_path = './data/frf_awac/freq_dependent_calib_coeff/avg_calibCoeff_freqDependent.npy'   #entire dataset averaged c(f)
    # dayAvg_C_f_paths = glob.glob('./data/frf_awac/freq_dependent_calib_coeff_weighted/*.npy')  #day averaged c(f)
    # filtered_files = sorted([f for f in dayAvg_C_f_paths if any(char.isdigit() for char in os.path.basename(f))], key=sequence_number)  #remove files without a number in the path
    # Calibrated_DAS_freq_dependent_v_Model(das_data_paths, model_data_dir, filtered_files, save_path, target_time, channel=402)

    '''FREQUENCY DEPENDENT CALIBRATION COEFFICIENT OVER TIME'''
    # # average frequency dependent calibration coefficient over all data bursts
    # save_dir = './figures'
    # model_dir = './data/frf_awac/calibration_coefficients_burst_600'
    # C_f_v_Time(model_dir, save_dir, channel=402)

    '''COMPUTE BULK WAVE STATISTICS AT A CHANNEL'''
    model_data_dir = './data/frf_awac/calibration_coefficients_burst_600_weighted'
    das_data_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)
    dayAvg_C_f_paths = glob.glob('./data/frf_awac/freq_dependent_calib_coeff_weighted/*.npy')  # day averaged c(f)
    filtered_files = sorted([f for f in dayAvg_C_f_paths if any(char.isdigit() for char in os.path.basename(f))], key=sequence_number)  # remove files without a number in the path
    save_path = './figures/wave_information'
    Compare_Bulk_Wave_Stats(model_data_dir, das_data_paths, filtered_files, save_path, channel=402, calibration='freq_dependent')




    #TODO:Finally do the wave direction stuff

    #TODO: f - k plots of the DAS PSD one in the cross shore leg and one on the alongshore leg (need gradient in x to compute the wave number k) - This will distort under oblique waves

    #TODO: look at residual spectrum - compute rmse.  function for compute residual - allow for either one x in t or one t in x.

    #TODO: look at spectrograms of the DAS data and the AWAC data over the entire dataset - plus directional spectra and directional spread

    #TODO: would ast v puv computation of awac/model directional spectrum matter?

    #TODO: make a run frf DAS Plots script at the end of this

    #TODO:check to make sure wave refraction is working - do this while checking with marcelas code - coarseness of the direction binning may affect the refraction - things may be shore normal but be marked in a bin a few degrees off.

    #TODO: change the name of where the model data resides to calibration_data - not calib coefficients

    #TODO: add a script/class that holds all data constants with an @dataclass decorator



