import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
from FRF_DAS_processing import sequence_number, time2dateTime
from FRF_DAS_plots import load_processedData
from scipy.signal import welch
from tqdm import tqdm



'''PSD plot along the channels'''
# def PSD_across_channels(file_paths, nperseg, noverlap, sampling_frequency=2, channel_spacing=1.6, averaged=False, dB=False):
#
#     tqdm.write(f'Plotting {len(file_paths)} files...')
#     psd_to_avg = [] #accumulate psd to average over
#
#     #sample file to get the shape of the strain array
#     sample_strain, sample_time = load_processedData(file_paths[0])
#
#     N_samples, N_channels = sample_strain.shape  # get shape of the strain array
#
#     # set up grid of cross shore channels
#     channel_positions = channel_spacing * np.arange(N_channels)
#
#     for file in tqdm(file_paths, colour='#ff5733'):  # loop through all file paths
#
#         strain, time = load_processedData(file) #pull data
#
#         #strain array of shape (t, x)
#         freqs, psd = welch(strain, fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap, axis=0)  #iterate through time, return PSD of shape (f, x)
#
#         if averaged:
#             psd_to_avg.append(psd)
#
#         if dB:
#             psd = 10 * np.log10(psd + 1e-20)  # convert to dB avoiding log(0) - if you want the averaged plot in dB
#
#         vmin = np.percentile(psd,
#                              5)  # set the colorbar floor to a value below the 5th percentile of the data to avoid outliers dominating the color scale
#         vmax = np.percentile(psd,
#                              99.9)  # set the colorbar ceiling to a value above the 99th percentile of the data to avoid outliers dominating the color scale
#
#         if not averaged:
#             fig, axes = plt.subplots(1, 1, figsize=(12, 8))
#             pcm = axes.pcolormesh(channel_positions, freqs, psd, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax) #plot the average frequency content from the 5 minute file.
#             time_array = time2dateTime(time)
#             save_time = time_array[0]
#
#             cbar = fig.colorbar(pcm, ax=axes)
#             cbar.set_label("Power Spectral Density (Strain**2/Hz)")
#
#             axes.set_xlabel('Along Cable Distance (m)')
#             axes.set_ylabel('Frequency (Hz)')
#             axes.set_title(f'PSD v Channel Distance {save_time.strftime("_%Y-%m-%d_%H:%M:%S")}')
#
#             plt.tight_layout()
#             plt.savefig(f'./figures/freq_dist_plots/freq_dist_{save_time.strftime("_%Y-%m-%d_%H-%M-%S")}.png')
#             # plt.show()
#             plt.close('all')
#
#     if averaged:
#         averaged_psd = np.average(psd_to_avg, axis=0)
#
#         vmin = np.percentile(averaged_psd,
#                              5)  # set the colorbar floor to a value below the 5th percentile of the data to avoid outliers dominating the color scale
#         vmax = np.percentile(averaged_psd,
#                              99.9)  # set the colorbar ceiling to a value above the 99th percentile of the data to avoid outliers dominating the color scale
#
#         fig, axes = plt.subplots(1, 1, figsize=(12, 8))
#         pcm = axes.pcolormesh(channel_positions, freqs, averaged_psd, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
#
#         cbar = fig.colorbar(pcm, ax=axes)
#         cbar.set_label("Power Spectral Density (Strain**2/Hz)")
#
#         axes.set_xlabel('Along Cable Distance (m)')
#         axes.set_ylabel('Frequency (Hz)')
#         axes.set_title(f'PSD v Channel Distance Averaged')
#
#         plt.tight_layout()
#         plt.savefig(f'./figures/freq_dist_plots/Averaged_freq_dist.png')
#         plt.show()
#         plt.close('all')
#
#
# file_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)
# nperseg = 256
# noverlap = nperseg //2
# PSD_across_channels(file_paths, nperseg, noverlap, averaged=True, dB=False)



#TODO: create a band integrated energy map




'''wave number from das data'''
# input_data_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)[6:12]
#
# #each file contains a timeseries of strain
# strain_batch = []
# time_batch = []
# for file in input_data_paths:
#     strain, time = load_processedData(file)
#     strain_batch.append(strain)
#     time_batch.append(time)
#
# strain_batch = np.concatenate(np.array(strain_batch), axis=0) #create one large strain(t, x) array
#
# def get_signal_amplitude(strain, channel:int, sampling_frequency=2):
#
#     #use the strain timeseries and welches method to get the PSD of the strain signal across frequency
#
#     # nperseg = number of samples per segment for Welch's method.
#     # Welch splits the signal into overlapping segments of this length,
#     # computes an FFT-based periodogram for each, and averages them
#     # to get a smoother, lower-variance PSD estimate.
#     #
#     # Tradeoff:
#     #   larger nperseg  -> finer frequency resolution (df = fs/nperseg), but fewer
#     #                      segments to average -> noisier PSD
#     #   smaller nperseg -> coarser frequency resolution, but more segments
#     #                      averaged -> smoother, more stable PSD
#
#     freqs, psd = welch(strain[:, channel], fs=sampling_frequency, nperseg=256) #return psd (strain**2/Hz) and frequency bins (Hz)
#
#     wave_band = (freqs >= 0.05) & (freqs <= 1)  # boolean mask of incident wave frequency band outlined in the DAS paper
#
#     Amplitude = np.sqrt(np.trapezoid(psd[wave_band], freqs[wave_band])) #integrate across the wave band to get the amplitude at the particular channel
#
#     return Amplitude
#
# def get_waveNumber(strain, channel_start=250, channel_end=300, dx = 1.6, sampling_frequency=2, peak=True, debug=False):
#     #use real fft to pull the peak wave number from the signal. Need to preserve phase information so no PSD.
#     # real fft because we only need the positive values of the frequency spectrum (bc we only need frequencies in the wave band) - hermitian symmetry
#     strain = strain[:, channel_start:channel_end] #defaults to the cross-shore section of the cable
#
#     n_samples, n_channels = np.shape(strain) #shape of the strain array
#
#     channel_positions = dx * np.arange(n_channels) #builds the x position (in meters) of the channels
#
#     # exposed_mask = (channel_positions < 90) | (channel_positions > 160) #exposed portion of the channel
#
#     X = np.fft.rfft(strain, axis=0) #compute the real fft of the strain data along the time axis
#     freqs = np.fft.rfftfreq(n_samples, d=1/sampling_frequency) #return the discrete fourier transform sampling frequencies
#
#     wave_band = (freqs >= 0.05) & (freqs <= 1)  # boolean mask of incident wave frequency band outlined in the DAS paper
#
#     band_idx = np.where(wave_band)[0] #actual rows of X that correspond to the wave band
#
#     power_in_band = np.sum(np.abs(X[band_idx, :])**2, axis=1) #for each frequency bin within the wave band, sum up the power |X|**2 across all channels. How much energy at a particular frequency over the entire array
#
#     if debug:
#         #debugging - sanity visualization of the power spectrum
#         plt.plot(freqs[band_idx], power_in_band)
#         plt.show()
#         plt.close('all')
#
#     if peak:
#         freq_index = band_idx[np.argmax(power_in_band)] #find the frequency bin index (in the wave band) with the most energy
#
#         phi = np.angle(X[freq_index, :])  # get the phase at the one dominant frequency across all channels (phi(x))
#
#         phi_unwrapped = np.unwrap(
#             phi)  # because np.angle only returns (-pi, pi] if the phase goes past pi it jumps to -pi. This makes the sequence continuous instead
#
#         k, phi_o = np.polyfit(channel_positions, phi_unwrapped,
#                               1)  # fits a straight line -> phi(x) = kx + phi_o. Through the position (x) and unwrapped phase points (phi(x)) using least squares,
#
#     else: #take a power weighted average of all wave numbers across every frequency bin in the wave band. helps avoid problems generated by a noisy or bimodal spectrum
#         k_per_freq = []  #empty array for k's across the entire wave band
#         power_per_freq = [] #empty array for power across the entire wave band
#         for idx in band_idx:  #for frequency bins inside the wave band...
#             phi_unwrapped = np.unwrap(np.angle(X[idx, :])) # because np.angle only returns (-pi, pi] if the phase goes past pi it jumps to -pi. This makes the sequence continuous instead
#             k, phi_o = np.polyfit(channel_positions, phi_unwrapped, 1) # fits a straight line (deg=1) -> phi(x) = kx + phi_o. Through the position (x) and unwrapped phase points (phi(x)) using least squares,
#             k_per_freq.append(k) #append k to the list
#             power_per_freq.append(np.sum(np.abs(X[idx, :]) ** 2)) #append the power at the frequency bin to the list.
#
#         k_per_freq = np.array(k_per_freq)
#         power_per_freq = np.array(power_per_freq)
#
#         k = np.average(k_per_freq, weights=power_per_freq) #take the power weighted average k
#
#     if debug:
#         # residuals = actual measured phase (phi_unwrapped, from the FFT) minus the
#         # predicted phase from the linear fit (k*x + phi_o), evaluated at each channel
#         # position. Large/systematic residuals indicate the true phase isn't well
#         # described by a single constant k over this span (e.g. shoaling-driven
#         # spatial variation in k). Scatter around zero gives a reasonable constant k assumption
#         residuals = phi_unwrapped - np.polyval([k, phi_o], channel_positions) #pulls the phase (phi(x)) evaluated at each channel position, then subtract the phase from the fourier transform
#         plt.figure(figsize=(8, 4))
#         plt.scatter(channel_positions, residuals, s=15)
#         plt.axhline(0, color='k', linewidth=0.8)
#         plt.xlabel('Along Cable Distance (m)')
#         plt.ylabel('Phase residual (rad)')
#         plt.title('Residuals of linear phase fit vs. position')
#         plt.show()
#
#     # returns only the slope (Local wavenumber k). phi_o would be the y intercept.
#     #  k is estimated from a single least-squares fit across all channels at the
#     #  peak frequency (or power weighted average), rather than from point-to-point phase differences, making it
#     #  more robust to per-channel noise in the phase measurements.
#     return k
#
# amp = get_signal_amplitude(strain_batch, channel=100)
#
# #remember this is either kx or ky depending on the orientation of the DAS cable
# k = get_waveNumber(strain_batch, channel_start=255, channel_end=285, peak=False)
#
#
# #TODO: wave number must be defined by small enough sliding windows in x - by definition the wave number is only valid locally using this math.


'''wave and wind information and plotting'''
# data_path = './data/433p1_rt.nc'
# data = xr.open_dataset(data_path)
#
# #wave information - drop metadata
# WaveHs = data['waveHs'].values
# T = data['waveTp'].values
# WaveTime = data['waveTime'].values
# meanDirection = data['waveMeanDirection'].values
# peakDirection = data['waveDp'].values
#
# #convert to pandas dataframe
# df = pd.DataFrame({
#     'Hs': WaveHs,
#     'Peak Period': T,
#     'Time': WaveTime,
#     'Peak Direction': peakDirection
# })
#
# #clip dataframe to desired date range
# start_time = datetime(2026, 6, 1, 0, 0, 0)
# end_time = datetime(2026, 6, 15, 0, 0, 0)
#
# df_clipped = df[(df['Time'] > start_time) & (df['Time'] < end_time)]
#
# #plot
# fig, axes = plt.subplots(2, 1, figsize=(15, 10))
#
# axes[0].scatter(df_clipped['Time'], df_clipped['Hs'], s=3, color='red')
# axes[0].set_xlabel('Time')
# axes[0].set_ylabel('Wave Height (m)')
# axes[0].set_title('Significant Wave Height Over Time')
# axes[0].grid()
#
# axes[1].scatter(df_clipped['Time'], df_clipped['Peak Direction'], s=3, color='blue')
# axes[1].set_xlabel('Time')
# axes[1].set_ylabel('Direction (degrees CW relative to true north)')
# axes[1].set_title('Peak Wave Direction Over Time')
# axes[1].grid()
# axes[1].set_ylim(0, 180)  #east coast beach
#
# fig.suptitle('CDIP 433 - Duck FRF 17m Buoy: Wave Information 06/01/2026 - 06/14/2026', fontsize=16, fontweight='bold')
#
# plt.tight_layout()
# plt.savefig('./figures/waveInformation.png')
# plt.show()
# plt.close('all')
#
# '''WIND'''
# wind_path = './data/windData.txt'
# wind_data = pd.read_csv(wind_path, sep=r'\s+', skiprows=[1])
# wind_data["Time"] = pd.to_datetime(
#     wind_data[["#YY", "MM", "DD", "hh", "mm"]].rename(
#         columns={
#             "#YY": "year",
#             "MM": "month",
#             "DD": "day",
#             "hh": "hour",
#             "mm": "minute"
#         }
#     )
# )
#
#
# #remove erroneous points
# wind_data = wind_data.replace([99, 999, 99.0, 999.0], np.nan)
# wind_data = wind_data.dropna(subset=["WSPD", "WDIR"])
#
# #clip to desired time
# wind_df = wind_data[(wind_data['Time'] > start_time) & (wind_data['Time'] < end_time)]
#
# #wind data plot
# fig, axes = plt.subplots(2, 1, figsize=(15, 10))
#
# axes[0].scatter(wind_df['Time'], wind_df['WSPD'], s=3, color='orange')
# axes[0].set_xlabel('Time')
# axes[0].set_ylabel('Wind Speed (m/s)')
# axes[0].set_title('Wind Speed Over Time')
# axes[0].grid()
#
# axes[1].scatter(wind_df['Time'], wind_df['WDIR'], s=3, color='green')
# axes[1].set_xlabel('Time')
# axes[1].set_ylabel('Wind Direction (degrees CW relative to True North)')
# axes[1].set_title('Wind Direction Over Time')
# axes[1].grid()
#
# fig.suptitle('NDBC Station DUKN7 - 8651370 - Duck Pier, NC: Wind Information', fontsize=16, fontweight='bold')
#
# plt.tight_layout()
# plt.savefig('./figures/windInformation.png')
# plt.show()
# plt.close('all')





