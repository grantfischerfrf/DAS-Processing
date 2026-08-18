import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import glob
from FRF_DAS_processing import sequence_number, time2dateTime, strain2phase
import re
from FRF_DAS_plots import load_processedData, waterFall
from scipy.signal import welch
from scipy import stats
from scipy.stats import binned_statistic
from Empirical_Calibration import get_DAS_PSD, get_DAS_PSD_burst, parse_das_timestamp
import scipy
from tqdm import tqdm
from scipy.interpolate import interp1d, LinearNDInterpolator

from geoprocess import ncsp2FRF


def get_burst_files(all_paths, start_time, burst_duration_min=35):
    """Pull all files whose timestamp falls within [start_time, start_time + burst_duration) """
    burst_end = start_time + timedelta(minutes=burst_duration_min)
    burst_files = [
        f for f in all_paths
        if start_time <= sequence_number(f) < burst_end
    ]
    return sorted(burst_files, key=sequence_number)

def find_spike(signal, sigma=5):
    # find where a signal shoots up from a spike - sigma is the sliding window size that the variance is calculated over.
    threshold = 0.01 * signal.max()
    t_min = np.argmax(signal < threshold)

    t_values = range(t_min, len(signal) - sigma)
    ratios = []

    for t in t_values:
        left = signal[
               t - sigma:t]  # checks the variance from bins left and right from a point. At the knee point, the ratio in left and right variance spikes
        right = signal[t:t + sigma]

        if len(left) > 1 and len(right) > 1:
            var_left = np.var(left)
            var_right = np.var(right)

            if var_right > 0:
                ratios.append(var_left / var_right)
            else:
                ratios.append(0)  # guard against divide by 0
        else:
            ratios.append(0)

    T = t_values[np.argmax(ratios)]
    return T, ratios

def wavenumber(h, T, g=9.81):

    #Solution for k using Simarro and Orfila (2013)
    # one step correction from Newton's method applied to Beji (2013) initial guess
    #h must be a positive quantity

    #define constants
    a = 1.09
    b0 = 1.55
    b1 = 1.3
    b2 = 0.216
    w = (2*np.pi) / T
    uo = ((w**2) * h)/g

    #solve for ub from Beji (2013)
    ub = (uo / (np.sqrt(np.tanh(uo)))) * (1 + (uo * a) * np.exp(-1 * (b0 + (b1 * uo) + (b2 * uo**2))))

    #solve for us from Simarro and Orfila (2013)
    us = ((ub **2) + uo * (np.cosh(ub))**2) / (ub + np.sinh(ub) * np.cosh(ub)) #kh

    k = us / h

    return k


def phaseSpeed(h, k, g=9.81):

    #compute phase speed for intermediate water depth

    return np.sqrt((g/k) * np.tanh(k * h))

def n(k, h):
    #compute group velocity factor for intermediate water depth

    return 0.5 * (1 + ((2 * k * h) / (np.sinh(2 * k * h))))

def groupSpeed(h, k, g=9.81):

    #compute group speed for intermediate water depth

    return phaseSpeed(h, k, g) * n(k, h)

def shoalingCoefficient(k0, h0, k1, h1):

    Cg0 = groupSpeed(h0, k0)

    Cg1 = groupSpeed(h1, k1)

    return np.sqrt(Cg0 / Cg1)

def refractionCoefficient(theta0, theta1):

    #compute refraction coefficient using snells law - assuming straight and parallel depth contours

    return np.sqrt(np.cos(theta0) / np.cos(theta1))  #ratio of cosines of the deep water and shallow water wave angles

def get_AWAC_data(file_path):

    # 300m cross-shore location
    data_300 = xr.open_dataset(file_path)
    # print(list(data_300.keys())) #list all data variables, each variable should have a time array attached to it

    dirspec = data_300['directionalWaveEnergyDensity']  # directional spectrum
    # PorAST = data_300['PorAST'].values #PUV or SUV mode
    gaugeDepth = data_300['gaugeDepth']  # depth from the water surface to the top of the gauge
    gaugeElevation = data_300['gaugeElevation']  # gauge vertical position in NAVD88
    bottomElevation = data_300['bottomEelevation']  # bottom vertical location from altimiter

    # correct for DAS data using spectral matching
    S_f_theta_eta = dirspec.values  # PSD S_eta(N_samples, f, theta)
    freqs = dirspec.coords['waveFrequency'].values  # frequency bins - 0.04 - 0.4975 Hz
    dirs = dirspec.coords['waveDirectionBins'].values  # direction bins - 0 - 355
    time = dirspec.coords['time'].values  # 719 time steps over the entire month collect - one 34 minute burst per hour - split DAS into 35 minute sections
    dtheta = np.diff(dirs)[0]  # bin width in degrees - 5
    df = np.gradient(freqs)  # frequency bin widths
    T = 1 / freqs  # wave period at each frequency

    return S_f_theta_eta, freqs, dirs, time, dtheta, df, T, gaugeDepth, gaugeElevation, bottomElevation


def get_DAS_PSD_channel(file_path, channel, nperseg, noverlap=None, sampling_frequency=2):

    #read file
    strain, time = load_processedData(file_path)

    #convert strain to phase - original strain to phase equation is not suitable for this calibration
    phase = strain2phase(strain)

    if noverlap is None:
        noverlap = nperseg // 2

    freqs, PSD = welch(phase[:, channel], fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap)

    return freqs, PSD


'''CALIBRATION COEFFICIENT original version'''
# #data is analyzed in 34 minute bursts on the hour.
#
# #PULL AWAC DATA
#
# #300m cross-shore location  #TODO: add in the 600m awac as well. The spectrum generated from the awacs should shoal together to an equivalent spectra in between the two.
# data_path_300 = './data/frf_awac/FRF-ocean_waves_sig940-300_202606.nc'
# S_f_theta_eta, awac_freqs, dirs, p_time, dtheta, df, T, gaugeDepth, gaugeElevation, bottomElevation = get_AWAC_data(data_path_300)
#
#
# #load in the DAS XYZ data
# channel_grid = pd.read_pickle('./data/frf_awac/channel_grid.pkl')
# #find the channel location closest to x=300m xFRF
# x_idx = np.argmin(np.abs(channel_grid['xFRF'] - 300))
# reference_channel = channel_grid['channel'].iloc[x_idx]  #find the channel closest to 300m xFRF AWAC
# reference_depth = channel_grid['z'].iloc[x_idx]  #find the depth at the reference channel
# N_channels = len(channel_grid)  #number of DAS channels
#
#
# #convert PSD of sea surface elevation to PSD of pressure using linear wave theory
# rho = 1025 #kg/m**3 #TODO: use equation of state to find this if you want to be more precise
# g = 9.81 #m/s**2
#
# #LOOP FOR CALIBRATION COEFFICIENTS
# processed_data_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)
# N_files = len(processed_data_paths)  #number of processed DAS files
#
# pressure_spectrum = np.zeros((N_files, N_channels, len(awac_freqs))) #empty array to hold the pressure spectrum at each channel
#
# gamma_break = 0.43 #breaker index for random/Rayleigh-distributed wave field (Thornton & Guza, FRF-calibrated) - can use the Goda parameterization too if needed
#
# m0_list = []#temp
# h1_list = []#temp
#
# valid_channels = 0
#
# print('Calculating Calibration Coefficients ...')
# for t in tqdm(range(N_files), colour='#ff5733'):
#
#     C = np.zeros(N_channels)  # empty array to hold the calibration coefficient c(t, x)
#
#     #find the starttime of the Das file
#     match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})', processed_data_paths[t])  # extract the timestamp from the first file name
#     timestamp = None
#     if match:
#         timestamp = match.group(1)  # pull the timestamp from the regex object
#     else:
#         print(f"No timestamp found in file name: {processed_data_paths[t]}") #break the loop and inspect file names instead
#         break
#
#     das_start_time = np.datetime64(datetime.strptime(timestamp, '%Y_%m_%d_%H_%M_%S'))  # starttime of the file
#     das_end_time = np.datetime64(das_start_time + timedelta(minutes=5))  # endtime of the file
#
#     # closest AWAC burst
#     idx = np.argmin(np.abs(p_time - das_start_time))
#
#     awac_start_time = p_time[idx]
#
#     awac_end_time = awac_start_time + np.timedelta64(34, 'm')  # AWAC burst is 34 minutes long - this line isn't super necessary but here for consistency
#
#     offset = np.abs(das_start_time - awac_start_time)
#     # print(
#     #     "das_file:" file,
#     #     "das_time:" das_start,
#     #     "awac_time:" awac_time,
#     #     "offset:" offset
#     # )
#
#     #clip the AWAC data based on the respective start and end time
#     time_mask = (p_time >= awac_start_time) & (p_time <= awac_end_time)
#     S_f_theta_eta_masked = S_f_theta_eta[time_mask, :, :] #this is in units of m**2/Hz/deg
#     gaugeDepth_masked = gaugeDepth.values[time_mask] #sensor depth using pressure
#     gaugeElevation_value = gaugeElevation.values.item()  #the distance from the NAVD88 reference level to the top of the gauge?
#     bottomElevation_masked = bottomElevation.values[time_mask] #the elevation of the seabed in reference to NAVD88
#     p_time_masked = p_time[time_mask]  #clip time array to be used
#
#     # depth from bottom to sensor
#     depth_from_bottom = gaugeElevation_value - bottomElevation_masked #height of sensor off the bottom of the bed - array
#     h_awac = gaugeDepth_masked + depth_from_bottom #full water depth - array
#
#     S_f_theta_eta_selected = S_f_theta_eta[idx] #select the AWAC burst closest to the DAS file start time - squeeze extra time dimension (1, 62, 72) -> shape (62, 72)
#
#     h0 = h_awac.item() #reference depth at the AWAC at time t - [number] use .item to pull out
#
#     surfaceElevation = gaugeElevation_value + gaugeDepth_masked.item() #free surface elevation in NAVD88
#
#     k0 = wavenumber(h0, T) #wavenumber at the AWAC at time t
#
#     for ch in tqdm(range(N_channels), colour='green', leave=False):
#
#         #skip the first ~100 channels of the DAS cable, there is no signal there  #TODO: May need to do this to keep the calibration factor valid.
#         # if ch < 120 or ch > N_channels - 120:
#         #     continue
#
#         #pull the DAS PSD at the current channel and time step
#         das_freqs, das_PSD = get_DAS_PSD_channel(processed_data_paths[t], ch, nperseg=256, sampling_frequency=2)  #this is in units of m**2/Hz
#         #TODO: probably will need to average the PSD for the 35 minute burst.
#
#         #get the depth at the current channel
#         bed = channel_grid['z'].iloc[ch]  #depth at the current channel
#
#         h1 = surfaceElevation - bed
#
#         h1_list.append(h1)
#
#         if h1 <= 0: #h1 must be positive for dispersion relation (wavenumber calculation) can use 1
#             continue
#
#         valid_channels += 1 #add a count for each channel that is allowed to pass through
#
#         S1_local = np.zeros((len(awac_freqs), len(dirs)))
#
#         k1_arr = np.zeros(len(awac_freqs))  #store k1 per frequency - no recomputation later
#
#         for i, f in enumerate(awac_freqs):
#
#             k0_i = k0[i] #wavenumber at the AWAC at time t and frequency f
#
#             #compute the local wavenumber
#             k1 = wavenumber(h1, T[i])  #wavenumber at a given frequency and depth
#
#             k1_arr[i] = k1
#
#             Ks = shoalingCoefficient(k0_i, h0, k1, h1)  # compute the shoaling coefficient
#
#             for j, theta0 in enumerate(dirs):
#
#                 theta0 = theta0 - 71.8535  #convert measure wave angle to xFRF shore normal convention for snells law
#                 theta0 = (theta0 + 180) % 360 - 180 #wrap to -180, 180
#
#                 if np.abs(theta0) >= 75:  #skip waves that are too oblique to xFRF. Causes the jacobian to converge to a singularity - very large values directional bins that shouldn't have a lot of energy
#                     continue
#
#                 theta0 = np.deg2rad(theta0) #convert to radians
#
#                 if np.cos(theta0) <= 0: #skip non-shoreward propagating waves/bins
#                     continue
#
#                 # AWAC directional spectrum
#                 S0 = S_f_theta_eta_selected[i, j]
#
#                 #apply Snell's law to get the local wave angle at the DAS channel
#                 sin_theta1 = (k0_i / k1) * np.sin(theta0)
#                 if np.abs(sin_theta1) > 1:  # no physical solution
#                     continue
#
#                 theta1 = np.arcsin(sin_theta1)
#
#                 J = np.abs((k1 / k0_i) * (np.cos(theta1) / np.cos(theta0))) #compute the Jacobian of the transformation from (f, theta0) to (f, theta1) - this is the change in the directional spectrum due to refraction
#                 #absolute value is used because the energy flux is always positive, but the Jacobian can be negative for certain angles. The absolute value ensures that we are only considering the magnitude of the change in the directional spectrum.
#                 #this will blow up if theta0 is close to 90 degrees - currently skipping those angles
#
#                 # if J > 20:
#                 #     print(np.rad2deg(theta0), J, S0)
#                 #     J_list.append(J)
#                 #     S0_list.append(S0)
#
#                 S1 = S0 * (Ks ** 2) * J #apply conservation of energy flux to the spectrum - square the coefficients to keep units consistent - energy scales with amplitude squared.
#
#                 S1_local[i, j] = S1  #shape(n_freq, n_dir) - store the local spectrum for this frequency and direction
#
#         # apply depth induced dissipation using the breaker index. #TODO: more sophisticated methods can be used later is needed
#         m0_per_freq = np.sum(S1_local, axis=1) * dtheta  # integrate over direction first -> shape (n_freq,)
#         m0 = np.sum(m0_per_freq * df) #integrate over frequency to get the total variance of the sea surface elevation spectrum
#
#         Hrms = np.sqrt(8 * m0) #compute the root-mean-square wave height from the variance of the sea surface elevation spectrum
#
#         Hmax = gamma_break * h1 #compute the maximum wave height using the breaker index -> gamma = Hb / hb (wave height at breaking / water depth at breaking)
#
#         if Hrms > Hmax:
#
#             scale = (Hmax / Hrms) ** 2  #compute the scaling factor to apply to the spectrum to account for depth-induced dissipation. Squared because energy scales with amplitude squared -> mo_new = s * mo_old. mo ~ Hrms**2
#             S1_local = S1_local * scale  #apply the scaling factor to the spectrum
#
#         #convert surface elevation spectrum to pressure spectrum using linear wave theory
#         for i, f in enumerate(awac_freqs):
#
#             k1 = k1_arr[i]  #get the local wavenumber for the current frequency
#
#             Kp = 1 / np.cosh(k1 * h1)  #compute the pressure response factor, assume we are trying to simulate pressure at the seabed z=-h
#
#             P1_f = S1_local[i, :] * (rho * g * Kp) ** 2  #convert the sea surface elevation spectrum to a pressure spectrum using linear wave theory
#
#             pressure_spectrum[t, ch, i] = (np.sum(P1_f) * dtheta)  #integrate over the direction bins to get a 1D spectrum
#
#         #comput the calibration coefficient at each channel
#         #first integrate both the DAS PSD and AWAC PSD over the desired wave band
#         das_wave_band = (das_freqs >= 0.05) & (das_freqs <= 0.3) #wave band defined over the das PSD frequency bins
#         awac_wave_band = (awac_freqs >= 0.05) & (awac_freqs <=0.3)
#         das_psd_variance = np.trapezoid(das_PSD[das_wave_band], das_freqs[das_wave_band]) #integrated power spectral density of the DAS
#         awac_psd_variance = np.trapezoid(pressure_spectrum[t, ch, awac_wave_band], awac_freqs[awac_wave_band]) #integrated power spectral density of the AWAC
#
#         if das_psd_variance <= 0:  #if the DAS PSD is zero or negative, huge or erroneous calibration coefficients will be computed.
#             C[ch] = np.nan
#             continue
#
#         calibration_coefficient = awac_psd_variance / das_psd_variance #compute the calibration coefficient as the ratio of the integrated AWAC PSD over the integrated DAS PSD
#         C[ch] = calibration_coefficient #store the coefficient for this time and channel
#
#     # save the calibration coefficients
#     np.save(f'./data/frf_awac/calibration_coefficients/CalibCoeff_{das_start_time.tolist().strftime("_%Y-%m-%d_%H:%M:%S")}', C)
#     tqdm.write(f'Saved calibration coefficient for {das_start_time.tolist().strftime("%Y-%m-%d %H:%M:%S")}')
#
#         #SANITY CHECK
#         # # derive Hrms from corrected S1_local
#         # m0_local_freq = np.sum(S1_local * dtheta,
#         #                        axis=1)  # integrate over direction to get the variance of the sea surface elevation spectrum at this frequency
#         # m0_local = np.sum(m0_local_freq * df)
#         # Hrms_local = np.sqrt(
#         #     8 * m0_local)  # compute the root-mean-square wave height from the variance of the sea surface elevation spectrum
#         # m0_list.append(Hrms_local)

# #SANITY CHECK
# # # plt.plot(channel_grid['channel'], np.median(pressure_spectrum.squeeze(), axis=1))
# # # plt.plot(np.arange(len(h1_list)), h1_list)
# # plt.plot(np.arange(len(m0_list)), m0_list)
# # plt.xlabel('channel number')
# # plt.ylabel('Hrms (m)')
# # plt.title('Hrms derived from corrected Directional Spectra at each DAS channel')
# # plt.show()
# # plt.close('all')


'''CALIBRATION COEFFICIENT FASTER VERSION - NO MULTIPROCESSING'''
# '''CALIBRATION COEFFICIENT'''
# #data is analyzed in 34 minute bursts on the hour.
#
# #PULL AWAC DATA
#
# #300m cross-shore location  #TODO: add in the 600m awac as well. Have it split at a certain channel the two - switch from calibration with 300 to 600. The attenuated spectra from both instruments should be equal at this range.
# data_path_300 = './data/frf_awac/FRF-ocean_waves_sig940-300_202606.nc'
# S_f_theta_eta, awac_freqs, dirs, p_time, dtheta, df, T, gaugeDepth, gaugeElevation, bottomElevation = get_AWAC_data(data_path_300)
#
# #load in the DAS XYZ data
# channel_grid = pd.read_pickle('./data/frf_awac/channel_grid.pkl')
# #find the channel location closest to x=300m xFRF
# x_idx = np.argmin(np.abs(channel_grid['xFRF'] - 300))
# reference_channel = channel_grid['channel'].iloc[x_idx]  #find the channel closest to 300m xFRF AWAC
# reference_depth = channel_grid['z'].iloc[x_idx]  #find the depth at the reference channel
# N_channels = len(channel_grid)  #number of DAS channels
#
# #convert PSD of sea surface elevation to PSD of pressure using linear wave theory
# rho = 1025 #kg/m**3 #TODO: use equation of state to find this if you want to be more precise
# g = 9.81 #m/s**2
#
# #LOOP FOR CALIBRATION COEFFICIENTS
# processed_data_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)
# N_files = len(processed_data_paths)  #number of processed DAS files
#
# pressure_spectrum = np.zeros((N_files, N_channels, len(awac_freqs))) #empty array to hold the pressure spectrum at each channel
#
# #precompute trig functions for wave refraction
# theta0_deg = dirs - 71.8535   #convert measure wave angle to xFRF shore normal convention for snells law
# theta0_deg = (theta0_deg + 180) % 360 - 180 #wrap to -180, 180
#
# valid_theta = (  #skip waves that are too oblique to xFRF. Causes the jacobian to converge to a singularity - very large values directional bins that shouldn't have a lot of energy
#     (np.abs(theta0_deg) < 75) &
#     (np.cos(np.deg2rad(theta0_deg)) > 0)  #skip non-shoreward propagating waves/bins
# )
#
# theta0_rad = np.deg2rad(theta0_deg) #convert to radians
# cos_theta0 = np.cos(theta0_rad)  #precomputation of cos(theta0)
# sin_theta0 = np.sin(theta0_rad)  #precomputation of sin(theta0)
#
# gamma_break = 0.43 #breaker index for random/Rayleigh-distributed wave field (Thornton & Guza, FRF-calibrated) - can use the Goda parameterization too if needed
#
# m0_list = []#temp
# h1_list = []#temp
#
# valid_channels = 0
#
# print('Calculating Calibration Coefficients ...')
# for t in tqdm(range(N_files), colour='#ff5733'):
#
#     C = np.zeros(N_channels)  # empty array to hold the calibration coefficient c(t, x)
#
#     #find the starttime of the Das file
#     match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})', processed_data_paths[t])  # extract the timestamp from the first file name
#     timestamp = None
#     if match:
#         timestamp = match.group(1)  # pull the timestamp from the regex object
#     else:
#         print(f"No timestamp found in file name: {processed_data_paths[t]}") #break the loop and inspect file names instead
#         break
#
#     das_start_time = np.datetime64(datetime.strptime(timestamp, '%Y_%m_%d_%H_%M_%S'))  # starttime of the file
#     das_end_time = np.datetime64(das_start_time + timedelta(minutes=5))  # endtime of the file
#
#     #pull the DAS PSD for the appropriate file
#     das_freqs, das_PSD_all_channels = get_DAS_PSD(processed_data_paths[t], nperseg=256, sampling_frequency=2)  # this is in units of m**2/Hz
#
#     # closest AWAC burst
#     idx = np.argmin(np.abs(p_time - das_start_time))
#
#     awac_start_time = p_time[idx]
#
#     awac_end_time = awac_start_time + np.timedelta64(34, 'm')  # AWAC burst is 34 minutes long - this line isn't super necessary but here for consistency
#
#     offset = np.abs(das_start_time - awac_start_time)
#     # print(
#     #     "das_file:" file,
#     #     "das_time:" das_start,
#     #     "awac_time:" awac_time,
#     #     "offset:" offset
#     # )
#
#     #clip the AWAC data based on the respective start and end time
#     time_mask = (p_time >= awac_start_time) & (p_time <= awac_end_time)
#     S_f_theta_eta_masked = S_f_theta_eta[time_mask, :, :] #this is in units of m**2/Hz/deg
#     gaugeDepth_masked = gaugeDepth.values[time_mask] #sensor depth using pressure
#     gaugeElevation_value = gaugeElevation.values.item()  #the distance from the NAVD88 reference level to the top of the gauge?
#     bottomElevation_masked = bottomElevation.values[time_mask] #the elevation of the seabed in reference to NAVD88
#     p_time_masked = p_time[time_mask]  #clip time array to be used
#
#     # depth from bottom to sensor
#     depth_from_bottom = gaugeElevation_value - bottomElevation_masked #height of sensor off the bottom of the bed - array
#     h_awac = gaugeDepth_masked + depth_from_bottom #full water depth - array
#
#     S_f_theta_eta_selected = S_f_theta_eta[idx] #select the AWAC burst closest to the DAS file start time - squeeze extra time dimension (1, 62, 72) -> shape (62, 72)
#
#     h0 = h_awac.item() #reference depth at the AWAC at time t - [number] use .item to pull out
#
#     surfaceElevation = gaugeElevation_value + gaugeDepth_masked.item() #free surface elevation in NAVD88
#
#     k0 = wavenumber(h0, T) #wavenumber at the AWAC at time t
#
#     #get the depth across all channels
#     bed = channel_grid['z'].values  #depth at all channels
#
#     for ch in tqdm(range(N_channels), colour='green', leave=False):
#
#         #skip the first ~100 channels of the DAS cable, there is no signal there  #TODO: May need to do this to keep the calibration factor valid.
#         # if ch < 120 or ch > N_channels - 120:
#         #     continue
#
#         #pull the DAS PSD at the current channel and time step
#         das_PSD = das_PSD_all_channels[:, ch]
#         #TODO: probably will need to average the PSD for the 35 minute burst.
#
#         h1 = surfaceElevation - bed[ch]
#
#         h1_list.append(h1)
#
#         if h1 <= 0: #h1 must be positive for dispersion relation (wavenumber calculation) can use 1
#             continue
#
#         valid_channels += 1 #add a count for each channel that is allowed to pass through
#
#         S1_local = np.zeros((len(awac_freqs), len(dirs)))
#
#         k1_arr = np.zeros(len(awac_freqs))  #store k1 per frequency - no recomputation later
#
#         for i, f in enumerate(awac_freqs):
#
#             k0_i = k0[i] #wavenumber at the AWAC at time t and frequency f
#
#             #compute the local wavenumber
#             k1 = wavenumber(h1, T[i])  #wavenumber at a given frequency and depth
#
#             k1_arr[i] = k1
#
#             Ks = shoalingCoefficient(k0_i, h0, k1, h1)  # compute the shoaling coefficient
#
#             for j in range(len(dirs)):
#
#                 if not valid_theta[j]:
#                     continue
#
#                 theta0 = theta0_rad[j]
#
#                 #apply Snell's law to get the local wave angle at the DAS channel
#                 sin_theta1 = (k0_i / k1) * sin_theta0[j]
#
#                 # AWAC directional spectrum
#                 S0 = S_f_theta_eta_selected[i, j]
#
#                 # sin_theta1 = (k0_i / k1) * np.sin(theta0)
#                 if np.abs(sin_theta1) > 1:  # no physical solution
#                     continue
#
#                 theta1 = np.arcsin(sin_theta1)
#
#                 J = np.abs((k1 / k0_i) * (np.cos(theta1) / cos_theta0[j])) #compute the Jacobian of the transformation from (f, theta0) to (f, theta1) - this is the change in the directional spectrum due to refraction
#                 #absolute value is used because the energy flux is always positive, but the Jacobian can be negative for certain angles. The absolute value ensures that we are only considering the magnitude of the change in the directional spectrum.
#                 #this will blow up if theta0 is close to 90 degrees - currently skipping those angles
#
#                 # if J > 20:
#                 #     print(np.rad2deg(theta0), J, S0)
#                 #     J_list.append(J)
#                 #     S0_list.append(S0)
#
#                 S1 = S0 * (Ks ** 2) * J #apply conservation of energy flux to the spectrum - square the coefficients to keep units consistent - energy scales with amplitude squared.
#
#                 S1_local[i, j] = S1  #shape(n_freq, n_dir) - store the local spectrum for this frequency and direction
#
#         # apply depth induced dissipation using the breaker index. #TODO: more sophisticated methods can be used later is needed
#         m0_per_freq = np.sum(S1_local, axis=1) * dtheta  # integrate over direction first -> shape (n_freq,)
#         m0 = np.sum(m0_per_freq * df) #integrate over frequency to get the total variance of the sea surface elevation spectrum
#
#         Hrms = np.sqrt(8 * m0) #compute the root-mean-square wave height from the variance of the sea surface elevation spectrum
#
#         Hmax = gamma_break * h1 #compute the maximum wave height using the breaker index -> gamma = Hb / hb (wave height at breaking / water depth at breaking)
#
#         if Hrms > Hmax:
#
#             scale = (Hmax / Hrms) ** 2  #compute the scaling factor to apply to the spectrum to account for depth-induced dissipation. Squared because energy scales with amplitude squared -> mo_new = s * mo_old. mo ~ Hrms**2
#             S1_local = S1_local * scale  #apply the scaling factor to the spectrum
#
#         #convert surface elevation spectrum to pressure spectrum using linear wave theory
#         for i, f in enumerate(awac_freqs):
#
#             k1 = k1_arr[i]  #get the local wavenumber for the current frequency
#
#             Kp = 1 / np.cosh(k1 * h1)  #compute the pressure response factor, assume we are trying to simulate pressure at the seabed z=-h
#
#             P1_f = S1_local[i, :] * (rho * g * Kp) ** 2  #convert the sea surface elevation spectrum to a pressure spectrum using linear wave theory
#
#             pressure_spectrum[t, ch, i] = (np.sum(P1_f) * dtheta)  #integrate over the direction bins to get a 1D spectrum
#
#         #comput the calibration coefficient at each channel
#         #first integrate both the DAS PSD and AWAC PSD over the desired wave band
#         das_wave_band = (das_freqs >= 0.05) & (das_freqs <= 0.3) #wave band defined over the das PSD frequency bins
#         awac_wave_band = (awac_freqs >= 0.05) & (awac_freqs <=0.3)
#         das_psd_variance = np.trapezoid(das_PSD[das_wave_band], das_freqs[das_wave_band]) #integrated power spectral density of the DAS
#         awac_psd_variance = np.trapezoid(pressure_spectrum[t, ch, awac_wave_band], awac_freqs[awac_wave_band]) #integrated power spectral density of the AWAC
#
#         if das_psd_variance <= 0:  #if the DAS PSD is zero or negative, huge or erroneous calibration coefficients will be computed.
#             C[ch] = np.nan
#             continue
#
#         calibration_coefficient = awac_psd_variance / das_psd_variance #compute the calibration coefficient as the ratio of the integrated AWAC PSD over the integrated DAS PSD
#         C[ch] = calibration_coefficient #store the coefficient for this time and channel
#
#     # save the calibration coefficients
#     np.save(f'./data/frf_awac/calibration_coefficients/CalibCoeff_{das_start_time.tolist().strftime("_%Y-%m-%d_%H_%M_%S")}', C)
#     tqdm.write(f'Saved calibration coefficient for {das_start_time.tolist().strftime("%Y-%m-%d %H:%M:%S")}')

        #SANITY CHECK
        # # derive Hrms from corrected S1_local
        # m0_local_freq = np.sum(S1_local * dtheta,
        #                        axis=1)  # integrate over direction to get the variance of the sea surface elevation spectrum at this frequency
        # m0_local = np.sum(m0_local_freq * df)
        # Hrms_local = np.sqrt(
        #     8 * m0_local)  # compute the root-mean-square wave height from the variance of the sea surface elevation spectrum
        # m0_list.append(Hrms_local)





#SANITY CHECK
# # plt.plot(channel_grid['channel'], np.median(pressure_spectrum.squeeze(), axis=1))
# # plt.plot(np.arange(len(h1_list)), h1_list)
# plt.plot(np.arange(len(m0_list)), m0_list)
# plt.xlabel('channel number')
# plt.ylabel('Hrms (m)')
# plt.title('Hrms derived from corrected Directional Spectra at each DAS channel')
# plt.show()
# plt.close('all')


'''BUILD DAS XYZ GRID'''
#find the DAS channel closest to the awac and get the PSD to a 34 (35 for DAS) minute burst (average)
# channel = 188 #around 300m out on the cross-shore
# channel = 375 #aroudn 600m out on the cross-shore

# file_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)[0:7]
# psds = []
# for file in tqdm(file_paths, colour='#ff5733'):
#
#     strain, time = load_processedData(file)
#
#     N_samples, N_channels = strain.shape  #(600, 1592) #1592 channels each at 1.6m channel spacing.
#
#     #compute averaged PSD at a certain channel
#     channel = 188
#
#     # obtain the power spectral density (and sample frequencies) of strain array
#     freqs, psd = welch(strain[:, channel], fs=2, nperseg=256, noverlap=256//2)
#
#     psds.append(psd)

#average psd
# avg_psd = np.mean(psds, axis=0)

# #Number of channels
# N_channels = 1592
#
# #length of grid in meters
# grid_len = (N_channels * 1.6) // 2  #divide by two because the cable wraps back around on itself.
#
# #find where the cable lays in x FRF to match it with a depth from the larc survey.
# larc_data_path = './data/frf_awac/FRF_geomorphology_elevationTransects_survey_20260611.nc'
# larc_data = xr.open_dataset(larc_data_path)
# larc_x = larc_data.xFRF.values
# larc_y = larc_data.yFRF.values
# profile_number = larc_data.profileNumber.values
# time = larc_data.elevation.coords['time'].values
# elevation = larc_data.elevation.values  #depth profile
#
# #plot profile
# # plt.scatter(larc_x, larc_y, c=elevation)
# # plt.show()
# # plt.close('all')
#
# #pull just the bathymetry data along the yFRF=1000m transect
# cross_shore_profile_mask = (profile_number==1006)
# masked_x = larc_x[cross_shore_profile_mask]
# masked_y = larc_y[cross_shore_profile_mask]
# masked_elevation = elevation[cross_shore_profile_mask]
#
# #pull just the bathymetry data along the xFRF = 778
# profiles = np.unique(profile_number)
#
# xs = []
# ys = []
# zs = []
#
# for p in profiles:
#     mask = profile_number == p
#
#     idx = np.argmin(np.abs(larc_x[mask] - 778))
#
#     xs.append(larc_x[mask][idx])
#     ys.append(larc_y[mask][idx])
#     zs.append(elevation[mask][idx])
#
# xs = np.array(xs)  #alongshore profile
# ys = np.array(ys)
# zs = np.array(zs)
#
# #DAS survey locations
# offshore_survey_path = './DAS_deploy/Alex_FiberLay_20250730.csv'
#
# offshore_survey = pd.read_csv(offshore_survey_path)
# offshore_survey_x = offshore_survey['X']
# offshore_survey_y = offshore_survey['Y']
#
# offshore_frf_x = []
# offshore_frf_y = []
# for x, y in zip(offshore_survey_x, offshore_survey_y):
#     dict = ncsp2FRF(x, y)
#     offshore_frf_x.append(dict['xFRF'])
#     offshore_frf_y.append(dict['yFRF'])
#
# offshore_frf_x = np.array(offshore_frf_x)
# offshore_frf_y = np.array(offshore_frf_y)
#
# dx = np.diff(offshore_frf_x)
# dy = np.diff(offshore_frf_y)
#
# distance = np.sqrt(dx**2 + dy**2)
#
# s_survey = np.concatenate(([0], np.cumsum(distance))) #total distance along the cable in meters from the survey
#
# #build grid to populate - in meters
# cable_grid = np.arange(0, grid_len, 1.6)  #1.6m channel spacing
# # cable_grid = np.arange(0, s_survey.max(), 1.6)
#
# # interpolate outbound leg x,y from the surveyed fiber path - given a distance along the coordinate whats the x or y coordinate
# interp_x = interp1d(s_survey, offshore_frf_x, bounds_error=False)
# interp_y = interp1d(s_survey, offshore_frf_y, bounds_error=False)
#
# x_out = interp_x(cable_grid)
# y_out = interp_y(cable_grid)
#
# # mirror for the return leg to fill out N_channels
# n_return = N_channels - len(cable_grid)
# x_ret = x_out[::-1][:n_return]
# y_ret = y_out[::-1][:n_return]
#
# channel_x = np.concatenate([x_out, x_ret])
# channel_y = np.concatenate([y_out, y_ret])
# channel_number = np.arange(N_channels)
#
# # split channels by whether they're on the cross-shore or alongshore leg
# cross_shore_mask = channel_x <= 777 #xFRF location where the cable should turn
# alongshore_mask = channel_x > 777
#
# channel_z = np.full(N_channels, np.nan)
#
# # cross-shore leg: z as a function of xFRF
# sort_idx_x = np.argsort(masked_x)
# interp_z_crossshore = interp1d(masked_x[sort_idx_x], masked_elevation[sort_idx_x],
#                                 bounds_error=False)
# channel_z[cross_shore_mask] = interp_z_crossshore(channel_x[cross_shore_mask])  #given an xFRF value what is the seafloor elevation here - cross-shore only
#
# # alongshore leg: z as a function of yFRF
# sort_idx_y = np.argsort(ys)
# interp_z_alongshore = interp1d(ys[sort_idx_y], zs[sort_idx_y],
#                                 bounds_error=False)
# channel_z[alongshore_mask] = interp_z_alongshore(channel_y[alongshore_mask])  #given a yFRF value what is the seafloor elevation here - alongshore only
#
# channel_grid = pd.DataFrame({
#     'channel': channel_number,
#     'xFRF': channel_x,
#     'yFRF': channel_y,
#     'z': channel_z,
# })
#
# # save the output grid.
# # channel_grid.to_pickle('./data/frf_awac/channel_grid.pkl')  #TODO: make this a function in a new utilities script
#
# #sanity plot
# plt.scatter(offshore_frf_x, offshore_frf_y, c='orange', s=6)
# plt.scatter(channel_grid['xFRF'], channel_grid['yFRF'], c=channel_grid['z'])
# plt.show()
# plt.close('all')



'''AWAC PSD_eta to PSD_pd'''

#data is analyzed in 34 minute bursts on the hour.

#300m cross-shore location
# data_path_300 = './data/frf_awac/FRF-ocean_waves_sig940-300_202606.nc'
# data_300 = xr.open_dataset(data_path_300)
# # print(list(data_300.keys())) #list all data variables, each variable should have a time array attached to it
#
# dirspec = data_300['directionalWaveEnergyDensity'] #directional spectrum
# pwrspec = data_300['waveEnergyDensity'] #power spectral density
# PorAST = data_300['PorAST'].values #PUV or SUV mode
# gaugeDepth = data_300['gaugeDepth'] #depth from the water surface to the top of the gauge
# gaugeElevation = data_300['gaugeElevation'] #gauge vertical position in NAVD88
# bottomElevation = data_300['bottomEelevation'] #bottom vertical location from altimiter
# nominalDepth = data_300['nominalDepth'] #deployment depth
#
# #correct for DAS data using spectral matching
# S_eta = dirspec.values #PSD S_eta(N_samples, f, theta)
# freqs = dirspec.coords['waveFrequency'].values  #frequency bins - 0.04 - 0.4975 Hz
# dirs = dirspec.coords['waveDirectionBins'].values #direction bins - 0 - 355
# time = dirspec.coords['time'].values #719 time steps over the entire month collect
# dtheta = np.diff(dirs)[0]  # bin width in degrees - 5
# T = 1/freqs  #wave period at each frequency
#
# #clip data over timestamps
# starttime = np.datetime64(datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)) #AWAC is in UTC
# endtime = np.datetime64(datetime(2026, 6, 14, 0, 0, 0, tzinfo=timezone.utc))
#
# #time mask
# time_mask = (time > starttime) & (time < endtime)
#
# #clip arrays to be used
# S_f_theta_eta = S_eta[time_mask, :, :]
# gaugeDepth = gaugeDepth.values[time_mask]
# gaugeElevation = gaugeElevation.values  #the distance from the NAVD88 reference level to the top of the gauge?
# bottomElevation = bottomElevation.values[time_mask] #the elevation of the seabed in reference to NAVD88
#
# #depth from bottom to sensor
# depth_from_bottom = gaugeElevation - bottomElevation #height of sensor off the bottom of the bed
#
# #convert PSD of sea surface elevation to PSD of pressure using linear wave theory
# rho = 1025 #kg/m**3 #TODO: use equation of state to find this if you want to be more precise
# g = 9.81 #m/s**2
#
# h = gaugeDepth + depth_from_bottom #full water depth
# z = -gaugeDepth #how far below the water surface the sensor is
#
# #compute the wave number for each time and frequency bin
# h_col = h[:, None] #(ntime, 1) #reshape the array to broadcast against time
# z_col = z[:, None] # (ntime, 1)
# k = wavenumber(h_col, T[None, :], g) #k is of shape (ntime, nfrequency)
#
# Kp = (np.cosh(k * (h_col + z_col))) / (np.cosh(k * h_col)) #Pressure response factor - kp
#
# #integrate the directional spectrum over the direction bins for broadcasting
# S_f_eta = np.sum(S_f_theta_eta, axis=-1) * dtheta
#
# S_f_p = (rho * g)**2 * Kp**2 * S_f_eta #power spectral density of the wave induced dynamic pressure fluctuations
# # has shape (N_time (samples), frequency bins)


'''LARC SURVEY DATA'''
# larc_data_path = './data/frf_awac/FRF_geomorphology_elevationTransects_survey_20260611.nc'
# larc_data = xr.open_dataset(larc_data_path)
# larc_x = larc_data.xFRF.values
# larc_y = larc_data.yFRF.values
# profile_number = larc_data.profileNumber.values
# time = larc_data.elevation.coords['time'].values
# elevation = larc_data.elevation.values  #depth profile
#
# #plot profile
# # plt.scatter(larc_x, larc_y, c=elevation)
# # plt.show()
# # plt.close('all')
#
# #pull just the bathymetry data along the yFRF=1000m transect
# profile_mask = (profile_number==1006)
# masked_x = larc_x[profile_mask]
# masked_y = larc_y[profile_mask]
# masked_elevation = elevation[profile_mask]
#
# #pull just the bathymetry data along the xFRF = 778
# profiles = np.unique(profile_number)
#
# xs = []
# ys = []
# zs = []
#
# for p in profiles:
#     mask = profile_number == p
#
#     idx = np.argmin(np.abs(larc_x[mask] - 778))
#
#     xs.append(larc_x[mask][idx])
#     ys.append(larc_y[mask][idx])
#     zs.append(elevation[mask][idx])
#
# xs = np.array(xs)  #alongshore profile
# ys = np.array(ys)
# zs = np.array(zs)

# order = np.argsort(ys)

# plt.figure(figsize=(6,8))
# plt.scatter(
#     xs,
#     ys,
#     c=zs,
#     cmap='viridis',
#     s=40,
# )
#
# plt.xlabel('xFRF (m)')
# plt.ylabel('yFRF (m)')
# plt.title('Nearest survey point to x = 778 for each profile')
# plt.axis('equal')
# plt.show()



'''Dune Lidar profile'''
# #THIS ONLY GETS THE DUNE, NOT THE NEARSHORE
# dune_profile_path = './data/frf_awac/FRF-geomorphology_elevationTransects_duneLidarTransect_202606.nc'
# lidar_data = xr.open_dataset(dune_profile_path)
# lidar_x = lidar_data.xFRF.values #
# lidar_y = lidar_data.yFRF.values
# elevation = lidar_data.elevation.values  #depth profile
# time = lidar_data.elevation.coords['time'].values
#
# fix, axes = plt.subplots(1, 1, figsize=(12, 8))
#
# axes.scatter(lidar_x, elevation[100])
# # axes.invert_yaxis()
# axes.set_xlabel('xFRF')
# axes.set_ylabel('Elevation')
#
# plt.tight_layout()
# plt.show()
# plt.close('all')



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
# axes[1].axhline(71.8538, color='black', linestyle='--', label='xFRF shore normal')
# axes[1].set_xlabel('Time')
# axes[1].set_ylabel('Direction (degrees CW relative to true north)')
# axes[1].set_title('Peak Wave Direction Over Time')
# axes[1].grid()
# # axes[1].set_ylim(71.8538 - 90, 71.8535 + 90)  #east coast beach
# axes[1].legend(loc='upper left')
#
# fig.suptitle('CDIP 433 - Duck FRF 17m Buoy: Wave Information 06/01/2026 - 06/14/2026', fontsize=16, fontweight='bold')
#
# plt.tight_layout()
# plt.savefig('./figures/waveInformation.png')
# plt.show()
# plt.close('all')

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


'''FLORENCE AWAC MAT FILE'''
# florence_awac_path = './data/frf_awac/S104015A001_2A17WEcho_1.mat'
# data = scipy.io.loadmat(florence_awac_path)
# config = data['Config']
# awac_data = data['Data']
# metadata = data['Descriptions']
# units = data['Units']
#
# avg_pressure = awac_data['Average_Pressure']
# burst_pressure = awac_data['Burst_Pressure']

#This data contains burst and averaged data. timestamps are in matlab datenum format and will need to be converted
#Sampling Rates:
# Frequency1: 500kHz
#Instrument: Nortek Signature 500

#TODO: what is the burst time period and what is the time window that the average is going over


'''DAS SIGNAL WRT WAVE DIRECTION ACROSS CHANNELS'''
# #open calibrated file
# # data_path = './data/frf_awac/calibration_coefficients/Calib_2026-06-01_00_05_01.npz'  #gets the data from one 5 minute DAS file
# data_path = './data/frf_awac/calibration_coefficients/Calib_2026-06-06_11_00_01.npz'
# data_struc = np.load(data_path, allow_pickle=True)
# data = data_struc['arr_0'].item()
#
# dirspec = data['awac_corrected_directional_spectrum']
# bad_channels = np.all(np.isnan(dirspec), axis=(1, 2)) #find which channels are bad - nans
# dirspec_clean = np.nan_to_num(dirspec, nan=0.0) #of shape (N_channels, N_freqs, N_dirs)
# freqs = data['awac_frequency_bins']
# dirs = data['awac_directional_bins']
# h = data['water_depth'] #water depth should be corrected for tides.
# channels = data['channel_number']
#
# df = np.gradient(freqs)
# dtheta = np.gradient(dirs)
#
# #get timestamp to save file
# match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})',
#                           data_path)  # extract the timestamp from the first file name
# timestamp = match.group(1)  # pull the timestamp from the regex object
#
# #pull out the representative wave for each channel
# #significant wave height - Hmo
# awac_wave_band = (freqs >= 0.05) & (freqs <= 0.3)
# m0 = np.trapezoid(np.trapezoid(dirspec_clean[:, awac_wave_band, :], dirs, axis=-1), freqs[awac_wave_band], axis=-1)
# Hs = 4.004 * np.sqrt(m0)
# Hs[bad_channels] = np.nan
#
# #frequency spectrum
# S_f = np.trapezoid(dirspec_clean, dirs, axis=-1)
# #direction spectrum
# S_theta = np.trapezoid(dirspec_clean, freqs, axis=1)
#
# #peak frequency   #TODO: careful if the spectrum is not narrow banded here.
# fp_idx = np.argmax(S_f, axis=1) #indices of the peak frequency bin
# fp = freqs[fp_idx] #peak frequency at each channel
# fp[bad_channels] = np.nan #restore invalid channels
#
# #peak direction
# theta_p_idx = np.argmax(S_theta, axis=1) #indices of the peak direction bin  #TODO: check to make sure that refraction is working on non shore normal days
# theta_p = dirs[theta_p_idx] #peak direction at each channel
# theta_p[bad_channels] = np.nan #restore invalid channels
#
# #pull the DAS channel grid to get the depth and calculate local wavenumber at each channel
# Tp = 1/fp #peak period
# k = wavenumber(h, Tp) #local wave number
#
# Kp = 1 / (np.cosh(k*h)) #pressure response factor
#
# #pressure gradient amplitude in x and y build from bulk statistics
# rho = 1025 #kg/m**3
# g = 9.81 #m/s**2
#
# # rotate to shore-normal frame and flip to propagation direction
# theta_p_local = theta_p - 71.8535 #convert to xFRF shore-normal orientation
# theta_p_prop = theta_p_local + 180  #converts waves coming from direction to waves propagating towards the shoreline
# theta_p_rad = np.deg2rad(theta_p_prop) #convert degrees to radians
#
# #pressure gradients - remember x increases offshore.
# dpdx = (0.5 * rho * g * Hs * k * np.cos(theta_p_rad) * Kp) #/ 10000 #units of decibars/m - fluid pressure is increasing/decreasing for every space along the x grid.
# dpdy = -(0.5 * rho * g * Hs * k * np.sin(theta_p_rad) * Kp) #/ 10000 #units of decibars/m - negative because positive xFRF is to the North (to the left of the shore normal transect)
# #a wave coming from the south (dirs=180 - 71.8535) gives a positive dpdy and a wave from the north (dirs=0 - 71.8535) gives a negative dpdy
#
# #pressure amplitude
# p_amp = 0.5 * rho * g * Hs * Kp
#
# #wavelengths: cross and alongshore
# kx = k * np.cos(theta_p_rad)
# ky = -k * np.sin(theta_p_rad) #ky is negative due to our coordinate system. + north and east, - south and west...ish based on FRF local coordinates.
# Lx = (2 * np.pi) / kx
# Ly = (2 * np.pi) / ky
#
# #'wave-steepness' like nondimensional parameter (Hs / L)
# # eps_x = p_amp / (rho * g * Lx) #essentially becomes Hs / Lx scaled by the pressure response factor Kp
# # eps_y = p_amp / (rho * g * Ly) #nondimensionalzing the pressure amplitude is not circular like dpdx or dpdy because it does not depend on the wavenumber
# eps_x = dpdx / (rho * g)
# eps_y = dpdy / (rho * g)
#
# #open the das data from the same day
# # das_data_path = './data/for_OSU_processed/20260601/Processed_data_2026_06_01_00_05_01.h5'
# das_data_path = './data/for_OSU_processed/20260611/Processed_data_2026_06_11_00_05_01.h5'
# das_freqs, das_PSD = get_DAS_PSD(das_data_path, nperseg=256)
# das_wave_band = (das_freqs >= 0.05) & (das_freqs <= 0.3)  # wave band defined over the das PSD frequency bins
# das_amplitude = np.sqrt(np.trapezoid(das_PSD[das_wave_band, :], das_freqs[das_wave_band], axis=0)) #single number summarizing the total energy across the entire band. Bulk quantity
#
# #correlate dpdx and dpdy with the DAS amplitude - pearson spatial correlation
# dpdx_mask = np.isfinite(das_amplitude) & np.isfinite(dpdx)
# dpdx_matrix = np.corrcoef(np.abs(dpdx[dpdx_mask]), das_amplitude[dpdx_mask]) #how does DAS amplitude respond to the magnitude of the pressure gradient?
#
# dpdy_mask = np.isfinite(das_amplitude) & np.isfinite(dpdy)
# dpdy_matrix = np.corrcoef(np.abs(dpdy[dpdy_mask]), das_amplitude[dpdy_mask])
#
# #open the interpolated grid for debugging
# channel_grid = pd.read_pickle('./data/frf_awac/channel_grid.pkl')
# channel_x = channel_grid['xFRF']
# z = channel_grid['z']
#
# # plt.scatter(channel_x, z)
# # plt.show()
# # plt.close('all')
#
# #PLOTTING
# #plot pressure gradients
# fig, axes = plt.subplots(2, 1, figsize=(12, 8))
#
# # dpdx_plot = np.nan_to_num(dpdx, nan=0.0)
# axes[0].scatter(channels, dpdx, color='blue', s=3)
# axes[0].set_xlabel('Channel Number')
# axes[0].set_ylabel('dp/dx (Pa/m)')
# axes[0].set_xlim([0, channels.max()])
#
# # dpdy_plot = np.nan_to_num(dpdy, nan=0.0)
# axes[1].scatter(channels, dpdy, color='red', s=3)
# axes[1].set_xlabel('Channel Number')
# axes[1].set_ylabel('dp/dy (Pa/m)')
# axes[1].set_xlim([0, channels.max()])
#
# fig.suptitle('Bulk Pressure Gradient v. Channel Number')
#
# plt.tight_layout()
# plt.savefig(f'./figures/{timestamp}_bulkPresGrad.png')
# plt.show()
# plt.close('all')
#
# # #plot pressure amplitude
# # fig, axes = plt.subplots(2, 1, figsize=(12, 8))
# #
# # axes[0].scatter(channels, eps_x, color='blue', s=3)
# # axes[0].set_xlabel('Channel Number')
# # axes[0].set_ylabel('ε_x (Hs/Lx)')
# # axes[0].set_xlim([0, channels.max()])
# #
# # axes[1].scatter(channels, eps_y, color='red', s=3)
# # axes[1].set_xlabel('Channel Number')
# # axes[1].set_ylabel('ε_y (Hs/Ly)')
# # axes[1].set_xlim([0, channels.max()])
# #
# # fig.suptitle('Bulk Pressure Amplitude v. Channel Number')
# #
# # plt.tight_layout()
# # plt.show()
# # plt.close('all')
#
# #plot bulk das amplitude
# fig, axes = plt.subplots(1, 1, figsize=(12, 8))
#
# axes.plot(channels, das_amplitude, color='blue')
# axes.set_xlabel('Channel Number')
# axes.set_ylabel('Phase Shifts (rad)')
# axes.set_title('Bulk DAS Amplitude Over Channels')
#
# plt.tight_layout()
# plt.savefig(f'./figures/{timestamp}_bulkDasAMP.png')
# plt.show()
# plt.close('all')


'''CHECK OF CALIBRATION - SHOALING AND REFRACTION'''
# #open calibrated file
# # data_path = './data/frf_awac/calibration_coefficients/Calib_2026-06-01_00_05_01.npz'  #gets the data from one 5 minute DAS file - SHORE NORMAL
# # data_path = './data/frf_awac/calibration_coefficients/Calib_2026-06-08_23_00_01.npz' #WAVES FROM THE NORTH
# data_path = './data/frf_awac/calibration_coefficients/Calib_2026-06-07_00_05_01.npz' #WAVES FROM THE SOUTH
# data_struc = np.load(data_path, allow_pickle=True)
# data = data_struc['arr_0'].item()
#
# dirspec = data['awac_corrected_directional_spectrum']
# bad_channels = np.all(np.isnan(dirspec), axis=(1, 2)) #find which channels are bad - nans
# dirspec_clean = np.nan_to_num(dirspec, nan=0.0) #of shape (N_channels, N_freqs, N_dirs)
# freqs = data['awac_frequency_bins']
# dirs = data['awac_directional_bins']
# h = data['water_depth'] #water depth should be corrected for tides.
# channels = data['channel_number']
#
# df = np.gradient(freqs)
# dtheta = np.gradient(dirs)
#
# #pull out the representative wave for each channel
# #significant wave height - Hmo
# awac_wave_band = (freqs >= 0.05) & (freqs <= 0.3)
# m0 = np.trapezoid(np.trapezoid(dirspec_clean[:, awac_wave_band, :], dirs, axis=-1), freqs[awac_wave_band], axis=-1)
# Hs = 4.004 * np.sqrt(m0)
# Hs[bad_channels] = np.nan
#
# #frequency spectrum
# S_f = np.trapezoid(dirspec_clean, dirs, axis=-1)
# #direction spectrum
# S_theta = np.trapezoid(dirspec_clean, freqs, axis=1)
#
# #peak frequency
# fp_idx = np.argmax(S_f, axis=1) #indices of the peak frequency bin
# fp = freqs[fp_idx] #peak frequency at each channel
# fp[bad_channels] = np.nan #restore invalid channels
#
# #peak direction
# theta_p_idx = np.argmax(S_theta, axis=1) #indices of the peak direction bin
# theta_p = dirs[theta_p_idx] #peak direction at each channel
# theta_p[bad_channels] = np.nan #restore invalid channels
#
# #pull the DAS channel grid to get the depth and calculate local wavenumber at each channel
# Tp = 1/fp #peak period
# k = wavenumber(h, Tp) #local wave number
#
# Kp = 1 / (np.cosh(k*h)) #pressure response factor
#
# #pressure gradient amplitude in x and y build from bulk statistics
# rho = 1025 #kg/m**3
# g = 9.81 #m/s**2
#
# # rotate to shore-normal frame and flip to propagation direction
# theta_p_local = theta_p - 71.8535 #convert to xFRF shore-normal orientation
#
# #REFRACTION
# plt.plot(channels, theta_p_local)
# plt.xlabel('channel number')
# plt.ylabel('degrees away from shore normal (0)')
# plt.show()
# plt.close('all')
#
# #SHOALING
# plt.plot(channels, Hs)
# plt.xlabel('channel number')
# plt.ylabel('Significant Wave Height (Hmo)')
# plt.show()
# plt.close('all')


'''CHECK OF CALIBRATION COEFFICIENT - S_eta(x, f)'''
#use the calibration to check the spectral area matching at different channels in the cross-shore
#TODO: remember that the lack of good bathymetry will be present in this as well. The DAS spectrum will vary from the shoaled awac spectrum
#TODO: As I go into the nearshore, the spectral peak of the AWAC increasingly overestimates the spectral peak of the DAS. This is indicative of the energy dissipation term being inaccurate.
#TODO: The alongshore spectrum is also junk - probably need to isolate only the alongshore components of wave energy for this
#TODO: I am current comparing the entire spectrum to the DAS measurements. If the cable only response to along-cable pressure (gradients) then I need to isolate the components of the spectra.

#CROSS-SHORE CHANNEL CLOSE TO THE AWAC
channel = 1000
#
#pull DAS spectrum
# das_data_path = './data/for_OSU_processed/20260611/Processed_data_2026_06_11_00_10_01.h5' #one 5 minute file
all_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)
start_time = datetime(2026, 6, 11, 0, 0, 0)

burst_files = get_burst_files(all_paths, start_time, 35)

# das_freqs, das_PSD = get_DAS_PSD(das_data_path, nperseg=256)
das_freqs, das_PSD = get_DAS_PSD_burst(burst_files, nperseg=256, sampling_frequency=2)
das_wave_band = (das_freqs >= 0.05) & (das_freqs <= 0.3)
das_PSD = das_PSD[das_wave_band, :]
das_freqs = das_freqs[das_wave_band]

#pull AWAC spectrum
awac_data_path = './data/frf_awac/calibration_coefficients_burst/Calib_2026-06-11_00_00_00.npz'
awac_data = np.load(awac_data_path, allow_pickle=True)
awac_data = awac_data['arr_0'].item()

C = awac_data['calibration_coefficient']
awac_dirspec = awac_data['awac_corrected_directional_spectrum']
awac_freqs = awac_data['awac_frequency_bins']
awac_dirs = awac_data['awac_directional_bins']
h = awac_data['water_depth']
calibration_flag = awac_data['calibration_flag']  # flag for whether the calibration worked or failed a check, 1 is good, 0 means fail

#correct DAS spectrum to convert from phase_shifts to sea surface
Sp_f_das = (das_PSD[:, channel] * C[channel]) #the calibration coefficient is for a pressure spectra

#integrate awac directional spectrum over directional bins
awac_wave_band = (awac_freqs >= 0.05) & (awac_freqs <= 0.3)
awac_freqs = awac_freqs[awac_wave_band]
Seta_f_awac = np.trapezoid(awac_dirspec[:, awac_wave_band, :], awac_dirs, axis=-1).T

#convert both awac and das SSE spectrum to pressure spectrum.
# define constants
rho = 1025
g = 9.81
h0 = h[channel]

awac_T = 1 / awac_freqs
awac_k = wavenumber(h0, awac_T) #wavenumber at each period / frequency bin
awac_Kp = 1 / np.cosh(awac_k * h0) #pressure response factor at each wavenumber

das_T = 1 / das_freqs
das_k = wavenumber(h0, das_T)
das_Kp = 1 / np.cosh(das_k * h0)

Seta_f_das = Sp_f_das / (rho * g * das_Kp)**2
Sp_f_awac = (Seta_f_awac[:, channel] * (rho * g * awac_Kp)**2) / 1e8 #convert pascals to decibars

#plot the spectra on the same plot
fig, axes = plt.subplots(1,1, figsize=(12, 8))

axes.plot(awac_freqs, Sp_f_awac, color='blue', label='awac pressure')
axes.plot(das_freqs, Sp_f_das / 1e8, color='red', label='DAS-derived') #- convert to dbars (must square for units)
# axes.plot(awac_freqs, Seta_f_awac[:, channel], color='blue')
# axes.plot(das_freqs, Seta_f_das, color='red')
# axes2 = axes.twinx()
# axes2.plot(das_freqs, Sp_f_das, color='red', label='DAS - derived')
axes.set_xlabel('Frequency (Hz)')
axes.set_ylabel('Pressure PSD (dbar²/Hz)')
axes.set_title(f'Spectral Area Matching; Channel: {channel}, Time: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
axes.legend()

plt.tight_layout()
plt.savefig(f'./figures/area_matching/channel_{channel}_time_{start_time}_area.png')
plt.show()
plt.close('all')



'''IDEALIZED DAS CABLE GRID'''

#
# #offshore LARC survey
# #DAS survey locations
# offshore_survey_path = './DAS_deploy/Alex_FiberLay_20250730.csv'
#
# offshore_survey = pd.read_csv(offshore_survey_path)
# offshore_survey_x = offshore_survey['X']
# offshore_survey_y = offshore_survey['Y']
#
# offshore_frf_x = []
# offshore_frf_y = []
# for x, y in zip(offshore_survey_x, offshore_survey_y):
#     dict = ncsp2FRF(x, y)
#     offshore_frf_x.append(dict['xFRF'])
#     offshore_frf_y.append(dict['yFRF'])
#
# offshore_frf_x = np.array(offshore_frf_x)
# offshore_frf_y = np.array(offshore_frf_y)
#
# dx = np.diff(offshore_frf_x)
# dy = np.diff(offshore_frf_y)
#
# distance = np.sqrt(dx**2 + dy**2)
#
# s_survey = np.concatenate(([0], np.cumsum(distance))) #total distance along the cable in meters from the survey
#
# current_length = s_survey.max()
#
# N_channels = 1592 #number of DAS channels
#
# target_length = (N_channels // 2) * 1.6
#
# missing_length = target_length - current_length
#
# spacing = 1.6 #channel spacing
# n_extra = int(np.ceil(missing_length / spacing))
#
# # Extend in the -x direction from the first survey point
# x_extra = offshore_frf_x[0] - np.arange(n_extra, 0, -1) * spacing
#
# # y remains constant because the cable is normal to x
# y_extra = np.full(n_extra, offshore_frf_y[0])
#
# # Prepend
# offshore_frf_x = np.concatenate((x_extra, offshore_frf_x))
# offshore_frf_y = np.concatenate((y_extra, offshore_frf_y))
#
# dx = np.diff(offshore_frf_x)
# dy = np.diff(offshore_frf_y)
#
# distance = np.sqrt(dx**2 + dy**2)
#
# full_distance = np.concatenate(([0], np.cumsum(distance)))  #total distance along the cable
#
# #find where the cable lays in x FRF to match it with a depth from the larc survey.
# larc_data_path = './data/frf_awac/FRF_geomorphology_elevationTransects_survey_20260611.nc'
# larc_data = xr.open_dataset(larc_data_path)
# larc_x = larc_data.xFRF.values
# larc_y = larc_data.yFRF.values
# profile_number = larc_data.profileNumber.values
# time = larc_data.elevation.coords['time'].values
# elevation = larc_data.elevation.values  #depth profile
#
# #plot profile
# # plt.scatter(larc_x, larc_y, c=elevation)
# # plt.show()
# # plt.close('all')
#
# #pull just the bathymetry data along the yFRF=1000m transect
# cross_shore_profile_mask = (profile_number==1006)
# masked_x = larc_x[cross_shore_profile_mask]
# masked_y = larc_y[cross_shore_profile_mask]
# masked_elevation = elevation[cross_shore_profile_mask]
#
# #pull just the bathymetry data along the xFRF = 778
# profiles = np.unique(profile_number)
#
# xs = []
# ys = []
# zs = []
#
# for p in profiles:
#     mask = profile_number == p
#
#     idx = np.argmin(np.abs(larc_x[mask] - 778))
#
#     xs.append(larc_x[mask][idx])
#     ys.append(larc_y[mask][idx])
#     zs.append(elevation[mask][idx])
#
# xs = np.array(xs)  #alongshore profile
# ys = np.array(ys)
# zs = np.array(zs)
#
#
# #build grid to populate - in meters
# # cable_grid = np.arange(0, grid_len, 1.6)  #1.6m channel spacing
# cable_grid = np.arange(0, int(full_distance.max()), 1.6)
#
# # interpolate outbound leg x,y from the surveyed fiber path - given a distance along the coordinate whats the x or y coordinate
# interp_x = interp1d(full_distance, offshore_frf_x, bounds_error=False, fill_value='extrapolate')
# interp_y = interp1d(full_distance, offshore_frf_y, bounds_error=False, fill_value='extrapolate')
#
# x_out = interp_x(cable_grid)
# y_out = interp_y(cable_grid)
#
# # mirror for the return leg to fill out N_channels
# n_return = N_channels - len(cable_grid)
# x_ret = x_out[::-1][:n_return]
# y_ret = y_out[::-1][:n_return]
#
# channel_x = np.concatenate([x_out, x_ret])
# channel_y = np.concatenate([y_out, y_ret])
# channel_number = np.arange(N_channels)
#
# # split channels by whether they're on the cross-shore or alongshore leg
# cross_shore_mask = channel_x <= 777 #xFRF location where the cable should turn
# alongshore_mask = channel_x > 777
#
# channel_z = np.full(N_channels, np.nan)
#
# # cross-shore leg: z as a function of xFRF
# sort_idx_x = np.argsort(masked_x)
# interp_z_crossshore = interp1d(masked_x[sort_idx_x], masked_elevation[sort_idx_x],
#                                 bounds_error=False, fill_value='extrapolate')
# channel_z[cross_shore_mask] = interp_z_crossshore(channel_x[cross_shore_mask])  #given an xFRF value what is the seafloor elevation here - cross-shore only
#
# # alongshore leg: z as a function of yFRF
# sort_idx_y = np.argsort(ys)
# interp_z_alongshore = interp1d(ys[sort_idx_y], zs[sort_idx_y],
#                                 bounds_error=False, fill_value='extrapolate')
# channel_z[alongshore_mask] = interp_z_alongshore(channel_y[alongshore_mask])  #given a yFRF value what is the seafloor elevation here - alongshore only
#
# channel_grid = pd.DataFrame({
#     'channel': channel_number,
#     'xFRF': channel_x,
#     'yFRF': channel_y,
#     'z': channel_z,
# })
#
# # save the output grid.
# channel_grid.to_pickle('./data/frf_awac/channel_grid.pkl')  #TODO: make this a function in a new utilities script
#
# #sanity plot
# plt.scatter(offshore_frf_x, offshore_frf_y, c='orange', s=6)
# plt.scatter(channel_grid['xFRF'], channel_grid['yFRF'], c=channel_grid['z'])
# plt.show()
# plt.close('all')



'''DAS SIGNAL V WAVE DIRECTION ACROSS TIME'''
das_file_paths = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number) #all processed DAS files
calibration_file_paths = sorted(glob.glob('./data/frf_awac/calibration_coefficients_burst/*'), key=sequence_number)

start_time = datetime(2026, 6, 1, 0, 0, 0)
end_time = datetime(2026, 6, 15, 23, 59, 59)
orientation = 'alongshore' #alongshore or cross_shore oriented channel
channel = 750 #channel to investigate

rho = 1025 #kg/m**3
g = 9.81 #m/s**2

# das_file_paths = [
#     f for f in das_file_paths
#     if start_time <= sequence_number(f) <= end_time
# ]
#
# calibration_file_paths = [
#     f for f in calibration_file_paths
#     if start_time <= sequence_number(f) <= end_time
# ]
#
# #get burst groups for DAS
# # group into top of the hour 35 minute bursts of files
# BURST_DURATION_MIN = 35
# MIN_FILES_PER_BURST = 6  # accept bursts with 6 or 7 files (allows for edge-of-dataset partial hours)
# BURST_TOLERANCE_SEC = 0 #tolerance in seconds
#
# bursts = defaultdict(list)  #TODO: Must explain everything this does.
# for f in das_file_paths:
#     ts = parse_das_timestamp(f)
#     if ts is None:
#         continue
#     burst_start = ts.replace(minute=0, second=0, microsecond=0)
#     burst_end = burst_start + timedelta(minutes=BURST_DURATION_MIN, seconds=BURST_TOLERANCE_SEC)
#     if burst_start <= ts < burst_end:
#         bursts[burst_start].append(f)
#
# # keep only complete bursts, and sort by time
# complete_bursts = {
#     b: sorted(files, key=sequence_number)
#     for b, files in bursts.items()
#     if len(files)  >= MIN_FILES_PER_BURST
# }
# burst_items = sorted(complete_bursts.items())  # list of (burst_start, [files]) tuples
#
# dpdx_array = np.zeros(len(calibration_file_paths)) #empty arrays to fill with data
# dpdy_array = np.zeros(len(calibration_file_paths))
# das_amplitude_array = np.zeros(len(calibration_file_paths))
# timestamps = []
# eps_x_array = np.zeros(len(calibration_file_paths))
# eps_y_array = np.zeros(len(calibration_file_paths))
# theta_incident = np.full(len(calibration_file_paths), np.nan)
# bulk_Hs = np.full(len(calibration_file_paths), np.nan)
#
# #SINGLE FILES
# # for i, (das_file, calibration_file) in enumerate(
# #     tqdm(
# #         zip(das_file_paths, calibration_file_paths),
# #         total=len(das_file_paths),
# #         colour='yellow'
# #     )
# # ):
#
# #BURST FILES
# for i, ((burst_start, burst_files), calibration_file) in enumerate(
#     tqdm(
#         zip(burst_items, calibration_file_paths),
#         total=len(calibration_file_paths),
#         colour='yellow'
#     )
# ):
#
#     data_path = calibration_file  #get data from one five minute DAS file
#     data_struc = np.load(data_path, allow_pickle=True)
#     data = data_struc['arr_0'].item()
#
#     dirspec = data['awac_corrected_directional_spectrum'][channel, :, :]
#     dirspec_clean = np.nan_to_num(dirspec, nan=0.0) #of shape (N_channels, N_freqs, N_dirs)
#     freqs = data['awac_frequency_bins']
#     dirs = data['awac_directional_bins']
#     h = data['water_depth'][channel] #water depth should be corrected for tides.
#     channels = data['channel_number']
#
#     df = np.gradient(freqs)
#     dtheta = np.gradient(dirs)
#
#     #get timestamp to save file
#     match = re.search(r'(\d{4}\D+\d{2}\D+\d{2}\D+\d{2}\D+\d{2}\D+\d{2})',
#                               data_path)  # extract the timestamp from the first file name
#     timestamp = match.group(1)  # pull the timestamp from the regex object
#
#     #pull out the representative wave
#     #significant wave height - Hmo
#     awac_wave_band = (freqs >= 0.05) & (freqs <= 0.3)
#     m0 = np.trapezoid(np.trapezoid(dirspec_clean[awac_wave_band, :], dirs, axis=-1), freqs[awac_wave_band], axis=-1)
#     Hs = 4.004 * np.sqrt(m0)  #broadband Hs
#     bulk_Hs[i] = Hs
#
#     #frequency spectrum
#     S_f = np.trapezoid(dirspec_clean[awac_wave_band, :], dirs, axis=-1)
#     #direction spectrum
#     S_theta = np.trapezoid(dirspec_clean[awac_wave_band, :], freqs[awac_wave_band], axis=0)
#
#     #NARROW BAND SPECTRAL PEAK APPROACH
#     # #peak frequency
#     # fp_idx = np.argmax(S_f) #indices of the peak frequency bin
#     # fp = freqs[fp_idx] #peak frequency at each channel
#     #
#     # #peak direction
#     theta_p_idx = np.argmax(S_theta) #indices of the peak direction bin
#     theta_p = dirs[theta_p_idx] #peak direction at each channel
#     #
#     # rotate to shore-normal frame and flip to propagation direction
#     theta_p_local = theta_p - 71.8535 #convert to xFRF shore-normal orientation
#     theta_p_prop = theta_p_local + 180  #converts waves coming from direction to waves propagating towards the shoreline
#     theta_p_rad = np.deg2rad(theta_p_prop) #convert degrees to radians
#
#     theta_incident[i] = theta_p_local #append the peak direction to an array for plotting
#     #
#     # #pull the DAS channel grid to get the depth and calculate local wavenumber at each channel
#     # Tp = 1/fp #peak period
#     # k = wavenumber(h, Tp) #local wave number
#     #
#     # Kp = 1 / (np.cosh(k*h)) #pressure response factor
#
#     #pressure gradient amplitude in x and y build from bulk statistics
#     #pressure gradients - remember x increases offshore.
#     # dpdx = (0.5 * rho * g * Hs * k * np.cos(theta_p_rad) * Kp) / 10000 #units of decibars/m - fluid pressure is increasing/decreasing for every space along the x grid.
#     # dpdy = -(0.5 * rho * g * Hs * k * np.sin(theta_p_rad) * Kp) / 10000 #units of decibars/m - negative because positive xFRF is to the North (to the left of the shore normal transect)
#     #a wave coming from the south (dirs=180 - 71.8535) gives a positive dpdy and a wave from the north (dirs=0 - 71.8535) gives a negative dpdy
#
#     #wavelengths: cross and alongshore
#     # kx = k * np.cos(theta_p_rad)
#     # ky = -k * np.sin(theta_p_rad) #ky is negative due to our coordinate system. + north and east, - south and west...ish based on FRF local coordinates.
#
#     #pressure amplitude
#     # p_amp = 0.5 * rho * g * Hs * Kp_f
#
#     # Lx = (2 * np.pi) / kx
#     # Ly = (2 * np.pi) / ky
#
#     #'wave-steepness' like nondimensional parameter (Hs / L)
#     # eps_x = p_amp / (rho * g * Lx) #essentially becomes Hs / Lx scaled by the pressure response factor Kp
#     # eps_y = p_amp / (rho * g * Ly) #nondimensionalzing the pressure amplitude is not circular like dpdx or dpdy because it does not depend on the wavenumber
#     # eps_x = dpdx / (rho * g)
#     # eps_y = dpdy / (rho * g)
#
#     #BROADBAND SPECTRAL APPROACH
#     k_f = np.array([wavenumber(h, 1/f) for f in freqs[awac_wave_band]])
#     Kp_f = 1 / np.cosh(k_f * h)
#
#     dirs_local = dirs - 71.8535
#     dirs_rad = np.deg2rad(dirs_local + 180)  # rotate + flip to propagation
#
#     S_f_band = dirspec_clean[awac_wave_band, :] #define sse spectrum over wave band
#
#     S_dpdx = (rho * g * k_f[:, None] * Kp_f[:, None] * np.cos(dirs_rad)[None, :])**2 * S_f_band #define the pressure gradient spectrum
#     S_dpdy = (rho * g * k_f[:, None] * Kp_f[:, None] * np.sin(dirs_rad)[None, :])**2 * S_f_band
#
#     S_dpdx_f = np.trapezoid(S_dpdx, dirs, axis=1) #integrate over direction
#     S_dpdy_f = np.trapezoid(S_dpdy, dirs, axis=1)  # integrate over direction
#
#     dpdx = np.sqrt(np.trapezoid(S_dpdx_f, freqs[awac_wave_band])) / 10000 #integrate over frequency and convert to dbar/m  #TODO: do I need to square the 10000 here?
#     dpdy = np.sqrt(np.trapezoid(S_dpdy_f, freqs[awac_wave_band])) / 10000  # integrate over frequency and convert to dbar/m
#
#     #open the das data from the same day
#     das_data_path = burst_files
#     das_freqs, das_PSD = get_DAS_PSD_burst(das_data_path, nperseg=256, sampling_frequency=2)
#     das_PSD = das_PSD[:, channel]
#     das_wave_band = (das_freqs >= 0.05) & (das_freqs <= 0.3)  # wave band defined over the das PSD frequency bins
#     das_amplitude = np.sqrt(np.trapezoid(das_PSD[das_wave_band], das_freqs[das_wave_band], axis=0)) #single number summarizing the total energy across the entire band. Bulk quantity - currently broadband
#
#     dpdx_array[i] = dpdx
#     dpdy_array[i] = dpdy
#     das_amplitude_array[i] = das_amplitude
#     timestamps.append(datetime.strptime(timestamp, "%Y-%m-%d_%H_%M_%S"))
#     # eps_x_array[i] = eps_x
#     # eps_y_array[i] = eps_y
#
# #save the arrays
# data = {
#     'dpdx':dpdx_array,
#     'dpdy':dpdy_array,
#     'timestamps':timestamps,
#     'das_amplitude':das_amplitude_array,
#     'theta_incident':theta_incident,
#     'Hs':bulk_Hs
# }
#
# np.savez(f'./data/frf_awac/wave_direction/Channel_{channel}_wavedir_vars_burst.npz', data)

saved_data = np.load(f'./data/frf_awac/wave_direction/Channel_{channel}_wavedir_vars_burst.npz', allow_pickle=True)

saved_data = saved_data['arr_0'].item()

dpdx_array = saved_data['dpdx']
dpdy_array = saved_data['dpdy']
timestamps = saved_data['timestamps']
das_amplitude_array = saved_data['das_amplitude']
# eps_x_array = saved_data['eps_x']
# eps_y_array = saved_data['eps_y']
theta_incident = saved_data['theta_incident']
bulk_Hs = saved_data['Hs']

#calculate the magnitude of the total pressure gradient
dp_mag = np.sqrt(dpdx_array**2 + dpdy_array**2)

#normalize the das amplitude
if orientation == 'cross_shore':
    grad = np.abs(dpdx_array)
if orientation == 'alongshore':
    grad = np.abs(dpdy_array)

denom_mask = grad > 0.05 * np.nanmax(grad)  # drop near-zero-gradient bursts
das_norm = np.where(denom_mask, das_amplitude_array / grad, np.nan)

mask = np.isfinite(grad) & np.isfinite(das_amplitude_array)
r, p = stats.pearsonr(grad[mask], das_amplitude_array[mask])
slope, intercept, r_lin, p_lin, se = stats.linregress(grad[mask], das_amplitude_array[mask])

x_fit = np.linspace(np.nanmin(grad[mask]), np.nanmax(grad[mask]))
y_fit = slope * x_fit + intercept

#MASKS FOR ISOLATING WAVE HEIGHT AND DIRECTION
theta_mask = (np.abs(theta_incident) >= 10)
# small_Hs_mask = (bulk_Hs < 0.75)
# mid_Hs_mask = (0.75 <= bulk_Hs) & (bulk_Hs < 1.25)
# large_Hs_mask = (1.25 <= bulk_Hs)

#pearson correlation
print(f"r={r:.2f}, p={p:.4f}, slope={slope:.3f}")

#bin by wave height
# bin_edges = np.arange(-90, 91, 5)
# med, edges, _ = binned_statistic(theta_incident, das_amplitude_array, statistic='median', bins=bin_edges)
# centers = 0.5*(edges[:-1]+edges[1:])

#WAVE DIRECTION AND SIGNIFICANT WAVE HEIGHT
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].scatter(timestamps, bulk_Hs, color='blue', s=3)
axes[0].set_xlabel('Time (UTC)')
axes[0].set_ylabel('Significant Wave Height (m)')
axes[0].set_title('Significant Wave Height v Time')
axes[0].set_facecolor('white')

axes[1].scatter(timestamps, theta_incident, color='red', s=3)
axes[1].set_xlabel('Time (UTC)')
axes[1].set_ylabel('Incident Wave Direction (Degrees)')
axes[1].set_title('Incident Wave Direction v Time')
axes[1].axhline(0, linestyle='--', color='black')
axes[1].invert_yaxis() #negative direction would be from the north
axes[1].set_facecolor('white')

plt.tight_layout()
plt.savefig(f'./figures/wave_information/Channel_{channel}_Hs_and_theta_{start_time}_{end_time}.png')
# plt.show()
plt.close('all')


#PRESSURE GRADIENT AMPLITUDE AND DAS AMPLITUDE OVER TIME
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

if orientation == 'cross_shore':
    axes[0].scatter(timestamps, np.abs(dpdx_array), color='blue', s=3)
    axes[0].set_ylabel('|dP/dx| (dbar/m)')
if orientation == 'alongshore':
    axes[0].scatter(timestamps, np.abs(dpdy_array), color='blue', s=3)
    axes[0].set_ylabel('|dP/dy| (dbar/m)')

axes[0].set_xlabel('Time (UTC)')
axes[0].set_title(f'Pressure Gradient Magnitude v Time')
axes[0].set_facecolor('white')

axes[1].scatter(timestamps, das_amplitude_array, color='red', s=3)
axes[1].set_xlabel('Time (UTC)')
axes[1].set_ylabel('DAS Amplitude (rad)')
axes[1].set_title('DAS Response Magnitude v Time')
axes[1].set_facecolor('white')

fig.suptitle(f'Correlation: r={r:.2f}, p={p:.4f}')

plt.tight_layout()
plt.savefig(f'./figures/{orientation}/Channel_{channel}_DasAmp_pGrad_v_Time_{start_time}_{end_time}.png')
# plt.show()
plt.close('all')


#DAS AMPLITUDE V SIGNIFICANT WAVE HEIGHT
fig, axes = plt.subplots(1, 1, figsize=(12, 12))

axes.scatter(bulk_Hs[theta_mask], das_amplitude_array[theta_mask],  color='blue', s=5)
# axes.set_xscale('log')
# axes.set_yscale('log')
axes.set_xlabel('Significant Wave Height (m)')
axes.set_ylabel('DAS Amplitude (rad)')
axes.set_title('Magnitude of DAS Response v Significant Wave Height')
axes.set_facecolor('white')

plt.tight_layout()
plt.savefig(f'./figures/{orientation}/Channel_{channel}_DasAmp_v_HS_ShoreNormal_{start_time}_{end_time}.png')
# plt.show()
plt.close('all')


#DAS AMPLITUDE V DIRECTION
fig, axes = plt.subplots(1, 1, figsize=(14, 12))

sc = axes.scatter(theta_incident, das_amplitude_array, c=bulk_Hs, s=5)
# axes.plot(centers, med, color='black', linewidth=2, zorder=5, label='Binned Median')
axes.set_xlabel('Incident Wave Direction (degrees)')
axes.set_ylabel('DAS Amplitude (rad)')
axes.set_title('Incident Wave Direction v DAS Response')
axes.set_facecolor('white')
axes.legend()
fig.colorbar(sc, ax=axes, label='Bulk Hs')

plt.tight_layout()
plt.savefig(f'./figures/{orientation}/Channel_{channel}_DasAmp_v_Theta_{start_time}_{end_time}.png')
# plt.show()
plt.close('all')


#NORMALIZED DAS AMPLITUDE V DIRECTION
#for shore normal
fig, axes = plt.subplots(1, 1, figsize=(14, 12))

sc = axes.scatter(theta_incident, das_norm, c=bulk_Hs, s=5)
axes.set_xlabel('Incident Wave Direction (degrees)')
axes.set_ylabel('Normalized DAS Amplitude (rad * m / dbar)')
axes.set_title('Incident Wave Direction v Normalized DAS Response')
axes.set_facecolor('white')
fig.colorbar(sc, ax=axes, label='Bulk Hs')

plt.tight_layout()
plt.savefig(f'./figures/{orientation}/Channel_{channel}_NormDasAmp_v_Theta_ShoreNormal_{start_time}_{end_time}.png')
# plt.show()
plt.close('all')


#PRESSURE GRADIENT V DAS AMPLITUDE
fig, axes = plt.subplots(1, 1, figsize=(14, 12))

if orientation == 'cross_shore':
    # axes.scatter(np.abs(dpdx_array[theta_mask]), das_amplitude_array[theta_mask], c=np.abs(theta_incident)[theta_mask], s=25, marker='*', cmap='coolwarm', vmin=-60, vmax=60, label='Shore Normal Waves')
    sc = axes.scatter(np.abs(dpdx_array[theta_mask]), das_amplitude_array[theta_mask], c=np.abs(theta_incident[theta_mask]), s=15, cmap='Reds')#, vmin=-60, vmax=60)
    # axes.plot(x_fit, y_fit, color='magenta', linewidth=2, linestyle='--',
    #           label=f'fit: slope={slope:.3f}, $r$={r:.2f}, $p$={p:.2e}')
    axes.set_xlabel('|dP/dx| (dbar/m)')
    axes.set_title('Magnitude of DAS Response v Cross-Shore Pressure Gradient Magnitude')

if orientation == 'alongshore':
    # axes.scatter(np.abs(dpdy_array[theta_mask]), das_amplitude_array[theta_mask], c=np.abs(theta_incident)[theta_mask], s=10,
    #              marker='*', cmap='coolwarm', vmin=-60, vmax=60, label='Shore Normal Waves')
    sc = axes.scatter(np.abs(dpdy_array), das_amplitude_array, c=np.abs(theta_incident), s=5, cmap='Reds')#, vmin=-60, vmax=60)
    # axes.plot(x_fit, y_fit, color='magenta', linewidth=2, linestyle='--',
    #           label=f'fit: slope={slope:.3f}, $r$={r:.2f}, $p$={p:.2e}')
    axes.set_xlabel('|dP/dy| (dbar/m)')
    axes.set_title('Magnitude of DAS Response v Alongshore Pressure Gradient Magnitude')

axes.set_ylabel('DAS Amplitude (rad)')
axes.set_facecolor('black')
axes.legend()
fig.colorbar(sc, ax=axes, label='Incident Wave Direction', cmap='Reds')

plt.tight_layout()
if orientation == 'cross_shore':
    plt.savefig(f'./figures/{orientation}/Channel_{channel}_dpdx_v_DasAmp_{start_time}_{end_time}.png')
if orientation == 'alongshore':
    plt.savefig(f'./figures/{orientation}/Channel_{channel}_dpdy_v_DasAmp_{start_time}_{end_time}.png')
# plt.show()
plt.close('all')


#PRESSURE GRADIENT V NORMALIZED DAS AMPLITUDE
fig, axes = plt.subplots(1, 1, figsize=(14, 12))   #TODO: add the x fit median line to this as well

if orientation == 'cross_shore':
    # axes.scatter(np.abs(dpdx_array[theta_mask]), das_norm[theta_mask], c=theta_incident[theta_mask], s=25,
    #              marker='*', cmap='coolwarm', vmin=-60, vmax=60, label='Shore Normal Waves')
    sc = axes.scatter(np.abs(dpdx_array[theta_mask]), das_norm[theta_mask], c=np.abs(theta_incident[theta_mask]), s=15, cmap='Reds')#, vmin=-60, vmax=60)
    axes.set_xlabel('|dP/dx| (dbar/m)')
    axes.set_title('Magnitude of Normalized DAS Response v Cross-Shore Pressure Gradient Magnitude')

if orientation == 'alongshore':
    # axes.scatter(np.abs(dpdy_array[theta_mask]), das_norm[theta_mask], c=theta_incident[theta_mask], s=25,
    #              marker='*', cmap='coolwarm', vmin=-60, vmax=60, label='Shore Normal Waves')
    sc = axes.scatter(np.abs(dpdy_array), das_norm, c=np.abs(theta_incident), s=15, cmap='Reds')#, vmin=-60, vmax=60)
    axes.set_xlabel('|dP/dy| (dbar/m)')
    axes.set_title('Magnitude of Normalized DAS Response v Alongshore Pressure Gradient Magnitude')

axes.set_ylabel('Normalized DAS Amplitude (rad*m/dbar)')
axes.set_facecolor('black')
axes.legend()
fig.colorbar(sc, ax=axes, label='Incident Wave Direction')

plt.tight_layout()
if orientation == 'cross_shore':
    plt.savefig(f'./figures/{orientation}/Channel_{channel}_dpdx_v_DasNormAmp_{start_time}_{end_time}.png')
if orientation == 'alongshore':
    plt.savefig(f'./figures/{orientation}/Channel_{channel}_dpdy_v_DasNormAmp_{start_time}_{end_time}.png')
# plt.show()
plt.close('all')


#PRESSURE GRADIENT V INCIDENT WAVE DIRECTION
# fig, axes = plt.subplots(1, 1, figsize=(14, 12))
#
# if orientation == 'cross_shore':
#     sc = axes.scatter(theta_incident, np.abs(dpdx_array), c=bulk_Hs, s=5)
#     axes.set_ylabel('|dP/dx| (dbar/m)')
#     axes.set_title('Incident Wave Direction v Cross-Shore Pressure Gradient Magnitude')
#
# if orientation == 'alongshore':
#     sc = axes.scatter(theta_incident, np.abs(dpdy_array), c=bulk_Hs, s=5)
#     axes.set_ylabel('|dP/dy| (dbar/m)')
#     axes.set_title('Incident Wave Direction v Alongshore Pressure Gradient Magnitude')
#
# axes.set_xlabel('Incident Wave Direction (degrees)')
# fig.colorbar(sc, ax=axes, label='Bulk Hs')
#
# plt.tight_layout()
# if orientation == 'cross_shore':
#     plt.savefig(f'./figures/{orientation}/Channel_{channel}_dpdx_v_Theta_{start_time}_{end_time}.png')
# if orientation == 'alongshore':
#     plt.savefig(f'./figures/{orientation}/Channel_{channel}_dpdy_v_Theta_{start_time}_{end_time}.png')
# # plt.show()
# plt.close('all')


# #RATIO OF DPDY/DPDX V WAVE DIRECTION
# fig, axes = plt.subplots(1, 1, figsize=(14, 12))
#
# sc = axes.scatter(theta_incident, np.abs(dpdy_array)/np.abs(dpdx_array), c=bulk_Hs, s=15)
# axes.set_xlabel('Wave Direction (degrees)')
# axes.set_ylabel('|dP/dy|/|dP/dx|')
# axes.set_title('Ratio of Alongshore and Cross-Shore Pressure Gradient v Incident Wave Direction')
# fig.colorbar(sc, ax=axes, label='Bulk Hs')
#
# plt.tight_layout()
# plt.savefig(f'./figures/{orientation}/Channel_{channel}_ratio_v_direction_{start_time}_{end_time}.png')
# # plt.show()
# plt.close('all')


'''COMPARISON OF ALONGSHORE AND CROSS-SHORE CHANNELS'''
# as_channel = 700
# cs_channel = 250
#
# cs = np.load(f'./data/frf_awac/Channel_{cs_channel}_wavedir_vars.npz', allow_pickle=True)
# cs = cs['arr_0'].item()
# as_ = np.load(f'./data/frf_awac/Channel_{as_channel}_wavedir_vars.npz', allow_pickle=True)
# as_ = as_['arr_0'].item()
#
# #make sure times are equal between the channels
# assert np.array_equal(cs['timestamps'], as_['timestamps']), "timestamp mismatch between channels"
#
# theta_cs = cs['theta_incident']
# das_ratio = as_['das_amplitude'] / cs['das_amplitude']
# tan_theta = np.tan(np.deg2rad(theta_cs))
#
# valid = np.abs(theta_cs) < 80  # exclude near shore-parallel where tan blows up
#
# mask_ratio = valid & np.isfinite(das_ratio) & np.isfinite(tan_theta)
# r_ratio, p_ratio = stats.pearsonr(tan_theta[mask_ratio], das_ratio[mask_ratio])
# slope_ratio, intercept_ratio, _, _, se_ratio = stats.linregress(tan_theta[mask_ratio], das_ratio[mask_ratio])
#
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# sc = ax.scatter(tan_theta[valid], das_ratio[valid], s=8, alpha=0.5, c=cs['Hs'][valid])
# fig.colorbar(sc, ax=ax, label='Hs (m)')
# lims = [-3, 3]
# ax.plot(lims, lims, linestyle='--', color='black', label='1:1 (predicted)')
# x_fit_r = np.linspace(*lims, 100)
# ax.plot(x_fit_r, slope_ratio * x_fit_r + intercept_ratio, color='green', linestyle=':',
#         label=f'fit: slope={slope_ratio:.2f}, r={r_ratio:.2f}, p={p_ratio:.2e}')
# ax.set_xlim(lims); ax.set_ylim(lims)
# ax.set_xlabel(r'$\tan(\theta)$')
# ax.set_ylabel('DAS Amplitude Ratio (AS/CS)')
# ax.legend()
# ax.set_facecolor('white')
#
# plt.tight_layout()
# plt.savefig(f'./figures/wave_information/Channel_cs{cs_channel}_as{as_channel}_DasAmpRatio.png')
# # plt.show()
# plt.close('all')
#
# fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
# sc = axes[0].scatter(cs['timestamps'], das_ratio, c=theta_cs, cmap='coolwarm', vmin=-90, vmax=90, s=6)
# axes[1].set_facecolor('white')
# fig.colorbar(sc, ax=axes[0], label='Direction (deg)')
# axes[1].scatter(cs['timestamps'], theta_cs, c='gray', s=4)
# axes[1].axhline(0, linestyle='--', color='black')
# axes[1].set_facecolor('white')
#
# plt.tight_layout()
# plt.savefig(f'./figures/wave_information/Channel_cs{cs_channel}_as{as_channel}_DasAmpRatiovTime.png')
# # plt.show()
# plt.close()







#TODO: DAS PSD is calculated from a 5 minute average. That is a short time period - noisy and may not match. The AWAC PSD is from a 34 minute burst. smoother and more robust.
#TODO: may need to run the empirical calibration again this time with the DAS averaged over the same time period


#TODO: Bulk DAS amplitude is a broadband quantity while the dpdx and dpdy forcing are both narrowband-monochromatic quantities. Either convert the bulk DAS amplitude to a monochromatic quantity or convert dpdx and dpdy to broadband
#TODO: 1. can spectrally derive the dpdx and dpdy.
#TODO: 2. evaluate DAS amplitude at the spectral peak like I am doing for dpdx and dpdy currently. Or integrate over a narrow band near the spectral peak to suppress noise
#TODO: started with one.

#TODO: may need to redo some stuff with teh new DAS intervals at 35 mins?




