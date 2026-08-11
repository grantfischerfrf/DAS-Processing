import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr
from FRF_DAS_processing import strain2phase
import re
from FRF_DAS_plots import load_processedData
from scipy.signal import welch
from tqdm import tqdm
import matplotlib.pyplot as plt


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

    dirspec = data_300['directionalWaveEnergyDensity']  # sea surface wave directional energy spectrum
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

def get_DAS_PSD(file_path, nperseg, noverlap=None, sampling_frequency=2):

    #read file
    strain, time = load_processedData(file_path)

    #convert strain to phase - original strain to phase equation is not suitable for this calibration
    phase = strain2phase(strain)

    if noverlap is None:
        noverlap = nperseg // 2

    freqs, PSD = welch(phase, fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap, axis=0) #compute PSD for each channel, returns PSD of shape (n_freqs, n_channels)

    return freqs, PSD

def compute_calibration_coefficient(files_to_process,
        save_folder,  #TODO: calibration coefficient is being save naively right now. There is no knowledge of what channel it belongs to. Save the channel number with the calibration coefficeint and a flag of whether it is good or not.
        channel_grid,
        S_f_theta_eta,
        awac_freqs,
        dirs,
        p_time,
        dtheta,
        df,
        T,
        gaugeDepth,
        gaugeElevation,
        bottomElevation,
        gamma_break):

    #define constants
    rho = 1025
    g = 9.81

    N_files = len(files_to_process) #number of files to process

    # get the depth across all channels
    bed = channel_grid['z'].values  # depth at all channels
    N_channels = len(bed) #number of channels in the DAS

    awac_wave_band = (awac_freqs >= 0.05) & (awac_freqs <= 0.3) #wave band to integrate over

    # precompute trig functions for wave refraction
    theta0_deg = dirs - 71.8535  # convert measure wave angle to xFRF shore normal convention for snells law - shore normal pointing offshore
    theta0_deg = (theta0_deg + 180) % 360 - 180  # wrap to -180, 180

    valid_theta = (
    # skip waves that are too oblique to xFRF. Causes the jacobian to converge to a singularity - very large values directional bins that shouldn't have a lot of energy
            (np.abs(theta0_deg) < 75) &
            (np.cos(np.deg2rad(theta0_deg)) > 0)  # skip non-shoreward propagating waves/bins
    )

    theta0_rad = np.deg2rad(theta0_deg)  # convert to radians
    cos_theta0 = np.cos(theta0_rad)  # precomputation of cos(theta0)
    sin_theta0 = np.sin(theta0_rad)  # precomputation of sin(theta0)

    gamma_break = gamma_break  # breaker index for random/Rayleigh-distributed wave field (Thornton & Guza, FRF-calibrated) - can use the Goda parameterization too if needed

    m0_list = []  # temp
    h1_list = []  # temp

    valid_channels = 0

    Channel_Numbers = np.array(range(N_channels)) + 1

    for t in tqdm(range(N_files), colour='yellow'):

        # find the starttime of the Das file
        match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})',
                          files_to_process[t])  # extract the timestamp from the first file name

        if match:
            timestamp = match.group(1)  # pull the timestamp from the regex object
        else:
            print(
                f"No timestamp found in file name: {files_to_process[t]}")  # break the loop and inspect file names instead
            break

        das_start_time = np.datetime64(datetime.strptime(timestamp, '%Y_%m_%d_%H_%M_%S'))  # starttime of the file

        # pull the DAS PSD for the appropriate file
        das_freqs, das_PSD_all_channels = get_DAS_PSD(files_to_process[t], nperseg=256,
                                                      sampling_frequency=2)  # this is in units of m**2/Hz

        # closest AWAC burst
        idx = np.argmin(np.abs(p_time - das_start_time))

        awac_start_time = p_time[idx]

        S_f_theta_eta_selected = S_f_theta_eta[idx] # select the AWAC burst closest to the DAS file start time - squeeze extra time dimension (1, 62, 72) -> shape (62, 72)

        gaugeDepth0 = gaugeDepth.values[idx] # sensor depth using pressure
        bottomElevation0 = bottomElevation.values[idx] # the elevation of the seabed in reference to NAVD88
        gaugeElevation0 = gaugeElevation.values.item() # the distance from the NAVD88 reference level to the top of the gauge

        depth_from_bottom = gaugeElevation0 - bottomElevation0  # height of sensor off the bottom of the bed - array
        h0 = gaugeDepth0 + depth_from_bottom # full water depth - array

        surfaceElevation = gaugeElevation0 + gaugeDepth0

        k0 = wavenumber(h0, T)  # wavenumber at the AWAC at time t

        calibration_flag = np.ones(N_channels,
                                   dtype=int)  # flag for whether the calibration worked or failed a check, 1 is good, 0 means fail

        C = np.full(N_channels, np.nan, dtype=np.float32)  # empty array to hold the calibration coefficient c(t, x)

        water_depth = np.full(N_channels, np.nan)

        directional_spectrum = np.full((N_channels, len(awac_freqs), len(dirs)), np.nan, dtype=np.float32) #nan array to fill with directional spectrum data - if bad calibration it stays as nans

        for ch in range(N_channels):

            # skip the first ~100 channels of the DAS cable, there is no signal there  #TODO: May need to do this to keep the calibration factor valid.
            # if ch < 120 or ch > N_channels - 120:
            #     continue

            pressure_spectrum = np.zeros(len(awac_freqs)) #empty array to hold pressure spectrum

            # pull the DAS PSD at the current channel and time step
            das_PSD = das_PSD_all_channels[:, ch]
            # TODO: probably will need to average the PSD for the 35 minute burst.

            h1 = surfaceElevation - bed[ch] #depth at each point - corrected for tides?

            # h1_list.append(h1)

            if h1 <= 0:  # h1 must be positive for dispersion relation (wavenumber calculation) can use 1
                calibration_flag[ch] = 0
                continue

            water_depth[ch] = h1 #append the water_depth for each channel #TODO: make sure this is correct - want to correct for tides but not resolving individual waves for h.

            valid_channels += 1  # add a count for each channel that is allowed to pass through

            S1_local = np.zeros((len(awac_freqs), len(dirs)), dtype=np.float32)

            k1_arr = np.zeros(len(awac_freqs))  # store k1 per frequency - no recomputation later

            for i, f in enumerate(awac_freqs):

                k0_i = k0[i]  # wavenumber at the AWAC at time t and frequency f

                # compute the local wavenumber
                k1 = wavenumber(h1, T[i])  # wavenumber at a given frequency and depth

                k1_arr[i] = k1

                Ks = shoalingCoefficient(k0_i, h0, k1, h1)  # compute the shoaling coefficient

                for j in range(len(dirs)):

                    if not valid_theta[j]:
                        # calibration_flag[ch] = 0
                        continue

                    theta0 = theta0_rad[j]

                    # apply Snell's law to get the local wave angle at the DAS channel
                    sin_theta1 = (k0_i / k1) * sin_theta0[j]

                    # AWAC directional spectrum
                    S0 = S_f_theta_eta_selected[i, j]

                    # sin_theta1 = (k0_i / k1) * np.sin(theta0)
                    if np.abs(sin_theta1) > 1:  # no physical solution
                        # calibration_flag[ch] = 0
                        continue

                    theta1 = np.arcsin(sin_theta1)

                    J = np.abs((k1 / k0_i) * (np.cos(theta1) / cos_theta0[
                        j]))  # compute the Jacobian of the transformation from (f, theta0) to (f, theta1) - this is the change in the directional spectrum due to refraction
                    # absolute value is used because the energy flux is always positive, but the Jacobian can be negative for certain angles. The absolute value ensures that we are only considering the magnitude of the change in the directional spectrum.
                    # this will blow up if theta0 is close to 90 degrees - currently skipping those angles

                    # if J > 20:
                    #     print(np.rad2deg(theta0), J, S0)
                    #     J_list.append(J)
                    #     S0_list.append(S0)

                    S1 = S0 * (
                                Ks ** 2) * J  # apply conservation of energy flux to the spectrum - square the coefficients to keep units consistent - energy scales with amplitude squared.

                    theta1_from_deg = np.rad2deg(theta1) + 71.8535 #convert radians to degrees -> convert back to original direction bin reference (true north)

                    #wrap back to 0-360 original direction convention
                    theta1_from_deg %= 360

                    diff = np.abs((dirs - theta1_from_deg + 180) % 360 - 180)
                    j_new = np.argmin(diff)

                    S1_local[
                        i, j_new] += S1  # shape(n_freq, n_dir) - store the local spectrum (at a particular channel) for this frequency and direction

            # apply depth induced dissipation using the breaker index. #TODO: more sophisticated methods can be used later is needed - goda parameterization
            m0_per_freq = np.nansum(S1_local, axis=1) * dtheta  # integrate over direction first -> shape (n_freq,)
            m0 = np.nansum(m0_per_freq * df)  # integrate over frequency to get the total variance of the sea surface elevation spectrum

            Hrms = np.sqrt(
                8 * m0)  # compute the root-mean-square wave height from the variance of the sea surface elevation spectrum

            Hmax = gamma_break * h1  # compute the maximum wave height using the breaker index -> gamma = Hb / hb (wave height at breaking / water depth at breaking)

            if Hrms > Hmax:
                scale = (Hmax / Hrms) ** 2  # compute the scaling factor to apply to the spectrum to account for depth-induced dissipation. Squared because energy scales with amplitude squared -> mo_new = s * mo_old. mo ~ Hrms**2
                S1_local = S1_local * scale  # apply the scaling factor to the spectrum

            directional_spectrum[ch, :, :] = S1_local #save the full directional spectrum

            # convert surface elevation spectrum to pressure spectrum using linear wave theory
            for i, f in enumerate(awac_freqs):
                k1 = k1_arr[i]  # get the local wavenumber for the current frequency

                Kp = 1 / np.cosh(
                    k1 * h1)  # compute the pressure response factor, assume we are trying to simulate pressure at the seabed z=-h

                P1_f = S1_local[i, :] * (
                            rho * g * Kp) ** 2  # convert the sea surface elevation spectrum to a pressure spectrum using linear wave theory

                pressure_spectrum[i] = np.nansum(P1_f) * dtheta  # integrate over the direction bins to get a 1D spectrum

            # compute the calibration coefficient at each channel
            # first integrate both the DAS PSD and AWAC PSD over the desired wave band
            das_wave_band = (das_freqs >= 0.05) & (
                        das_freqs <= 0.3)  # wave band defined over the das PSD frequency bins
            das_psd_variance = np.trapezoid(das_PSD[das_wave_band],
                                            das_freqs[das_wave_band])  # integrated power spectral density of the DAS
            awac_psd_variance = np.trapezoid(
                pressure_spectrum[awac_wave_band],
                awac_freqs[awac_wave_band]
            ) # integrated power spectral density of the AWAC

            if das_psd_variance <= 0:  # if the DAS PSD is zero or negative, huge or erroneous calibration coefficients will be computed.
                calibration_flag[ch] = 0
                continue

            calibration_coefficient = awac_psd_variance / das_psd_variance  # compute the calibration coefficient as the ratio of the integrated AWAC PSD over the integrated DAS PSD
            C[ch] = calibration_coefficient  # store the coefficient for this time and channel

            # # SANITY CHECK
            # # derive Hrms from corrected S1_local
            # m0_local_freq = np.sum(S1_local * dtheta,
            #                        axis=1)  # integrate over direction to get the variance of the sea surface elevation spectrum at frequencies
            # m0_local = np.sum(m0_local_freq * df)
            # Hrms_local = np.sqrt(
            #     8 * m0_local)  # compute the root-mean-square wave height from the variance of the sea surface elevation spectrum
            # m0_list.append(Hrms_local)

        save_dict = {  #FIXME: add das spectrum?
            "calibration_coefficient": C,
            "awac_corrected_directional_spectrum": directional_spectrum,
            'awac_frequency_bins': awac_freqs,
            'awac_directional_bins': dirs,
            "channel_number": Channel_Numbers,
            "water_depth": water_depth,
            "calibration_flag": calibration_flag
        }

        # save the data
        np.savez_compressed(
            f'{save_folder}/Calib{das_start_time.tolist().strftime("_%Y-%m-%d_%H_%M_%S")}.npz',
            save_dict)
        # tqdm.write(f'Saved calibration coefficient for {das_start_time.tolist().strftime("%Y-%m-%d %H:%M:%S")}')

        # # SANITY CHECK
        # plt.plot(np.arange(len(m0_list)), m0_list)
        # plt.xlabel('channel number')
        # plt.ylabel('Hrms (m)')
        # plt.title('Hrms derived from corrected Directional Spectra at each DAS channel')
        # plt.show()
        # plt.close('all')


def process_worker(files_to_process, save_folder, DAS_XYZ_path, AWAC1_path, AWAC2_path=None, gamma_break=0.43):

    '''
    Args:
        AWAC1_path: 300m AWAC path
        AWAC2_path: 600m AWAC path  #TODO: add later - will need to refactor
        DAS_XYZ_path: x, y, depth of the DAS cable datatable path
    '''

    #load data from the 300m awac. Only do once to save computational time
    S_f_theta_eta, awac_freqs, dirs, p_time, dtheta, df, T, \
        gaugeDepth, gaugeElevation, bottomElevation = get_AWAC_data(AWAC1_path)

    channel_grid = pd.read_pickle(DAS_XYZ_path)

    compute_calibration_coefficient(
        files_to_process=files_to_process,
        save_folder=save_folder,
        channel_grid=channel_grid,
        S_f_theta_eta=S_f_theta_eta,
        awac_freqs=awac_freqs,
        dirs=dirs,
        p_time=p_time,
        dtheta=dtheta,
        df=df,
        T=T,
        gaugeDepth=gaugeDepth,
        gaugeElevation=gaugeElevation,
        bottomElevation=bottomElevation,
        gamma_break=gamma_break,
    )




#TODO: check the code for errors. compare the spectra derived wave heights to measured wave heights.
#TODO: burst level calibration - because the AWAC only records pressure spectra for a 34 minute burst (26 minute off)
#TODO: need to add the 600m AWAC in as well!


#TODO: check whether the AWAC is getting its spectra from SUV or AUV
#TODO: wave_frequency, wave energy density, directional wave energy density, from the florence AWAC.



