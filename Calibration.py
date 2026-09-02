import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import re
from utils import load_processedData, get_DAS_PSD, get_DAS_PSD_burst, strain2phase, get_AWAC_data
from scipy.signal import welch
from tqdm import tqdm

import matplotlib.pyplot as plt

def wavenumber(h, T, tol=1e-6, max_iter=50):
    """
    Solve the linear dispersion relation exactly via Newton-Raphson.
    h: water depth (scalar)
    T: wave period(s), scalar or array
    Returns k with same shape as T.
    """
    #newton-raphson formula is: xn+1 = xn - f(xn)/f'(xn)

    g = 9.81
    T = np.atleast_1d(T).astype(float)
    omega = 2 * np.pi / T
    k = omega**2 / g  # deep-water guess as starting point

    for _ in range(max_iter):
        kh = k * h  # dimensionless relative depth (wavenumber x depth)
        tanh_kh = np.tanh(kh)  # tanh(kh)
        f = g * k * tanh_kh - omega ** 2  # residual: f(k) = g*k*tanh(kh) - omega^2, should be zero at the true root
        fp = g * tanh_kh + g * kh * (1 - tanh_kh ** 2)  # f'(k): derivative of f w.r.t. k (product rule + chain rule on tanh(kh))
        dk = f / fp  # Newton-Raphson step size: dk = f(k) / f'(k)
        k -= dk  # update k toward the root: k_new = k_old - dk
        if np.max(np.abs(dk)) < tol:
            break

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

def get_burst_start(file_path):
    match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})', file_path)
    if not match:
        return None
    ts = np.datetime64(datetime.strptime(match.group(1), '%Y_%m_%d_%H_%M_%S'))
    # floor to the top of the hour
    burst_start = ts.astype('datetime64[h]')
    return burst_start

def parse_das_timestamp(file_path):
    """Single source of truth for parsing DAS filename timestamps."""
    match = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})', file_path)
    if not match:
        return None
    return datetime.strptime(match.group(1), '%Y_%m_%d_%H_%M_%S')




def compute_calibration_coefficient(files_to_process,
        save_folder,
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


def compute_calibration_coefficient_burst(burst_chunk, save_folder, channel_grid,
                                           S_f_theta_eta, awac_freqs, dirs, p_time,
                                           dtheta, df, T, gaugeDepth, gaugeElevation,
                                           bottomElevation, gamma_break):

    rho = 1025
    g = 9.81
    awac_channel = 402 #xfrf 300m awac channel location 208 - approximate location. 600m awac is located at channel 402
    alongshore_channel_start = 501 #channel that the alongshore leg should approximately start at
    alongshore_channel_end = 1091 #channel that the alongshore leg should approximately end at

    # get the depth across all channels
    bed = channel_grid['z'].values  # depth at all channels along the grid
    N_channels = len(bed)  # number of channels in the DAS

    awac_wave_band = (awac_freqs >= 0.05) & (awac_freqs <= 0.3)  # wave band to integrate over

    # precompute trig functions for wave refraction
    theta0_deg = dirs - 71.8535  # convert measure wave angle to xFRF shore normal convention for snells law - shore normal pointing offshore
    theta0_deg = (theta0_deg + 180) % 360 - 180  # wrap to -180, 180. #waves propagating in opposite direction of positive x axis

    theta0_rad = np.deg2rad(theta0_deg)  # convert to radians
    cos_theta0 = np.cos(theta0_rad)  # precomputation of cos(theta0)
    sin_theta0 = np.sin(theta0_rad)  # precomputation of sin(theta0)

    Channel_Numbers = np.array(range(N_channels)) + 1

    # burst_chunk is a list of (burst_start, file_list) tuples -- already grouped
    # and validated as complete bursts before this function is ever called
    for b_start, burst_files in tqdm(burst_chunk, colour='yellow'):

        das_start_time = np.datetime64(b_start)

        # burst-level PSD: concatenate raw data across all files in this burst, Welch once
        das_freqs, das_PSD_all_channels = get_DAS_PSD_burst(burst_files, nperseg=256, sampling_frequency=2)

        idx = np.argmin(np.abs(p_time - das_start_time))
        awac_start_time = p_time[idx]

        time_mismatch = np.abs(awac_start_time - das_start_time)
        if time_mismatch > np.timedelta64(5, 'm'):
            tqdm.write(f"Warning: AWAC burst mismatch for DAS burst {b_start}: "
                       f"matched AWAC at {awac_start_time} (offset {time_mismatch})")

        S_f_theta_eta_selected = S_f_theta_eta[idx] #direction spectrum from the awac at a particular time index - need to filter the valid directions here if using a direction mask
        #this is an array of shape (62, 72) or (n_frequency bins, n_direction bins)

        gaugeDepth0 = gaugeDepth.values[idx] # depth from the water surface to the top of the gauge - positive value
        bottomElevation0 = bottomElevation.values[idx]# gauge vertical position in NAVD88 - negative value
        gaugeElevation0 = gaugeElevation.values.item() # bottom vertical location from altimiter - negative value

        depth_from_bottom = gaugeElevation0 - bottomElevation0 #gives a positive depth up from the bottom
        h0 = gaugeDepth0 + depth_from_bottom #gives the total positive depth at the location of the awac at a particular time ([idx] gives the time index)

        surfaceElevation = gaugeElevation0 + gaugeDepth0 #gives the surface elevation above or below NAVD88 reference

        k0 = wavenumber(h0, T)

        calibration_flag = np.ones(N_channels, dtype=int)
        C = np.full(N_channels, np.nan, dtype=np.float32) #frequency independent calibration factor
        C_f = np.full((N_channels, len(awac_freqs[awac_wave_band])), np.nan, dtype=np.float32) #frequency dependent calibration factor
        water_depth = np.full(N_channels, np.nan)
        S1_directional_spectrum = np.full((N_channels, len(awac_freqs), len(dirs)), np.nan, dtype=np.float32)
        S1_integrated_spectrum = np.full((N_channels, len(awac_freqs)), np.nan, dtype=np.float32)

        S_corner = None #spectrum for the corner of the cross and alongshore channels
        k_corner = None
        h1_corner = None

        valid_channels = 0

        for ch in range(N_channels):

            # pressure_spectrum = np.zeros(len(awac_freqs), dtype=np.float32)  # empty array to hold pressure spectrum - if desired

            # pull the DAS PSD at the current channel and time step
            das_PSD = das_PSD_all_channels[:, ch]

            if ch == awac_channel:  #if the channel is the awac channel location
                S1_local = S_f_theta_eta_selected.copy() #take the direction spectrum measured by the awac

                water_depth[ch] = h0 #set the water depth equal to water depth at the awac location
                valid_channels += 1  #channle is valid

                S1_directional_spectrum[ch, :, :] = S1_local  #append to directional spectrum array for later saving
                S1_integrated_spectrum[ch, :] = (np.nansum(S1_local, axis=1) * dtheta)

                calibration_flag[ch] = 0 #set calibration flag to 0 because this is not a shoaled spectrum
                continue

            h1 = surfaceElevation - bed[ch]  # depth at the channel location. bed[ch] is a negative number. subtracting a negative number gives a positive water depth

            # h1_list.append(h1)

            if h1 <= 0:  # h1 must be positive for dispersion relation (wavenumber calculation) can use 1
                calibration_flag[ch] = 0
                continue

            water_depth[ch] = h1  # append the water_depth for each channel

            k1_arr = wavenumber(h1, T)  # wavenumber for all frequencies and depths

            valid_channels += 1  # add a count for each channel that is allowed to pass through

            S1_local = np.zeros((len(awac_freqs), len(dirs)), dtype=np.float32) #empty array to hold to 2D wave spectrum
            S1_integrated = np.zeros((len(awac_freqs)),  dtype=np.float32) #empty array to hold the 1D wave spectrum

            # S0_integrated_plot = np.zeros(len(awa c_freqs)) #DEBUG ONLY
            # S1_integrated_plot = np.zeros(len(awac_freqs)) #DEBUG ONLY

            if (ch < alongshore_channel_start) or (ch > alongshore_channel_end): #cross-shore channels
                for i, f in enumerate(awac_freqs):

                    k0_i = k0[i]  # wavenumber at the AWAC at time t (different depth over time) and frequency f

                    # get the local wavenumber
                    k1 = k1_arr[i]

                    Ks = shoalingCoefficient(k0_i, h0, k1, h1)  # compute the shoaling coefficient

                    for j in range(len(dirs)):

                        # Snell's law: refract each directional wave component
                        # theta0 is the offshore/AWAC direction and theta1 is the local direction.
                        # At fixed frequency, C1/C0 = k0/k1.
                        sin_theta1 = (k0_i / k1) * sin_theta0[j]

                        # skip directions that cannot refract to this depth
                        # skip waves that are too oblique to xFRF. Causes the jacobian to converge to a singularity - very large values directional bins that shouldn't have a lot of energy
                        if np.abs(sin_theta1) > 1 or np.abs(theta0_deg[j]) >= 80:
                            # Retain the original direction and apply shoaling only.
                            # This prevents energy from being discarded when a directional
                            # component cannot be represented by the refraction calculation.
                            S0 = S_f_theta_eta_selected[i, j]
                            S1 = S0 * (Ks ** 2)
                            S1_local[i, j] += S1
                            S1_integrated[i] += S1 * dtheta
                            continue

                        # local direction
                        theta1 = np.arcsin(sin_theta1)

                        # Directional Jacobian:
                        # J = d(theta0) / d(theta1)
                        # This converts the source directional density to the local
                        # directional density after refraction.
                        J = np.abs((k1 / k0_i) * (np.cos(theta1) / cos_theta0[j]))

                        # original directional spectrum
                        S0 = S_f_theta_eta_selected[i, j]

                        # Continuous local directional spectral density.
                        # Shoaling conserves ray energy flux:
                        #     E0 * Cg0 = E1 * Cg1
                        # so the shoaling factor is Cg0/Cg1 = Ks^2.
                        # The Jacobian accounts for compression/stretching of the
                        # directional coordinate caused by refraction.
                        S1_density = S0 * (Ks ** 2) * J
                        #TODO: do I need a k1/k0 ratio here (from Cg cos(theta))? This depends on what energy flux is being conserved.
                        # if we are only conserving the component of energy flux normal to the cross shore cable then yes
                        # If we are conserving the energy flux along a wave ray then no

                        # Integrate the continuous local spectrum over the local
                        # directional coordinate:
                        #
                        #     dtheta1 = dtheta0 / J
                        #
                        # Therefore:
                        #     S1_density * dtheta1
                        #       = S0 * Ks^2 * J * (dtheta/J)
                        #       = S0 * Ks^2 * dtheta
                        #
                        # This conserves the energy associated with the individual ray.
                        S1_integrated[i] += S1_density * (dtheta / J)  #the dtheta / J is redundant here but is included for mathematical clarity

                        # # DEBUG ONLY: AWAC spectrum retained before propagation
                        # S0_integrated_plot[i] += S0
                        # # propagated spectrum - DEBUG ONLY
                        # S1_integrated_plot[i] += S1 * (dtheta / J)  #same here

                        # Convert the refracted direction back to the direction convention
                        # used by the AWAC/DAS directional grid.
                        # convert radians to degrees -> convert back to original direction bin reference (true north), #wrap back to 0-360 original direction convention
                        theta1_from_deg = (np.rad2deg(theta1) + 71.8535) % 360

                        # Find the destination direction bin on the fixed direction grid.
                        diff = np.abs((dirs - theta1_from_deg + 180) % 360 - 180)  #find the nearest direction bin
                        j_new = np.argmin(diff)

                        # Redistribute the transformed ray energy onto the fixed-width
                        # numerical direction grid.
                        #
                        # S1_density is the continuous local density and contains the
                        # Jacobian. Because the destination grid has fixed dtheta,
                        # divide by J here to convert the local density back to the
                        # energy represented by the original directional bin:
                        #
                        #     S1_density * (dtheta/J)
                        #       = S0 * Ks^2 * dtheta
                        #
                        # Thus the deposited spectral density is S0 * Ks^2 for equal
                        # source and destination bin widths.
                        S1_local[i, j_new] += S1_density / J

                        # #sanity check - 2d wave spectrum and 1d wave spectrum shoudl integrate to the same value, absolute difference should be near zero
                        # S1_integrated_from_2D = (np.nansum(S1_local, axis=1) * dtheta)
                        # difference = (S1_integrated_from_2D - S1_integrated)
                        # print("Max absolute difference:", np.nanmax(np.abs(difference)))

            else: #alongshore channels
                #For the alongshore channels stop the conservation of energy flux, the alongshore leg is not a continuation of the wave propagation path.
                if S_corner is None: #check to make sure the corner spectrum is created
                    raise RuntimeError("Corner spectrum was not created before reaching the alongshore leg.")

                k1_arr_along = wavenumber(h1, T)  # local wavenumber at this alongshore channel's own depth
                Ks_along = shoalingCoefficient(k_corner, h1_corner, k1_arr_along, h1)  # corner to local depth
                S1_local = S_corner * (Ks_along ** 2)[:, None]  # broadcast over direction bins
                S1_integrated = np.nansum(S1_local, axis=1) * dtheta
                # S1_local = S_corner.copy()  #alongshore channel spectra should just be the copy of the corner channel spectrum
                # S1_integrated = np.nansum(S1_local, axis=1) * dtheta  #directly integrated version

            # convert directional sum to directional integral
            # S0_integrated_plot *= dtheta #debug only - this does not need to happen for

            #SANITY CHECK: to make sure energy flux is conserved between S0 and S1
            # if ch==180:
            #     Cg0 = groupSpeed(h0, k0)
            #     Cg1 = groupSpeed(h1, k1_arr)
            #     F0 = rho * g * Cg0 * S0_integrated_plot
            #     F1 = rho * g * Cg1 * S1_integrated_plot
            #     ratio = F1 / F0
            #     plt.figure(figsize=(10, 6))
            #     plt.plot(awac_freqs, ratio, color='green', linewidth=5)
            #     plt.axhline(1, linestyle='--', color='black')
            #     plt.ylim(0, 2)
            #     plt.xlabel('Frequency (Hz)')
            #     plt.ylabel(r'$F_1(f)/F_0(f)$')
            #     plt.title('Energy Flux Conservation Check')
            #     plt.grid()
            #     plt.show()
            #
            #cross-check: does the redistributed directional spectrum
            # integrate (over dirs, at fixed dtheta) to the same total as the scalar path
            # S1_local_integrated_check = np.nansum(S1_local, axis=1) * dtheta
            # print("max abs diff, scalar vs redistributed:",
            #       np.nanmax(np.abs(S1_local_integrated_check - S1_integrated)))

            # apply depth induced dissipation using the breaker index. #TODO: more sophisticated methods can be used later is needed - goda parameterization
            m0_per_freq = S1_integrated
            m0 = np.nansum(m0_per_freq * df)  # integrate over frequency to get the total variance of the sea surface elevation spectrum

            Hrms = np.sqrt(8 * m0)  # compute the root-mean-square wave height from the variance of the sea surface elevation spectrum

            Hmax = gamma_break * h1  # compute the maximum wave height using the breaker index -> gamma = Hb / hb (wave height at breaking / water depth at breaking)

            if Hrms > Hmax:
                # compute the scaling factor to apply to the spectrum to account for depth-induced dissipation. Squared because energy scales with amplitude squared -> mo_new = s * mo_old. mo ~ Hrms**2
                scale = (Hmax / Hrms) ** 2
                S1_integrated *= scale #apply scaling factor to the spectrum
                S1_local *= scale  # apply the scaling factor to the spectrum

            if ch == alongshore_channel_start - 1:  #if we are at the corner channel
                h1_corner = h1 #save the water depth at the corner channel for later use in alongshore channels
                k_corner = k1_arr.copy() #save the wavenumber at the corner channel for later use in alongshore channels
                S_corner = S1_local.copy()  #save the last spectrum to copy along the alongshore in the future

            S1_directional_spectrum[ch, :, :] = S1_local  # save the full directional spectrum
            S1_integrated_spectrum[ch, :] = S1_integrated #save the integrated frequency spectrum

            # convert surface elevation spectrum to pressure spectrum using linear wave theory
            Kp = 1 / np.cosh(k1_arr * h1) # compute the pressure response factor, assume we are trying to simulate pressure at the seabed z=-h

            pressure_spectrum = S1_integrated * (rho * g * Kp) ** 2 # convert the sea surface elevation spectrum to a pressure spectrum using linear wave theory, shape : (n_freqs)

            # compute the calibration coefficient at each channel
            # first integrate both the DAS PSD and AWAC PSD over the desired wave band
            das_wave_band = (das_freqs >= 0.05) & (das_freqs <= 0.3)  # wave band defined over the das PSD frequency bins

            das_psd_variance = np.trapezoid(das_PSD[das_wave_band], das_freqs[das_wave_band])  # integrated power spectral density of the DAS

            awac_psd_variance = np.trapezoid(pressure_spectrum[awac_wave_band], awac_freqs[awac_wave_band])  # integrated power spectral density of the AWAC

            if das_psd_variance <= 0:  # if the DAS PSD is zero or negative, huge or erroneous calibration coefficients will be computed.
                calibration_flag[ch] = 0
                continue

            #frequency independent calibration
            calibration_coefficient = awac_psd_variance / das_psd_variance  # compute the calibration coefficient as the ratio of the integrated AWAC PSD over the integrated DAS PSD
            C[ch] = calibration_coefficient  # store the coefficient for this time and channel

            #frequency dependent calibration
            # interpolate the DAS PSD onto the AWAC frequency grid, original DAS PSD is on a different frequency grid with a small mismatch
            das_PSD_on_awac_grid = np.interp(awac_freqs, das_freqs, das_PSD)  #FIXME: interpolation could possible introduce minor error
            calib_coeff_freq_dependent = pressure_spectrum[awac_wave_band] / das_PSD_on_awac_grid[awac_wave_band] # compute the frequency-dependent calibration coefficient as the ratio of the AWAC pressure spectrum over the DAS PSD
            C_f[ch, :] = calib_coeff_freq_dependent #store the frequency-dependent calibration coefficient for this time and channel, vectorized for all frequencies

            # # SANITY CHECK
            # # derive Hrms from corrected S1_local
            # m0_local_freq = np.sum(S1_local * dtheta,
            #                        axis=1)  # integrate over direction to get the variance of the sea surface elevation spectrum at frequencies
            # m0_local = np.sum(m0_local_freq * df)
            # Hrms_local = np.sqrt(
            #     8 * m0_local)  # compute the root-mean-square wave height from the variance of the sea surface elevation spectrum
            # m0_list.append(Hrms_local)

        save_dict = {
            "calibration_coefficient": C,
            "calibration_coefficient_freq_dependent": C_f,
            "awac_corrected_directional_spectrum": S1_directional_spectrum,
            "awac_corrected_frequency_spectrum": S1_integrated_spectrum,
            'awac_frequency_bins': awac_freqs,
            'awac_directional_bins': dirs,
            "channel_number": Channel_Numbers,
            "water_depth": water_depth,
            "calibration_flag": calibration_flag
        }

        np.savez_compressed(
            f'{save_folder}/Calib{b_start.strftime("_%Y-%m-%d_%H_%M_%S")}.npz',
            save_dict
        )

        # # SANITY CHECK
        # plt.plot(np.arange(len(m0_list)), m0_list)
        # plt.xlabel('channel number')
        # plt.ylabel('Hrms (m)')
        # plt.title('Hrms derived from corrected Directional Spectra at each DAS channel')
        # plt.show()
        # plt.close('all')



def process_worker_burst(burst_chunk, save_folder, DAS_XYZ_path, AWAC1_path, AWAC2_path=None, gamma_break=0.43):
    '''
        Args:
            burst_chunk: list of (burst_start, file_list)
            AWAC1_path: 300m AWAC path
            AWAC2_path: 600m AWAC path  #TODO: add later - will need to refactor
            DAS_XYZ_path: x, y, depth of the DAS cable datatable path
        '''

    # load data from the 300m awac. Only do once to save computational time
    S_f_theta_eta, awac_freqs, dirs, p_time, dtheta, df, T, \
        gaugeDepth, gaugeElevation, bottomElevation = get_AWAC_data(AWAC1_path)

    channel_grid = pd.read_pickle(DAS_XYZ_path)

    compute_calibration_coefficient_burst(
        burst_chunk=burst_chunk,
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


#TODO: need to add the 600m AWAC in as well!


#TODO: check whether the AWAC is getting its spectra from SUV or AUV
#TODO: wave_frequency, wave energy density, directional wave energy density, from the florence AWAC.



