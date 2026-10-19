import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import re
from utils import load_processedData, get_DAS_PSD, get_DAS_PSD_burst, strain2phase, get_AWAC_data, rebin_psd
from scipy.signal import welch
from tqdm import tqdm
from config import DASConfig


def wavenumber(h, T, tol=1e-6, max_iter=50):
    """
    Solve the linear dispersion relation exactly via Newton-Raphson.
    h: water depth (scalar)
    T: wave period(s), scalar or array
    Returns k with same shape as T.
    """
    #newton-raphson formula is: xn+1 = xn - f(xn)/f'(xn)
    g = 9.81
    h = np.asarray(h, dtype=float) # ensure h is a float array
    sigma = 2 * np.pi / np.asarray(T, dtype=float) # ensure T is a float array and convert to angular frequency
    scalar_out = (h.ndim == 0) and (sigma.ndim == 0) # if both h and T are scalars, return a scalar k
    h = np.where(h > 0, h, np.nan) # avoid negative depths set to NaN
    k = np.broadcast_to(sigma**2 / g, np.broadcast(sigma, h).shape).copy() # initial guess for k based on deep water approximation

    for _ in range(max_iter):
        kh = k * h  # dimensionless relative depth (wavenumber x depth)
        tanh_kh = np.tanh(kh)  # tanh(kh)
        f = g * k * tanh_kh - sigma ** 2  # residual: f(k) = g*k*tanh(kh) - omega^2, should be zero at the true root
        fp = g * tanh_kh + g * kh * (1 - tanh_kh ** 2)  # f'(k): derivative of f w.r.t. k (product rule + chain rule on tanh(kh))
        dk = f / fp  # Newton-Raphson step size: dk = f(k) / f'(k)
        k = k - dk # update k toward the root: k_new = k_old - dk
        if np.max(np.abs(dk)) < tol:
            break

    return float(k) if scalar_out else k

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

def process_worker_burst(burst_chunk, save_folder, DAS_XYZ_path, AWAC1_path, config):
    '''
        Args:
            burst_chunk: list of (burst_start, file_list)
            AWAC1_path: 600m AWAC path - ground truth for calibration
            DAS_XYZ_path: x, y, depth of the DAS cable datatable path
            config: DASConfig object with configuration parameters
        '''

    # load data from the 600m awac. Only do once to save computational time
    S_f_theta_eta, awac_freqs, dirs, time, dtheta, df, T, \
        gaugeDepth, gaugeElevation, bottomElevation = get_AWAC_data(AWAC1_path)

    channel_grid = pd.read_pickle(DAS_XYZ_path)

    propagate_spectrum(
        burst_chunk=burst_chunk,  #chunk of DAS bursts
        save_folder=save_folder,  #folder to save output to
        channel_grid=channel_grid,  #grid of DAS channels with x, y, depth
        S_f_theta_eta=S_f_theta_eta,  #2D directional spectrum from the AWAC
        awac_freqs=awac_freqs, # frequency bins from the AWAC
        dirs=dirs,  #direction bins from the AWAC
        time=time, # time vector from the AWAC
        dtheta=dtheta,
        df=df,
        T=T,
        gaugeDepth=gaugeDepth,
        gaugeElevation=gaugeElevation,
        bottomElevation=bottomElevation,
        config=config
    )


def propagate_spectrum(burst_chunk, save_folder, channel_grid,
                                           S_f_theta_eta, awac_freqs, dirs, time,
                                           dtheta, df, T, gaugeDepth, gaugeElevation,
                                           bottomElevation, config):

    '''Propagate the 2D directional spectrum from the AWAC along the DAS grid using linear wave theory'''

    #get the depth grid along the DAS cable
    bed = channel_grid['z'].values  # depth at all channels along the grid
    n_channels = config.N_channels # number of channels in the DAS

    nf, ndir = len(awac_freqs), len(dirs) #number of frequency and direction bins

    alongshore_channel_start = config.alongshore_channel_start  # approximate first channel of the alongshore leg of the DAS cable
    alongshore_channel_end = config.alongshore_channel_end  # approximate last channel of the alongshore leg of the DAS cable

    awac_wave_band = (awac_freqs >= config.wave_band_min) & (awac_freqs <= config.wave_band_max)  # wave band to bandpass signal

    # trig for wave refraction
    theta0_deg_grid = dirs - config.FRF_orientation  # convert measure wave angle to xFRF shore normal convention for snells law - shore normal pointing offshore. direction bins oriented to FRF
    theta0_deg_grid = (theta0_deg_grid + 180) % 360 - 180  # wrap to -180, 180. #waves propagating in opposite direction of positive x axis. (positive x points offshore, bin 0 is now bin -71.8535)

    theta0_rad_grid = np.deg2rad(theta0_deg_grid)  # convert to radians: shape of (dirs,) - directional bins in radians - no values inside the bins yet
    cos_theta0 = np.cos(theta0_rad_grid)  # precomputation of cos(theta0)
    sin_theta0 = np.sin(theta0_rad_grid)  # precomputation of sin(theta0)

    order = np.argsort(theta0_rad_grid) # sort the directional bins so the linear remap is monotonic
    normal_order = np.argsort(order) # inverse operation for reordering the directional bins back to the original order after remapping
    theta0_rad_sorted = theta0_rad_grid[order]  # sorted directional bins in radians (dirs,)
    cos0_sorted = np.cos(theta0_rad_sorted)[None, :]  # cos of the initial direction bins - sorted (1, ndir)

    refract_dir = (np.abs(theta0_deg_grid[order]) < 80)[None, :]  # (1, ndir) # only consider waves that are within 80 degrees of the shore normal for refraction. avoid singularity in Snells law
    cos1_min = 0.1  # minimum cos(theta1) to avoid singularity in Snell's law. This is a small angle of incidence (theta1 ~ 84 degrees)

    dir_exp = 2  # direction exponent  #TODO: is 2 the correct exponent here? -> initial first guess because energy scales as trig_function^2(theta)
    D_cross = np.abs(cos_theta0) ** dir_exp  # cross-shore cable: along-axis waves produce the greatest response
    D_along = np.abs(sin_theta0) ** dir_exp  # alongshore cable: waves perpendicular to the cable procduce the smallest response

    channel_numbers = np.array(range(n_channels)) + 1  #add one because python uses 0 index. 1-1592 grid

    dtheta_rad = np.mean(np.diff(theta0_rad_sorted)) # average directional bin width in radians - if the bin widths are constant this is the same as dtheta_rad = np.diff(theta0_rad_sorted)[0]
    f_idx = np.repeat(np.arange(nf)[:, None], ndir, axis=1) # frequency index array for the 2D directional spectrum - shape (nf, ndir)

    # burst_chunk is a list of (burst_start, file_list) tuples -- already grouped
    # and validated as complete bursts before this function is ever called
    for b_start, burst_files in tqdm(burst_chunk, colour='yellow'):

        def directional_weight(ch):  # helper function for directional weighting
            # same channel logic as your cross-shore / alongshore branches
            if (ch < alongshore_channel_start) or (ch > alongshore_channel_end):
                return D_cross
            return D_along

        das_start_time = np.datetime64(b_start)  #start time of the burst file

        # burst-level PSD: concatenate raw data across all files in this burst, Welch once
        nperseg = round(config.decimated_sampling_frequency / np.diff(awac_freqs)[0])  # assumes constant spacing of the model frequency bins, computes the nperseg such that PSD bins equal those of the model
        das_freqs, das_PSD_all_channels = get_DAS_PSD_burst(burst_files, nperseg=nperseg, sampling_frequency=config.decimated_sampling_frequency) #calculates PSD and freq bins through welch's method

        time_idx = np.argmin(np.abs(time - das_start_time)) #find the index of the closest time in the AWAC time vector to the DAS burst start time
        awac_start_time = time[time_idx] #get the closest time in the AWAC time vector to the DAS burst start time

        time_mismatch = np.abs(awac_start_time - das_start_time) # compute the mismatch in time between the AWAC and DAS start time
        if time_mismatch > np.timedelta64(5, 'm'): # if the time mismatch is greater than 5 minutes, issue a warning
            tqdm.write(f"Warning: AWAC burst mismatch for DAS burst {b_start}: "
                       f"matched AWAC at {awac_start_time} (offset {time_mismatch})")

        S_f_theta_eta_awac = np.nan_to_num(S_f_theta_eta[time_idx], nan=0.0)  # direction spectrum from the awac at a particular time index - need to filter the valid directions here if using a direction mask
        S_f_theta_eta_awac_sorted = S_f_theta_eta_awac[:, order] # sort the directional bins of the awac spectrum
        # this is an array of shape (62, 72) or (n_frequency bins, n_direction bins)

        #define the water depth at the AWAC for the specific time index
        #Remember these are burst averaged quantities - averaged over 34 minutes
        gaugeDepth0 = gaugeDepth.values[time_idx] # depth from the water surface to the top of the gauge - positive value
        bottomElevation0 = bottomElevation.values[time_idx]# gauge vertical position in NAVD88 - negative value
        gaugeElevation0 = gaugeElevation.values.item() # bottom vertical location from altimiter - negative value

        depth_from_bottom = gaugeElevation0 - bottomElevation0  #float: gives a positive depth up from the bottom
        h0 = gaugeDepth0 + depth_from_bottom  #float: gives the total positive depth at the location of the awac at a particular time ([time_idx] gives the time index)

        surfaceElevation = gaugeElevation0 + gaugeDepth0  #float: gives the surface elevation above or below NAVD88 reference - basically the average tidal signal (+/-) over the 34 minutes

        #generate grid of depths over the das cable
        h_grid = surfaceElevation - bed  #depth grid along the DAS cable, shape(1592,)
        h_grid[config.awac600_channel_location] = h0  #replace the depth at the awac location with the actual depth measured by the awac

        #wavenumber computation
        k0 = wavenumber(h0, T) #compute wavenumber at the AWAC location using dispersion relation. shape(n_freqs)
        k_grid = wavenumber(h_grid[:, None], T) #compute wavenumber at the DAS grid locations using dispersion relation. shape(n_channels, n_freqs)

        #pressure response factor computation
        Kp0 = 1 / np.cosh(k0 * h0) # compute the pressure response factor, assume we are trying to simulate pressure at the seabed z=-h. (shape(n_freqs))
        Kp_grid = 1 / np.cosh(k_grid * h_grid[:, None]) # compute the pressure response factor at the DAS grid locations: (n_channels, n_freqs)

        #group speed computation
        cg0 = groupSpeed(h0, k0) #compute group speed at the AWAC location using dispersion relation. shape(n_freqs)
        cg_grid = groupSpeed(h_grid[:, None], k_grid) #compute group speed at the DAS grid locations using dispersion relation. shape(n_channels, n_freqs)

        #initialize arrays to hold the results for each channel
        calibration_flag = np.ones(config.N_channels, dtype=int) #flag to identify bad calibration channels
        calibration_flag[~(h_grid > 0)] = 0 # flag channels that are above the water surface (h_grid < 0) as bad calibration channels

        C = np.full(config.N_channels, np.nan, dtype=np.float32)  # frequency independent calibration factor
        C_f = np.full((config.N_channels, len(awac_freqs[awac_wave_band])), np.nan, dtype=np.float32)  # frequency dependent calibration factor
        water_depth = np.full(config.N_channels, np.nan) # water depth at each channel location
        S1_directional_spectrum = np.full((config.N_channels, len(awac_freqs), len(dirs)), np.nan, dtype=np.float32) # 2D spectrum at each channel location
        S1_integrated_spectrum = np.full((config.N_channels, len(awac_freqs)), np.nan, dtype=np.float32) # 1D spectrum at each channel location

        # direction weighted spectrum
        S1_weight_spectrum = np.full((config.N_channels, len(awac_freqs)), np.nan, dtype=np.float32) # 1D weighted spectrum at each channel location
        D_weight = np.full(config.N_channels, np.nan, dtype=np.float32) # directional weighting factor at each channel location

        S_corner = None  # spectrum for the corner of the cross and alongshore channels
        cg_corner = None

        for ch in range(config.N_channels):

            if calibration_flag[ch] == 0: # if the channel is flagged as bad, skip it
                continue
            h1 = h_grid[ch] # get the water depth at the current channel location

            das_PSD_raw = das_PSD_all_channels[:, ch]  # pull the DAS PSD at the current channel and time step
            das_PSD = rebin_psd(das_freqs, das_PSD_raw, awac_freqs) # rebin onto the full AWAC grid, then clip both with the same mask

            if ch == config.awac600_channel_location:  #if the channel is the awac channel location
                S1_local = S_f_theta_eta_awac.copy() #take the direction spectrum measured by the awac: shape (n_frequency bins, n_direction bins)
                S1_integrated = np.nansum(S1_local, axis=1) * dtheta #integrate the directional spectrum over all directions to get the frequency spectrum

                water_depth[ch] = h0 #set the water depth equal to water depth at the awac location

                S1_directional_spectrum[ch, :, :] = S1_local  #append to directional spectrum array for later saving
                S1_integrated_spectrum[ch, :] = S1_integrated  #append to integrated spectrum array for later saving

                # compute calibration coefficient at the AWAC location using the unshoaled spectrum
                D = directional_weight(ch)  # get the directional weight for this channel, shape : (n_dirs)
                S1_weight = np.nansum(S1_local * D[None, :], axis=1) * dtheta  # compute the weighted spectrum by integrating over direction with the directional weight, shape : (n_freqs)
                pressure_spectrum = S1_weight * ((config.rho * config.g * Kp0) ** 2)  # convert the sea surface elevation spectrum to a pressure spectrum using linear wave theory, shape : (n_freqs)

                # allocate the effective spectrum and effective depth for later saving
                S1_weight_spectrum[ch, :] = S1_weight # store the weighted spectrum for later saving
                D_weight[ch] = np.nansum(S1_local * D[None, :]) / np.nansum(S1_local) # compute the effective directional weight for this channel and store for later saving

                das_psd_variance = np.trapezoid(das_PSD[awac_wave_band], awac_freqs[awac_wave_band]) # compute the variance of the DAS pressure spectrum over the AWAC wave band
                awac_psd_variance = np.trapezoid(pressure_spectrum[awac_wave_band], awac_freqs[awac_wave_band]) # compute the variance of the AWAC pressure spectrum over the AWAC wave band

                awac_ok = np.isfinite(awac_psd_variance) and awac_psd_variance > 0 # check if the AWAC variance is finite and positive
                if (not np.isfinite(das_psd_variance)) or das_psd_variance <= 0 or not awac_ok: # if it is not:
                    calibration_flag[ch] = 0 # flag the channel as bad calibration
                    continue

                C[ch] = awac_psd_variance / das_psd_variance  # frequency independent calibration
                C_f[ch, :] = pressure_spectrum[awac_wave_band] / das_PSD[awac_wave_band]  # frequency dependent calibration
                continue

            k1 = k_grid[ch, :][:, None]  # wavenumber at the current channel location, shape : (n_freqs, 1)
            cg1 = cg_grid[ch, :][:, None]  # group speed at the current channel location, shape : (n_freqs, 1)
            Ks_squared = cg0[:, None] / cg1    # (nf, 1)  Cg0/Cg1 - shoaling coefficient

            if (ch < alongshore_channel_start) or (ch > alongshore_channel_end): #cross-shore channels

                ky = k0[:, None] * np.sin(theta0_rad_sorted)[None, :]  # (nf, ndir), constant for every channel due to assumption of straight and parallel depth contours
                kx_squared = k1 ** 2 - ky ** 2  #shape (nf, ndir) contains some Nans from where there are dry channels

                with np.errstate(invalid="ignore", divide="ignore"):
                    kx = np.sqrt(np.where(kx_squared > 0, kx_squared, np.nan))  # positive root: refracting bins have cos(theta0) > 0
                    cos1 = kx / k1 #recover the cos of the new direction at the current channel location
                    theta1 = np.arctan2(ky, kx) # compute the new direction at the current channel location

                theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi # wrap the new direction to be between -pi and pi: shape(nf, ndir)

                ok = refract_dir & np.isfinite(cos1) & (cos1 >= cos1_min) # mask for valid refracted directions: shape(nf, ndir)

                with np.errstate(invalid="ignore", divide="ignore"): # ignore invalid and divide by zero warnings for the following calculations
                    #conserve energy flux only where the refracted direction is valid, otherwise set to zero
                    deposit = np.where(ok, S_f_theta_eta_awac_sorted * Ks_squared * cos0_sorted / cos1, 0.0) # shape(nf, ndir)  # energy flux conservation: S1 = S0 * Ks^2 * Kr^2

                # Spread each refracted component over the two nearest direction bins. - weighted by the position of the refracted direction within the directional grid.
                pos = np.where(ok, (theta1 - theta0_rad_sorted[0]) / dtheta_rad, 0.0)#want to find new directions position in the directional grid - how many bins from the first bin have we gone
                j0 = np.floor(pos).astype(int)  # rounds down to the nearest integer to find the index of the lower direction bin
                frac = pos - j0 #finds the fractional part of the position ex: if position = 2.3, j0 = 2, frac = 0.3
                j0 = j0 % ndir # wraps the index of the lower direction bin to ensure it is within the bounds of the directional grid - applies modulus
                j1 = (j0 + 1) % ndir # finds the index of the upper direction bin by adding 1 to the index of the lower direction bin and wrapping it to ensure it is within the bounds

                S1_sorted = np.zeros((nf, ndir)) # initialize the refracted spectrum at the current channel location to zero
                # np.add.at allows energy accumulation in the same bin like += operator
                np.add.at(S1_sorted, (f_idx, j0), deposit * (1 - frac)) # add the contribution of the refracted spectrum to the lower direction bin, weighted by the fractional part of the position
                np.add.at(S1_sorted, (f_idx, j1), deposit * frac) # add the contribution of the refracted spectrum to the upper direction bin, weighted by the fractional part of the position

                S1_sorted += np.where(ok, 0.0, S_f_theta_eta_awac_sorted * Ks_squared)  # shoaling only, same direction

                S1_local = S1_sorted[:, normal_order]  # unsort the directions to match the original order of theta0_rad. shape (nf, ndir)

            else:  # alongshore channels
                if S_corner is None:  # corner was dry or flagged
                    calibration_flag[ch] = 0
                    continue
                S1_local = S_corner * (cg_corner / cg1)  # re-shoal from corner, no refraction; (nf,ndir)*(nf,1)

            # everything below runs for both branches
            S1_integrated = np.nansum(S1_local, axis=1) * dtheta
            water_depth[ch] = h1

            # depth-limited breaking cap
            m0 = np.nansum(S1_integrated * df)
            Hrms = np.sqrt(8 * m0)
            Hmax = config.gamma * h1
            if Hrms > Hmax:
                scale = (Hmax / Hrms) ** 2
                S1_local = S1_local * scale
                S1_integrated = S1_integrated * scale

            if ch == alongshore_channel_start - 1:  # corner channel
                S_corner = S1_local.copy()
                cg_corner = cg1.copy()  # (nf, 1)

            S1_directional_spectrum[ch] = S1_local
            S1_integrated_spectrum[ch] = S1_integrated

            # cable weighting and pressure
            D = directional_weight(ch)
            S1_weight = np.nansum(S1_local * D[None, :], axis=1) * dtheta
            pressure_spectrum = S1_weight * (config.rho * config.g * Kp_grid[ch]) ** 2
            S1_weight_spectrum[ch] = S1_weight
            D_weight[ch] = np.nansum(S1_local * D[None, :]) / np.nansum(S1_local)

            # calibration
            das_psd_variance = np.trapezoid(das_PSD[awac_wave_band], awac_freqs[awac_wave_band])
            awac_psd_variance = np.trapezoid(pressure_spectrum[awac_wave_band], awac_freqs[awac_wave_band])
            if not (np.isfinite(das_psd_variance) and das_psd_variance > 0
                    and np.isfinite(awac_psd_variance) and awac_psd_variance > 0):
                calibration_flag[ch] = 0
                continue

            C[ch] = awac_psd_variance / das_psd_variance
            with np.errstate(divide="ignore", invalid="ignore"):
                C_f[ch, :] = pressure_spectrum[awac_wave_band] / das_PSD[awac_wave_band]

        # back at the burst-loop level, after the channel loop
        save_dict = {
            "calibration_coefficient": C,
            "calibration_coefficient_freq_dependent": C_f,
            "awac_corrected_directional_spectrum": S1_directional_spectrum,
            "awac_corrected_frequency_spectrum": S1_integrated_spectrum,
            "awac_cable_weighted_frequency_spectrum": S1_weight_spectrum,
            "directional_weight_effective": D_weight,
            "awac_frequency_bins": awac_freqs,
            "awac_directional_bins": dirs,
            "channel_number": channel_numbers,
            "water_depth": water_depth,
            "calibration_flag": calibration_flag,
        }
        np.savez_compressed(
            f'{save_folder}/Calib{b_start.strftime("_%Y-%m-%d_%H_%M_%S")}.npz',
            **save_dict,
        )











