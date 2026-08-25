import numpy as np
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import re
import os
import json
import h5py
from scipy.signal import welch
from scipy.interpolate import interp1d



def sequence_number(path):
    filename = os.path.basename(path) #pull filename

    #downsampled format
    match = re.search(r'(\d{4}\D+\d{2}\D+\d{2}\D+\d{2}\D+\d{2}\D+\d{2})', filename) #looks for a string of dates - regex. '\D+' matches any character
    if match:
        timestamp = re.sub(r'\D+', '', match.group())
        return datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        # return datetime.strptime(match.group(1), "%Y-%m-%d_%H.%M.%S")

    #old raw format
    match = re.search(r'_(\d+)\.h5$', filename) #looks for a string of the form "_<number>.h5" at the end of the filename
    if match:
        return int(match.group(1)) #return the number as an integer

    raise ValueError(f"Filename format not recognized: {filename}")


def read_Json(filepath):

    def print_json(obj, indent=0):
        prefix = "    " * indent  #resursively read all data in the json file

        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    print(f"{prefix}{key}:")
                    print_json(value, indent + 1)
                else:
                    print(f"{prefix}{key}: {value}")

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    print(f"{prefix}[{i}]")
                    print_json(item, indent + 1)
                else:
                    print(f"{prefix}[{i}]: {item}")

    with open(filepath, 'r') as f:  #open input file
        data = json.load(f)  #load data

    print_json(data)  #print


def clip_by_time(file_paths, start_time: datetime, end_time: datetime):
    # clip the number of files that are present based on a starting and ending stime

    clipped_file_paths = [
        f for f in file_paths
        if start_time <= sequence_number(f) <= end_time
    ]
    return clipped_file_paths


def chunk_list(lst, n):
    """Split a list into n roughly equal contiguous chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def time2dateTime(time_array):

    #convert array of unix timestamps to datetime objects
    if not isinstance(time_array, np.ndarray):
        dateTime_array = datetime.fromtimestamp(time_array, tz=timezone.utc)

    else:
        dateTime_array = np.array([datetime.fromtimestamp(t, tz=timezone.utc) for t in time_array])

    return dateTime_array


def read_dasFile(file_path:str):

    data_vars = {}
    with h5py.File(file_path) as f:

        # keys = list(f.keys()) #list all keys

        if "Acquisition" in f:
            raw = f['Acquisition']['Raw[0]']  #headers in h5 structure

        else:
            raw = f  #already decimated files without the header

        #read through each key and save as a variable to the dictionary
        for key in raw.keys():
            data_vars[key] = np.array(raw[key])

    return data_vars


def print_all_attributes(name, obj):
    if obj.attrs:
        print(f"\nAttributes for [{name}]:")
        for key, value in obj.attrs.items():
            print(f"  {key}: {value}")


def load_processedData(processed_data_path):

    with h5py.File(processed_data_path, 'r') as f:  #open hdf5 file and read the strain and time datasets
        strain = f['strain'][:]  # shape (time, N_channels)
        time = f['time'][:]  # shape (time,)

    return strain, time


def phase2strain(phase_shifts, gauge_length=1.6):
    #use equations from Hartog 2017. Use linear theory to convert phase shifts to axial strain (Exx(t, x))
    L = 1550e-9  #wavelength of light in meters
    psi = 0.79  #pockel's coefficient for single mode glass fiber   #FIXME: This is outdated. keep things in phase shifts in the future
    nc = 1.46  #index of refraction for the optical fiber

    # strain = (L / (4 * np.pi * nc * gauge_length * psi)) * phase_shifts
    coeff = np.float32(L / (4 * np.pi * nc * gauge_length * psi)) #cut to float32 to save memory
    phase_shifts *= coeff  #multiply in place to avoid the creation of a second large array

    return phase_shifts


def strain2phase(strain, gauge_length=1.6):
    #invert the phase2strain function to convert strain back to phase shifts
    L = 1550e-9  #wavelength of light in meters
    psi = 0.79  #pockel's coefficient for single mode glass fiber
    nc = 1.46  #index of refraction for the optical fiber

    coeff = np.float32(((4 * np.pi) * nc * gauge_length * psi) / L)  #cut to float32 to save memory
    strain *= coeff  #multiply in place to avoid the creation of a second large array

    return strain


def get_DAS_PSD(file_path, nperseg=256, noverlap=None, sampling_frequency=2):

    #read file
    strain, time = load_processedData(file_path)

    #convert strain to phase - original strain to phase equation is not suitable for this calibration
    phase = strain2phase(strain)

    if noverlap is None:
        noverlap = nperseg // 2

    freqs, PSD = welch(phase, fs=sampling_frequency, nperseg=nperseg, noverlap=noverlap, axis=0) #compute PSD for each channel, returns PSD of shape (n_freqs, n_channels)

    return freqs, PSD


def get_DAS_PSD_burst(file_list, nperseg=256, sampling_frequency=2):
    """
    Concatenate raw DAS strain data across all files in a burst,
    then compute a single Welch PSD over the full burst -- matches
    how the AWAC's own 35-min burst PSD is computed.
    """
    raw_phase = []
    for f in file_list:
        strain, time = load_processedData(f)
        phase = strain2phase(strain)
        raw_phase.append(phase)

    raw_concat = np.concatenate(raw_phase, axis=0)  # concatenate along the time axis
    freqs, PSD = welch(raw_concat, fs=sampling_frequency, nperseg=nperseg, axis=0)
    return freqs, PSD


def get_burst_files(all_paths, start_time, burst_duration_min=35):
    """Pull all files whose timestamp falls within [start_time, start_time + burst_duration) """
    burst_end = start_time + timedelta(minutes=burst_duration_min)
    burst_files = [
        f for f in all_paths
        if start_time <= sequence_number(f) < burst_end
    ]
    return sorted(burst_files, key=sequence_number)


def get_orientation(channel):
    if channel >= 500 and channel <= 1092:
        orientation = 'alongshore'
    else:
        orientation = 'cross_shore'

    return orientation


def ncsp2FRF(p1, p2):
    """this function converts nc StatePlane (3200 fips) to FRF coordinates
    based on kent Hathaways Code
    #
    #  15 Dec 2014
    #  Kent Hathaway.
    #  Translated from Matlab to python 2015-11-30 - Spicer Bak
    #
    #  Uses new fit (angles and scales) Bill Birkemeier determined in Nov 2014
    #
    #  This version will determine the input based on values, outputs FRF, lat/lon,
    #  and state plane coordinates.  Uses NAD83-2011.
    #
    #  IO:
    #  p1 = FRF X (m), or Longitude (deg + or -), or state plane Easting (m)
    #  p2 = FRF Y (m), or Latitude (deg), or state plane Northing (m)
    #
    #  X = FRF cross-shore (m)
    #  Y = FRF longshore (m)
    #  ALat = latitude (decimal degrees)
    #  ALon = longitude (decimal degrees, positive, or W)
    #  spN = state plane northing (m)
    #  spE = state plane easting (m)

    NAD83-86	2014
    Origin Latitude       36.1775975
    Origin Longitude      75.7496860
    m/degLat              110963.357
    m/degLon               89953.364
    GridAngle (deg)          18.1465
    Angle FRF to Lat/Lon     71.8535
    Angle FRF to State Grid  69.9747
    FRF Origin Northing  274093.1562
    Easting              901951.6805

    #  Debugging values
    p1=566.93;  p2=515.11;  % south rail at 1860
    ALat = 36.1836000
    ALon = 75.7454804
    p2= 36.18359977;
    p1=-75.74548109;
    SP:  p1 = 902307.92;
    p2 = 274771.22;

    Args:
      spE: North carolina state plane coordinate system - Easting
      spN: North carolina state plane coordinate system - Northing
      p1: first point
      p2: second point

    Returns:
      dictionary
       'xFRF': cross shore location in FRF coordinates

       'yFRF': alongshore location in FRF coodrindate system

       'StateplaneE': north carolina state plane coordinate system - easting

       'StateplaneN': north carolina state plane coordinate system - northing

    """
    r2d = 180.0 / np.pi;
    Eom = 901951.6805;  # % E Origin State Plane
    Nom = 274093.1562;  # % N Origin State Plane
    spAngle = (90 - 69.974707831) / r2d

    spE = p1
    spN = p2  # designating stateplane vars

    # to FRF coords
    spLengE = p1 - Eom
    spLengN = p2 - Nom
    R = np.sqrt(spLengE ** 2 + spLengN ** 2)
    Ang1 = np.arctan2(spLengE, spLengN)
    Ang2 = Ang1 + spAngle
    # to FRF
    X = R * np.sin(Ang2)
    Y = R * np.cos(Ang2)
    # to Lat Lon
    ans = {'xFRF': X,
           'yFRF': Y,
           'StateplaneE': spE,
           'StateplaneN': spN}
    return ans


def build_DAS_XYZ_grid():
    #TODO: refine this

    # find the DAS channel closest to the awac and get the PSD to a 34 (35 for DAS) minute burst (average)
    # channel = 188 #around 300m out on the cross-shore
    # channel = 375 #aroudn 600m out on the cross-shore

    #Number of channels
    N_channels = 1592

    #length of grid in meters
    grid_len = (N_channels * 1.6) // 2  #divide by two because the cable wraps back around on itself.

    #find where the cable lays in x FRF to match it with a depth from the larc survey.
    larc_data_path = './data/frf_awac/FRF_geomorphology_elevationTransects_survey_20260611.nc'
    larc_data = xr.open_dataset(larc_data_path)
    larc_x = larc_data.xFRF.values
    larc_y = larc_data.yFRF.values
    profile_number = larc_data.profileNumber.values
    time = larc_data.elevation.coords['time'].values
    elevation = larc_data.elevation.values  #depth profile

    #plot profile
    # plt.scatter(larc_x, larc_y, c=elevation)
    # plt.show()
    # plt.close('all')

    #pull just the bathymetry data along the yFRF=1000m transect
    cross_shore_profile_mask = (profile_number==1006)
    masked_x = larc_x[cross_shore_profile_mask]
    masked_y = larc_y[cross_shore_profile_mask]
    masked_elevation = elevation[cross_shore_profile_mask]

    #pull just the bathymetry data along the xFRF = 778
    profiles = np.unique(profile_number)

    xs = []
    ys = []
    zs = []

    for p in profiles:
        mask = profile_number == p

        idx = np.argmin(np.abs(larc_x[mask] - 778))

        xs.append(larc_x[mask][idx])
        ys.append(larc_y[mask][idx])
        zs.append(elevation[mask][idx])

    xs = np.array(xs)  #alongshore profile
    ys = np.array(ys)
    zs = np.array(zs)

    #DAS survey locations
    offshore_survey_path = './DAS_deploy/Alex_FiberLay_20250730.csv'

    offshore_survey = pd.read_csv(offshore_survey_path)
    offshore_survey_x = offshore_survey['X']
    offshore_survey_y = offshore_survey['Y']

    offshore_frf_x = []
    offshore_frf_y = []
    for x, y in zip(offshore_survey_x, offshore_survey_y):
        dict = ncsp2FRF(x, y)
        offshore_frf_x.append(dict['xFRF'])
        offshore_frf_y.append(dict['yFRF'])

    offshore_frf_x = np.array(offshore_frf_x)
    offshore_frf_y = np.array(offshore_frf_y)

    dx = np.diff(offshore_frf_x)
    dy = np.diff(offshore_frf_y)

    distance = np.sqrt(dx**2 + dy**2)

    s_survey = np.concatenate(([0], np.cumsum(distance))) #total distance along the cable in meters from the survey

    #build grid to populate - in meters
    cable_grid = np.arange(0, grid_len, 1.6)  #1.6m channel spacing
    # cable_grid = np.arange(0, s_survey.max(), 1.6)

    # interpolate outbound leg x,y from the surveyed fiber path - given a distance along the coordinate whats the x or y coordinate
    interp_x = interp1d(s_survey, offshore_frf_x, bounds_error=False)
    interp_y = interp1d(s_survey, offshore_frf_y, bounds_error=False)

    x_out = interp_x(cable_grid)
    y_out = interp_y(cable_grid)

    # mirror for the return leg to fill out N_channels
    n_return = N_channels - len(cable_grid)
    x_ret = x_out[::-1][:n_return]
    y_ret = y_out[::-1][:n_return]

    channel_x = np.concatenate([x_out, x_ret])
    channel_y = np.concatenate([y_out, y_ret])
    channel_number = np.arange(N_channels)

    # split channels by whether they're on the cross-shore or alongshore leg
    cross_shore_mask = channel_x <= 777 #xFRF location where the cable should turn
    alongshore_mask = channel_x > 777

    channel_z = np.full(N_channels, np.nan)

    # cross-shore leg: z as a function of xFRF
    sort_idx_x = np.argsort(masked_x)
    interp_z_crossshore = interp1d(masked_x[sort_idx_x], masked_elevation[sort_idx_x],
                                    bounds_error=False)
    channel_z[cross_shore_mask] = interp_z_crossshore(channel_x[cross_shore_mask])  #given an xFRF value what is the seafloor elevation here - cross-shore only

    # alongshore leg: z as a function of yFRF
    sort_idx_y = np.argsort(ys)
    interp_z_alongshore = interp1d(ys[sort_idx_y], zs[sort_idx_y],
                                    bounds_error=False)
    channel_z[alongshore_mask] = interp_z_alongshore(channel_y[alongshore_mask])  #given a yFRF value what is the seafloor elevation here - alongshore only

    channel_grid = pd.DataFrame({
        'channel': channel_number,
        'xFRF': channel_x,
        'yFRF': channel_y,
        'z': channel_z,
    })

    # save the output grid.
    # channel_grid.to_pickle('./data/frf_awac/channel_grid.pkl')

    #sanity plot
    plt.scatter(offshore_frf_x, offshore_frf_y, c='orange', s=6)
    plt.scatter(channel_grid['xFRF'], channel_grid['yFRF'], c=channel_grid['z'])
    plt.show()
    plt.close('all')



if __name__ == "__main__":

    #read_Json Files - DAS configuration and engineering chain
    optical_setup = './documentation/FRF_Fiber1Mode8_Fiber2Mode10_1.6mGL_opticalSetUp.json'
    engineering_chain = './documentation/ONYX-0564 Fiber 1__SimpleLinearAssetChainFiber1_V2_v50_engineeringChain.json'
    raw_file = os.getcwd() + '/data/raw_66/1400_UTC/decimator_2026-06-06_13.55.01_UTC_011456.h5'
    read_Json(optical_setup)

    #read attributes
    with h5py.File(raw_file, 'r') as f:
        # First, print root attributes
        print_all_attributes(raw_file, f)
        # Then, recursively visit all objects inside the file
        f.visititems(print_all_attributes)

