import glob
from FRF_DAS_processing import sequence_number
import os
from functools import partial
from Calibration import process_worker_burst, parse_das_timestamp
from multiprocessing import Pool
from collections import defaultdict
from datetime import timedelta
from utils import chunk_list
from config import DASConfig



if __name__ == "__main__":

    # --- BURST LEVEL CALIBRATION ---
    config = DASConfig()

    # all_files = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number)

    all_files = sorted(glob.glob(f'{config.data_directory}/for_OSU_processed/**/*'), key=sequence_number) #list of sorted DAS files in the data folder

    save_folder = config.data_directory / 'frf_awac/calibration_coefficients_burst_600_weighted'
    DAS_XYZ_path = config.data_directory / 'frf_awac/channel_grid.pkl'
    AWAC1_path = config.data_directory / 'frf_awac/FRF-ocean_waves_sig940-600_202606.nc'

    os.makedirs(save_folder, exist_ok=True)

    # group into top of the hour 35 minute bursts of files
    BURST_DURATION_MIN = config.burst_duration  #how long each burst is -> 35 minutes
    MIN_FILES_PER_BURST = 6  # accept bursts greater than 6 files (allows for edge-of-dataset partial hours)  #TODO: add to config.py
    BURST_TOLERANCE_SEC = 0 #tolerance in seconds

    bursts = defaultdict(list)  #creates a special dictionary the automatically creates an empty list whenever you access a new key. [] - [files]
    for f in all_files: #loop through every das file
        ts = parse_das_timestamp(f)  #extract timestamps from the file name
        if ts is None:  #if the timestamp returns nothing skip the file
            continue
        burst_start = ts.replace(minute=0, second=0, microsecond=0) #round the timestamp (datetime) down to the beginning of the hour. setting min sec usec to zero
        burst_end = burst_start + timedelta(minutes=BURST_DURATION_MIN, seconds=BURST_TOLERANCE_SEC) #add the burst duration in minutes and seconds to define the end of the burst
        if burst_start <= ts < burst_end: #if the timestamp of the file falls within the start and the end
            bursts[burst_start].append(f) #append the start of the burst to the defaultdict key - [timestamp] - [file_path1, 2, 3, ...]
            #basically creating hourly groups of files

    # keep only complete bursts, and sort by time
    complete_bursts = {
        b: sorted(files, key=sequence_number)   #sort the files by the timestamp keys
        for b, files in bursts.items()  #for the key and values in the bursts default dict. If the length of the number of files is >= 6 keep the key value pair.
        if len(files)  >= MIN_FILES_PER_BURST
    }
    burst_items = sorted(complete_bursts.items())  # list of (burst_start, [files]) tuples

    print(f"Found {len(burst_items)} complete bursts out of {len(bursts)} candidate bursts")

    n_workers = 6
    # split bursts across workers each chunk is a list of (burst_start, file_list) tuples
    burst_chunks = chunk_list(burst_items, n_workers)

    worker = partial(process_worker_burst,
                     save_folder=save_folder,
                     DAS_XYZ_path=DAS_XYZ_path,
                     AWAC1_path=AWAC1_path,
                     config=config)

    print('Calculating Calibration Coefficients ...')
    with Pool(processes=n_workers) as pool:
        for _ in pool.imap_unordered(worker, burst_chunks):
            pass

