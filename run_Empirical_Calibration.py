import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import glob
from FRF_DAS_processing import sequence_number, strain2phase
import os
import re
from FRF_DAS_plots import load_processedData
from scipy.signal import welch
from tqdm import tqdm
from functools import partial
from Empirical_Calibration import process_worker
from multiprocessing import Pool



if __name__ == "__main__":

    all_files = sorted(glob.glob('./data/for_OSU_processed/**/*'), key=sequence_number) #all files to loop through

    save_folder = './data/frf_awac/calibration_coefficients'  #directory to save the data to

    DAS_XYZ_path = './data/frf_awac/channel_grid.pkl'

    AWAC1_path = './data/frf_awac/FRF-ocean_waves_sig940-300_202606.nc'

    os.makedirs(save_folder, exist_ok=True) #make save folder if it does not already exist

    n_workers = 6
    chunks = np.array_split(all_files, n_workers)  # split files across workers

    worker = partial(process_worker,
                      save_folder=save_folder,
                      DAS_XYZ_path=DAS_XYZ_path,
                      AWAC1_path=AWAC1_path,
                      gamma_break=0.43)

    print('Calculating Calibration Coefficients ...')
    with Pool(processes=n_workers) as pool:
        for _ in pool.imap_unordered(worker, chunks):
            pass

