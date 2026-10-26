import os
import glob
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from decimator import decimate_data, sequence_number
from config import DASConfig


if __name__ == "__main__":

    config = DASConfig()

    datafolder = config.data_directory / 'raw_66/test'  #USER INPUT - data directory - a folder containing .h5 files
    save_dir = config.data_directory / 'processed_66/test'  #USER INPUT - save directory

    #creates a sorted list of files for processing
    all_files = sorted(
        glob.glob(os.path.join(datafolder, "*.h5")), #pull all hdf5 files in the folder
        key=sequence_number
    )

    #check if save directory exists - create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a worker function where save_dir is fixed so multiprocessing
    # only needs to pass each input file path to decimate_data()
    worker = partial(
        decimate_data,
        save_dir=save_dir, initial_frequency=500.0, decimated_frequency=2.0
    )

    # Create a pool of 'N' worker processes to run decimation in parallel
    # Each worker independently processes a different HDF5 file
    # Make sure that you are not running into a ceiling of either CPU usage or Disk Usage.
    # Disk should be high ~80-90% for optimal processing - Not stuck at 100% - CPU can be near 100%
    with Pool(processes=6) as pool:

        #apply the worker function to every file in all_files
        list(
            tqdm(
                pool.imap(worker, all_files),
                total=len(all_files), colour='yellow', desc='Files'
            )
        )


