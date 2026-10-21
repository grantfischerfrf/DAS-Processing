import os
import glob
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from decimator import decimate_data, sequence_number


if __name__ == "__main__":

    datafolder = "/mnt/d/For_OSU_raw/20260614/"  #USER INPUT - data directory
    save_dir = os.path.join(os.getcwd(), "data", "for_OSU", "20260614")  #USER INPUT

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
        save_dir=save_dir
    )

    # Create a pool of 6 worker processes to run decimation in parallel
    # Each worker independently processes a different HDF5 file
    # Make sure that you are not running into a ceiling of either CPU usage or Disk Usage.
    # Disk should be high ~80-90% for optimal processing - Not stuck at 100% - CPU can be 100%
    with Pool(processes=6) as pool:

        #apply the worker function to every file in all_files
        list(
            tqdm(
                pool.imap(worker, all_files),
                total=len(all_files)
            )
        )


