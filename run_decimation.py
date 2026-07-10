import os
import glob
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from decimator import decimate_data
from FRF_DAS_processing import sequence_number


if __name__ == "__main__":

    #!!multiprocessing must be run from a separate script that loads the functions

    datafolder = "/mnt/d/For_OSU_raw/20260602/"
    save_dir = os.getcwd() + "/data/for_OSU/20260602"

    #creates a sorted list of files for processing
    all_files = sorted(
        glob.glob(os.path.join(datafolder, "*.h5")),
        key=sequence_number
    )

    # Create a worker function where save_dir is fixed so multiprocessing
    # only needs to pass each input file path to decimate_data()
    worker = partial(
        decimate_data,
        save_dir=save_dir
    )

    # Create a pool of 6 worker processes to run decimation in parallel
    # Each worker independently processes a different HDF5 file
    with Pool(processes=6) as pool:

        #apply the worker function to every file in all_files
        list(
            tqdm(
                pool.imap(worker, all_files),
                total=len(all_files)
            )
        )


