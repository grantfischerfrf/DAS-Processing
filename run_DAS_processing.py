import os
import glob
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

from FRF_DAS_processing import process_worker, sequence_number

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"   #This must happen otherwise the workers get stuck and only the last file is saved

if __name__ == '__main__':
    datafolder = os.getcwd() + '/data/raw_66/1300_UTC'
    savefolder = os.getcwd() + '/data/processed_06062026/1300_UTC/'

    all_files = sorted(glob.glob(os.path.join(datafolder, '*.h5')), key=sequence_number)
    indices = range(1, len(all_files) - 1)

    worker = partial(process_worker, all_files=all_files, savefolder=savefolder)

    with Pool(processes=2) as pool:
        for i in tqdm(pool.imap_unordered(worker, indices), total=len(indices)):
            pass