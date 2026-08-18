import os
import glob
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"   #This must happen otherwise the workers get stuck and only the last file is saved

from FRF_DAS_processing import process_worker, sequence_number


if __name__ == '__main__':

    #folder containing the data to be processed
    datafolder = os.getcwd() + '/data/for_OSU/20260614'

    #folder where the data is to be saved
    savefolder = './data/for_OSU_processed/20260614'

    #create folder if it does not exist
    os.makedirs(savefolder, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(datafolder, '*.h5')), key=sequence_number) #create a list of all files
    indices = range(1, len(all_files) - 1)  #indices to loop through - skip first file to avoid edge effects

    worker = partial(process_worker, all_files=all_files, savefolder=savefolder) #define a worker function for the multiprocessing

    with Pool(processes=6) as pool:  #loop through all the files and process with 'x' number of workers
        for msg in tqdm(pool.imap_unordered(worker, indices), total=len(indices), colour='yellow'):  #Processes the files in an unordered manner - can return 1, 3, 2
            pass

            tqdm.write(msg) #write message when one iteration is finished



