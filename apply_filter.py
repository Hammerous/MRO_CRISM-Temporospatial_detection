import os
from Toolbox.raster_manipulate import freq_cutoff
from Toolbox.shp_process import open_csv
import multiprocessing as mp
from tqdm import tqdm

cutoff_dict_path = r'Parameters_Summary\percentile_summary.csv'
pair_txt = 'CRISM_Metadata_Database.txt'
base_dir = r"G:\MTRDR_retrimed"
trg_dir = r'G:\MTRDR_filtered'

def worker(task):
    """
    The worker function is defined at the module level, making it pickleable. 
    This function simply unpacks the arguments from the tuple and calls rst.shp_cut_raster
    """
    img_path_input, img_path_output, cutoff_dict = task
    freq_cutoff(img_path_input, img_path_output, cutoff_dict)
    return 

if __name__ == "__main__":
    cutoff_dict = open_csv(cutoff_dict_path)
    cutoff_dict = cutoff_dict[['band_name', 'final_cutoff']].set_index(['band_name']).to_dict(index=True)['final_cutoff']

    os.makedirs(trg_dir, exist_ok=True)
    #rst.freq_cutoff(r"MRTR_Clipped\19-0\frt0000824e_07_sr163j_mtr3.tif",r"MRTR_Clipped\19-0\frt0000824e_07_sr163j_mtr3.tif", cutoff_dict)
    # Read input file and build task list
    roi_to_IMGid = {}

    # Step 1: Read input and organize image paths by ROI
    with open(pair_txt, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            roi_id, base_img, warp_img_lst = line.strip().split()
            roi_to_IMGid[roi_id] = [base_img] + warp_img_lst.split(",")
            os.makedirs(os.path.join(trg_dir, roi_id), exist_ok=True)

    # Step 2: Build flat task list for multiprocessing
    task_lst = [(os.path.join(os.path.join(base_dir, roi_id), id + ".tif"),\
                 os.path.join(os.path.join(trg_dir, roi_id), id + ".tif"), cutoff_dict)\
                for roi_id, ids in roi_to_IMGid.items() for id in ids]

    # Parallel processing with a progress bar
    with tqdm(total=len(task_lst), desc="Valid Value Filtering", unit="img") as pbar:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for _ in pool.imap_unordered(worker, task_lst):
                pbar.update(1)
