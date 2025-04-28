import os
import Toolbox.raster_manipulate as rst
from Toolbox.shp_process import open_csv
import multiprocessing as mp
from tqdm import tqdm

cutoff_dict_path = r'Parameters_Summary\percentile_summary.csv'
pair_txt = 'CRISM_Metadata_Database.txt'
base_dir = r"H:\MTRDR_retrimed"
trg_dir = r'H:\MTRDR_filtered'

def worker(args):
    """
    The worker function is defined at the module level, making it pickleable. 
    This function simply unpacks the arguments from the tuple and calls rst.shp_cut_raster
    """
    return rst.freq_cutoff(*args)

if __name__ == "__main__":
    cutoff_dict = open_csv(cutoff_dict_path)
    cutoff_dict = cutoff_dict[['band_name', 'final_cutoff']].set_index(['band_name']).to_dict(index=True)

    os.makedirs(trg_dir, exist_ok=True)
    #rst.freq_cutoff(r"MRTR_Clipped\19-0\frt0000824e_07_sr163j_mtr3.tif",r"MRTR_Clipped\19-0\frt0000824e_07_sr163j_mtr3.tif", cutoff_dict)
    # Read input file and build task list
    roi_to_paths = {}

    # Step 1: Read input and organize image paths by ROI
    with open(pair_txt, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            ROI_id, base_img, warp_img_lst = line.strip().split()
            base_folder_path = os.path.join(base_dir, ROI_id)
            img_paths = [os.path.join(base_folder_path, base_img + ".tif")]
            img_paths += [os.path.join(base_folder_path, warp_img + ".tif") for warp_img in warp_img_lst.split(",")]
            roi_to_paths[ROI_id] = img_paths

    # Step 2: Build flat task list for multiprocessing
    task_list = [(roi_id, path) for roi_id, paths in roi_to_paths.items() for path in paths]

    # Parallel processing with a progress bar
    with tqdm(total=len(task_lst), desc="Change Detecting", unit="img") as pbar:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for _ in pool.imap_unordered(worker, task_lst):
                pbar.update(1)
