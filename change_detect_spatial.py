import os
import Toolbox.raster_manipulate as rst
import multiprocessing as mp
from tqdm import tqdm

pair_txt = 'CRISM_Metadata_Database.txt'
base_dir = "MRTR_Clipped"
resample_dir = "Round2"

trg_dir = "Change_Detect"

def worker(args):
    """
    The worker function is defined at the module level, making it pickleable. 
    This function simply unpacks the arguments from the tuple and calls rst.shp_cut_raster
    """
    return rst.change_detect(*args)

if __name__ == "__main__":
    os.makedirs(trg_dir, exist_ok=True)
    #rst.change_detect(r"MRTR_Clipped\19-0\frt0000824e_07_sr163j_mtr3.tif",r"Round2\19-0\frt00005af7_07_sr163j_mtr3.tif", r"output.tif")
    task_lst = []
    # Read input file and build task list
    with open(pair_txt, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            ROI_id, base_img, warp_img_lst = line.split()
            base_folder_path = os.path.join(base_dir, ROI_id)
            resample_folder_path = os.path.join(resample_dir, ROI_id)
            trg_folder_path = os.path.join(trg_dir, ROI_id)
            os.makedirs(trg_folder_path, exist_ok=True)
            task_lst.extend([(os.path.join(base_folder_path, base_img+".tif"),
                              os.path.join(resample_folder_path, warp_img+".tif"), 
                              os.path.join(trg_folder_path, warp_img+".tif"))
                             for warp_img in warp_img_lst.split(",")])

    # Parallel processing with a progress bar
    with tqdm(total=len(task_lst), desc="Change Detecting", unit="img") as pbar:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for _ in pool.imap_unordered(worker, task_lst):
                pbar.update(1)
