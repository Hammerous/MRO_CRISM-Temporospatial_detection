import os
import Toolbox.raster_manipulate as rst
import multiprocessing as mp
from tqdm import tqdm

pair_txt = 'IMG2SHP_results.txt'
trg_dir = r"G:\MRTR_Clipped"
raster_dir = r"G:\MRTR_SR"
shp_dir = r"Buffers"
os.makedirs(trg_dir, exist_ok=True)

def worker(args):
    """
    The worker function is defined at the module level, making it pickleable. 
    This function simply unpacks the arguments from the tuple and calls rst.shp_cut_raster
    """
    return rst.shp_cut_raster(*args)

if __name__ == "__main__":
    task_lst = []
    # Read input file and build task list
    with open(pair_txt, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            ROI_id, base_img, warp_img_lst = line.split()
            folder_path = os.path.join(trg_dir, ROI_id)
            #sub_raster_dir = os.path.join(raster_dir, ROI_id)
            os.makedirs(folder_path, exist_ok=True)
            shp_path = os.path.join(shp_dir, ROI_id+".shp")
            task_lst.extend([(shp_path, os.path.join(raster_dir, warp_img+".img"), os.path.join(folder_path, warp_img+".tif")) for warp_img in warp_img_lst.split(",")])
            task_lst.append((shp_path, os.path.join(raster_dir, base_img+".img"), os.path.join(folder_path, base_img+".tif")))

    # Parallel processing with a progress bar
    with tqdm(total=len(task_lst), desc="Clipping", unit="img") as pbar:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for _ in pool.imap_unordered(worker, task_lst):
                pbar.update(1)
