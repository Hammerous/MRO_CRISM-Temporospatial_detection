import os
import Toolbox.cv_match as cvm
import Toolbox.raster_manipulate as rst
import multiprocessing as mp
from tqdm import tqdm

pair_txt = 'merged.txt'
trg_dir = "Round2"
src_dir = r"H:\MRTR_Clipped"
os.makedirs(trg_dir, exist_ok=True)
warp_serial = (15, 15, 15)

def georeferencing(ROI_id, base_img, warp_img):
    # Step 1: Read the images
    base_img_data = rst.open_img(os.path.join(src_dir + f"/{ROI_id}", base_img + ".tif"))
    img2prj = rst.open_img(os.path.join(src_dir + f"/{ROI_id}", warp_img + ".tif"))
    img1_gray, img1_alpha = rst.BGR2GRAY(base_img_data, warp_serial)
    img2_gray, img2_alpha = rst.BGR2GRAY(img2prj, warp_serial)

    # Step 2: Feature matching
    h_mat, points1, points2 = cvm.denseSIFT(img1_gray, img1_alpha, img2_gray, img2_alpha,
                                    output_file_name=os.path.join(os.path.join(trg_dir, base_img), warp_img + "_SIFT"),\
                                    ratio_threshold=0.70, ransac_reproj_threshold=5.0, Layers=10, grid_spacing=3)
    
    del img1_gray, img1_alpha, img2_gray, img2_alpha

    # Step 3: Apply affine transformation
    img2prj = rst.affine_trans(h_mat, points1, points2, img2prj, base_img_data.GetProjection(), base_img_data.GetGeoTransform())
    if not img2prj:
        print(f"{warp_img} failed to be affinable")
        return
    rst.save_gdal_dataset(img2prj, os.path.join(os.path.join(trg_dir, ROI_id), warp_img + ".tif"))

def worker(args):
    """
    The worker function is defined at the module level, making it pickleable. 
    This function simply unpacks the arguments from the tuple and calls rst.shp_cut_raster
    """
    return georeferencing(*args)

if __name__ == "__main__":
    task_lst = []
    ROI_id_lst = set()

    # Read input file and build task list
    with open(pair_txt, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            ROI_id, base_img, warp_img_lst = line.split()
            task_lst.extend([(ROI_id, base_img, warp_img) for warp_img in warp_img_lst.split(",")])
            ROI_id_lst.add(ROI_id)
    ROI_id_lst = list(ROI_id_lst)
    
    for ROI_id in ROI_id_lst:
        folder_path = os.path.join(trg_dir, ROI_id)
        os.makedirs(folder_path, exist_ok=True)

    # Parallel processing with a progress bar
    with tqdm(total=len(task_lst), desc="Georeferencing", unit="pair") as pbar:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for _ in pool.imap_unordered(worker, task_lst):
                pbar.update(1)
