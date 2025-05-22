import os
import Toolbox.cv_match as cvm
import Toolbox.raster_manipulate as rst
import multiprocessing as mp
from tqdm import tqdm

pair_txt = 'IMG2SHP_results.txt'
trg_dir = "Round2"
src_dir = r"H:\MRTR_Clipped"
os.makedirs(trg_dir, exist_ok=True)
warp_serial = (15, 15, 15) #(R600, R530, R440)

def georeferencing(ROI_id, base_img, warp_img):
    try:
        # Step 1: Read the images
        base_img_data = rst.open_img(os.path.join(src_dir + f"/{ROI_id}", base_img + ".tif"))
        img2prj = rst.open_img(os.path.join(src_dir + f"/{ROI_id}", warp_img + ".tif"))
        img1_gray, img1_alpha = rst.BGR2GRAY(base_img_data, warp_serial)
        img2_gray, img2_alpha = rst.BGR2GRAY(img2prj, warp_serial)

        # Step 2: Feature matching
        h_mat, points1, points2, status = cvm.denseSIFT(img1_gray, img1_alpha, img2_gray, img2_alpha,
                                        output_file_name=None,\
                                        ratio_threshold=0.70, ransac_reproj_threshold=5.0, Layers=10, grid_spacing=4)
        
        del img1_gray, img1_alpha, img2_gray, img2_alpha

        if status:
            # Step 3: Apply affine transformation
            count, rmse, mae = rst.affine_trans(h_mat, points1, points2, img2prj, base_img_data.GetProjection(), base_img_data.GetGeoTransform())
            return base_img, warp_img, count, rmse, mae
        else: 
            return None, None, None, None, None
    except Exception as e:
            # log the error and return empty result tuple
            return None, None, None, None, None
    # img2prj = rst.affine_trans(h_mat, points1, points2, img2prj, base_img_data.GetProjection(), base_img_data.GetGeoTransform())
    # if not img2prj:
    #     print(f"{warp_img} failed to be affinable")
    #     return
    #rst.save_gdal_dataset(img2prj, os.path.join(os.path.join(trg_dir, ROI_id), warp_img + ".tif"))

def worker(args):
    """
    The worker function is defined at the module level, making it pickleable. 
    This function simply unpacks the arguments from the tuple and calls rst.shp_cut_raster
    """
    return georeferencing(*args)

if __name__ == "__main__":
    # 1) Build task list
    task_lst    = []
    roi_set     = set()
    with open(pair_txt, 'r', encoding='utf-8') as in_f:
        for line in in_f:
            ROI_id, base_img, warp_list = line.strip().split()
            roi_set.add(ROI_id)
            for warp_img in warp_list.split(','):
                task_lst.append((ROI_id, base_img, warp_img))

    # 2) Prepare directories
    os.makedirs(trg_dir, exist_ok=True)
    for ROI_id in roi_set:
        roi_folder = os.path.join(trg_dir, ROI_id)
        os.makedirs(roi_folder, exist_ok=True)
    # also ensure base_img subfolders exist as we write SIFT outputs there
    for _, base_img, _ in task_lst:
        os.makedirs(os.path.join(trg_dir, ROI_id, base_img), exist_ok=True)

    # 3) Open a single TXT for writing all results
    out_txt = os.path.join(trg_dir, "georef_results.txt")
    with open(out_txt, 'w', encoding='utf-8') as out_f:
        # 4) Parallel processing with a progress bar
        with tqdm(total=len(task_lst), desc="Georeferencing", unit="pair") as pbar:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                for result in pool.imap_unordered(worker, task_lst):
                    # result is (ROI_id, base_img, warp_img, count, rmse, mae)
                    if result[0]:
                        out_f.write("\t".join(map(str, result)) + "\n")
                    pbar.update(1)