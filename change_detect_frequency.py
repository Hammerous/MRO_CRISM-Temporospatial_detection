import os
import pandas as pd
import Toolbox.raster_manipulate as rst
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict

pair_txt = 'CRISM_Metadata_Database.txt'
pair_folder = 'CRISM_Metadata_Database'
base_dir = r"G:\MTRDR_filtered"
trg_dir = "Change_Detect"

# Define column names based on freq_summary output
freq_columns = [
    "band_name",
    "count",
    "mean_val",
    "std_val",
    "skewness",
    "min_val",
    "first_quantile",
    "median_val",
    "third_quantile",
    "max_val"
]

def worker(task):
    ROI_id, img_path = task
    result = rst.freq_summary(img_path)  # shape: [60, 9]
    df = pd.DataFrame(result, columns=freq_columns)
    df["ProductId"] = os.path.basename(img_path).split('.')[0]  # add image name as a column
    return ROI_id, df

if __name__ == "__main__":
    os.makedirs(trg_dir, exist_ok=True)
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

    # Step 3: Parallel processing
    results_dict = defaultdict(list)

    with tqdm(total=len(task_list), desc="Summarizing images", unit="img") as pbar:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for ROI_id, df in pool.imap_unordered(worker, task_list):
                results_dict[ROI_id].append(df)
                pbar.update(1)

    # Step 4: Combine and save per ROI
    for ROI_id, _ in roi_to_paths.items():
        meta_df = pd.read_csv(os.path.join(pair_folder, ROI_id+".csv"))

        # Concatenate all DataFrames for the ROI
        full_df = pd.concat(results_dict[ROI_id], ignore_index=True)

        # Merge frequency summary data with metadata on 'ProductId'
        full_df = pd.merge(full_df, meta_df, on="ProductId", how="left")

        full_df = full_df.sort_values(by="UTCstart")

        # Save CSV
        out_path = os.path.join(trg_dir, f"{ROI_id}.csv")
        full_df.to_csv(out_path, index=False)
