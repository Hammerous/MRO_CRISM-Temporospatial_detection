import GeoRef_Test.pair_trimed_v2 as trs_aff
import overlap_test.projection_cut_stack as prj_cut
# import Pre_Process.shp_process as pre
import pandas as pd
import ast, os

if __name__ == "__main__":
    # directory = r'mars_mro_crism_mtrdr_c0a'
    # merged_gdf = pre.load_and_merge_shp_files(directory, ['ProductId', 'LabelURL', 'UTCstart', 'geometry'])
    # # 将字符串列转换为 datetime 格式
    # merged_gdf['UTCstart'] = pre.pd.to_datetime(merged_gdf['UTCstart'])
    # print("Shp File Loaded !!!\nFinding intersecting polygons ...")
    # intsec_idxs = pre.find_overlapping_polygons(merged_gdf)
    # assessment = {'ProductIds':[], 'ProductURLs':[], 'View Num': [], 'Area(km^2)':[], 'Time Range': []}
    # print("\nAssessing Intersecting Area and View Numbers")
    # count_num = 0
    # work_num = len(intsec_idxs)
    # for set_sqc in intsec_idxs:
    #     print(f"\r {count_num}/{work_num}         ", end='')
    #     assessment['View Num'].append(len(set_sqc))
    #     # 计算所有多边形的交集
    #     intsec_gdf = merged_gdf.iloc[list(set_sqc)]
    #     # Step 2: Compute the intersection of all polygons
    #     max_overlap, max_overlap_size, max_time_span = pre.compute_max_overlap(intsec_gdf, 2)
    #     # Step 3: Calculate the area of the intersected region
    #     assessment['Area(km^2)'].append(max_overlap.area/1e6)
    #     # 将时间跨度转换为天数
    #     assessment['Time Range'].append(max_time_span.days)
    #     assessment['ProductIds'].append(intsec_gdf['ProductId'].values)
    #     assessment['ProductURLs'].append(intsec_gdf['LabelURL'].values)
    #     count_num += 1

    # # 将字典转换为DataFrame
    # assessment = pre.pd.DataFrame(assessment)


    # assessment = pd.read_csv("Pre_Process/assessment.csv")
    # # Filter records where "View Num" equals 2
    # filtered_data = assessment[assessment['View Num'] == 2]
    # # Sort the records by "Area(km^2)" column
    # sorted_data = filtered_data.sort_values(by='Area(km^2)', ascending=False)
    # # Filter out records where "ProductIds" contains "hrl"
    # filtered_no_hrl = sorted_data[~sorted_data['ProductIds'].str.contains('hrl', case=False, na=False)]
    # # Select the first ten records
    # top_ten_records = filtered_no_hrl.head(10)
    # # Convert the tuple in "ProductIds" field from string format to real tuple
    # top_ten_records.loc[:, 'ProductIds'] = top_ten_records['ProductIds'].apply(
    #     lambda x: ast.literal_eval(x.replace(" ", ",").replace("if", "sr")) if isinstance(x, str) else x
    # )
    # print(top_ten_records['ProductIds'].values)
    # original_dir = r"G:\MRTR_SR"
    target_directory = "Round1"
    # os.makedirs(target_directory,  exist_ok=True) 
    # warp_rgb = ('R2529', 'R1506', 'R1080')
    # for base, warp in top_ten_records['ProductIds'].values:
    #     base_img_path = os.path.join(original_dir, base+".img")
    #     warp_img_path = os.path.join(original_dir, warp+".img")
    #     result_path = os.path.join(target_directory, f"{base}_{warp}")
    #     os.makedirs(result_path,  exist_ok=True) 
    #     trs_aff.georeferencing(os.path.join(result_path, warp), base_img_path, warp_img_path, warp_rgb)

    # Traverse the directory tree
    for root, dirs, files in os.walk(target_directory):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            prj_cut.main(folder_path, folder_path+".tif")