import Toolbox.shp_process as shp
import multiprocessing as mp
import os, hashlib

raster_grouped = "mars_mro_crism_mtrdr_c0a.csv"
input_folder = r"G:\MRTR_SR"
output_folder = "IMG2SHP"

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    raster_df = shp.open_csv(raster_grouped)
    buffered_results = {}  # Buffer for storing results for each group

    # Create a pool using all available CPUs
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Process each group from the CSV
        for idx, group in raster_df.groupby(['Max View', 'Group Id']):
            print(f"\r{idx[0]}-{idx[1]}             ", end='')
            product_ids = group['ProductId'].values

            # Use pool.starmap to call find_convert_raster concurrently for each ProductId
            # starmap takes an iterable of argument tuples.
            results = pool.starmap(shp.find_convert_raster, [(input_folder, pid) for pid in product_ids])
            
            # Assign the parallel results (each a tuple of (geometry, crs)) to the group DataFrame
            group[['geometry', 'crs']] = results
            buffered_results[idx] = group.copy()  # Buffer the processed group
    
    print("\nSaving to shps ...")
    # After all multiprocessing tasks have completed, write the shapefiles
    for idx, df in buffered_results.items():
        new_crs = set(df['crs'])
        if len(new_crs) == 1:
            df.drop(columns=['crs'], inplace=True)
            gdf = shp.df2gdf(df, list(new_crs)[0])
            out_file = os.path.join(output_folder, f"{idx[0]}-{idx[1]}.shp")
            gdf.to_file(out_file)
        else:
            for crs_value in new_crs:
                sub_df = df[df['crs'] == crs_value].drop(columns=['crs'])
                gdf = shp.df2gdf(sub_df, crs_value)
                crs_id = f"CRS_{hashlib.md5(crs_value.encode()).hexdigest()[:6]}"
                out_file = os.path.join(output_folder, f"{idx[0]}-{idx[1]}_crs{crs_id}.shp")
                gdf.to_file(out_file)
            print(f"{idx[0]}-{idx[1]} contains multiple CRS values. Saved separately.")