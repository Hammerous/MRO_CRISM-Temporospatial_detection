import Toolbox.shp_process as shp
import Toolbox.graph_process as gph
from itertools import combinations
import geopandas as gpd
import multiprocessing as mp
import os

pixel_size = 18
minimun_pixels = 400**2
minimun_area = minimun_pixels * pixel_size ** 2
input_dir = r'IMG2SHP'
output_folder = r'IMG2SHP/Processed'
invalid_folder = os.path.join(input_dir, "Invalid")

def weighted_relations(gdf):
    idx_comb = list(combinations(gdf.index, 2))
    # Compute weight as intersection area divided by union area (Jaccard similarity)
    edgelist = [(idx1, idx2, gdf.geometry.iloc[idx1].intersection(gdf.geometry.iloc[idx2]).area /
                          gdf.geometry.iloc[idx1].union(gdf.geometry.iloc[idx2]).area)
                for idx1, idx2 in idx_comb]
    return edgelist

def find_best_stacks(input_shp):
    print(f"\rProcessing {input_shp}            ", end='')
    # Step 1: Open the input shapefile
    gdf = shp.open_shp(input_shp, ['ProductId', 'geometry'])
    gdf, dropped = shp.convex_filter(gdf, threshold=0.75)
    
    if dropped.shape[0]:
        print(f"Product: {dropped['ProductId'].values} insufficient coverage to ensure integrity")
        dropped.to_file(os.path.join(invalid_folder, os.path.basename(input_shp)))
    
    if gdf.shape[0]:
        weighted_pairlst = weighted_relations(gdf)
        weighted_dgs = gph.find_weighted_degrees(weighted_pairlst)
        max_idx = max(weighted_dgs, key=weighted_dgs.get)

        while gdf.shape[0] > 1:
            overlap = shp.com_overlap(gdf.geometry)
            min_idx = min(weighted_dgs, key=weighted_dgs.get)
            if overlap.area > minimun_area and shp.length_to_width(overlap, threshold=5) and overlap.geom_type == 'Polygon':
                max_pid = gdf.loc[max_idx]['ProductId']
                if output_folder:
                    output_shp_path = os.path.join(output_folder, os.path.basename(input_shp))
                    new_gdf = gpd.GeoDataFrame({"Base Id": [max_pid]}, geometry=[overlap], crs=gdf.crs)
                    new_gdf.to_file(output_shp_path)
                return f"{os.path.basename(input_shp).split('.')[0]} {shp.convert_pid(max_pid)} " +\
                        ",".join([shp.convert_pid(pid) for pid in gdf.drop(index=max_idx)['ProductId'].to_list()])
            else:
                gdf.drop(index=min_idx, inplace=True)
                weighted_dgs.pop(min_idx, None)  # Remove from dictionary if it exists

    return False

if __name__ == "__main__":
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    os.makedirs(invalid_folder, exist_ok=True)
    shp_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".shp")]
    with mp.Pool(mp.cpu_count()) as pool:
         results = pool.map(find_best_stacks, shp_files)
    # Save results to a text file
    print()
    with open(input_dir + "_results.txt", "w") as f:
        for str in results:
            if str:
                f.write(f"{str}\n")
    print("Processing completed for all shapefiles.")