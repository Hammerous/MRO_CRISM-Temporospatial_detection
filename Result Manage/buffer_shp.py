import os
import geopandas as gpd
from shapely.geometry import Polygon
from multiprocessing import Pool, cpu_count

def process_shapefile(file_path, output_folder, buffer_distance):
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    
    # Read shapefile
    gdf = gpd.read_file(file_path)
    
    # Create buffer
    gdf['geometry'] = gdf.geometry.buffer(buffer_distance)
    
    # Save buffered shapefile
    gdf.to_file(output_path)
    print(f"\rBuffered shapefile saved: {os.path.basename(file_path)}           ", end='')
    

if __name__ == "__main__":
    # Example usage
    input_folder = "IMG2SHP\Processed"  # Change this to your actual folder path
    output_folder = "Buffers"  # Change this to your actual output folder
    buffer_distance = 540
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare file list
    shapefiles = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(".shp")]
    
    # Prepare arguments for multiprocessing
    file_info_list = [(file, output_folder, buffer_distance) for file in shapefiles]
    
    # Use multiprocessing to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_shapefile, file_info_list)
