import os
import geopandas as gpd
import pandas as pd

mars_equal_area_crs = (
    "+proj=cea +lon_0=0 +lat_ts=0 +a=3396190 +b=3396190 +units=m +no_defs"
    )


def print_crs_of_shp_file(file_path):
    # 将shapefile加载到GeoDataFrame中
    gdf = gpd.read_file(file_path)
    # 打印坐标参考系统（CRS）信息
    print(gdf.crs)
    gdf_projected = gdf.to_crs(mars_equal_area_crs)
    print(gdf_projected.crs)
    return gdf_projected

def load_and_merge_shp_files(directory):
    # 初始化一个空列表来存储每个GeoDataFrame
    gdfs = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.shp'):
            # 将shapefile加载到GeoDataFrame中
            gdf = print_crs_of_shp_file(os.path.join(directory, filename))
            #gdf = gpd.read_file(os.path.join(directory, filename))
            # 将GeoDataFrame添加到列表中
            gdfs.append(gdf)

    # 将所有GeoDataFrame合并为一个大的GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=mars_equal_area_crs)

    return merged_gdf

target_type = ('Polygon', 'MultiPolygon')
def find_overlapping_polygons(gdf):
    # Initialize a list to store the unique intersection geometries
    intersection_geometries = set()  # Using a set to automatically handle duplicates
    # Create a spatial index to quickly find intersections
    spatial_index = gdf.sindex
    
    # Iterate through each polygon in the GeoDataFrame
    for idx, poly in gdf.iterrows():
        if not idx%100:
            print(f"\r{idx}         ", end="")
        
        # Use the spatial index to find potential intersections
        possible_matches_index = list(spatial_index.intersection(poly['geometry'].bounds))
        
        # Filter the potential matches by checking if they actually intersect
        intersecting_polys = gdf.iloc[possible_matches_index]
        intersecting_polys = intersecting_polys[intersecting_polys.intersects(poly['geometry'])]
        
        # For each intersecting polygon, calculate the actual intersection geometry
        for _, other_poly in intersecting_polys.iterrows():
            # Skip if it's the same polygon (ignore self-intersection)
            if idx == _:
                continue
            
            # Get the intersection geometry between the two polygons
            intersection_geom = poly['geometry'].intersection(other_poly['geometry'])

            # Only add non-empty and valid intersection geometries
            if intersection_geom.is_valid and not intersection_geom.is_empty and intersection_geom.geom_type in target_type:
                intersection_geometries.add(intersection_geom)
    
    # Return the list of unique intersection geometries
    return list(intersection_geometries)

def process_and_dissolve_polygons(gdf, area_threshold_km2=3.24):
    # Calculate area of each geometry (ensure CRS is in meters for area calculation)   
    gdf['area'] = gdf['geometry'].area / 10**6  # Area in square kilometers (since we are using meters)
    
    # Filter polygons based on area threshold (e.g., polygons larger than 3.24 km²)
    filtered_gdf = gdf[gdf['area'] >= area_threshold_km2]
    
    return filtered_gdf

if __name__ == "__main__":
    directory = r'mars_mro_crism_mtrdr_c0a'
    merged_gdf = load_and_merge_shp_files(directory)[['ProductId', 'LabelURL', 'UTCstart', 'geometry']]
    merged_gdf['UTCstart'] = pd.to_datetime(merged_gdf['UTCstart'])

    print("Shp File Loaded !!!\nFinding intersecting polygons ...")
    intersection_polygons = find_overlapping_polygons(merged_gdf)

    # Optionally, create a new GeoDataFrame with the intersection geometries
    intersection_gdf = gpd.GeoDataFrame(geometry=intersection_polygons, crs=merged_gdf.crs)

    # You can now save or visualize the intersection polygons
    intersection_gdf.to_file("intersection_polygons.shp")

    # Process polygons, filter by area
    filtered_gdf = process_and_dissolve_polygons(intersection_gdf)

    # Optionally, save the results to shapefiles
    filtered_gdf.to_file("filtered_polygons.shp")  # Polygons filtered by area