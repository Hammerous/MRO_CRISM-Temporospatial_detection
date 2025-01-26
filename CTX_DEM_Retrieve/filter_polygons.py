from shapely.geometry import box
from pyproj import CRS
import geopandas as gpd

file_path = "filtered_polygons.shp"

mars_lonlat = CRS.from_proj4(
    "+proj=longlat +a=3396190 +b=3396190 +no_defs"
)

def filter_within_bbox(gdf, bbox):
    # Create a bounding box geometry using the provided coordinates
    min_x, min_y, max_x, max_y = bbox
    bbox_geom = box(min_x, min_y, max_x, max_y)
    
    # Filter the geometries that are completely within the bounding box
    filtered_gdf = gdf[gdf['geometry'].within(bbox_geom)]
    
    return filtered_gdf

if __name__ == "__main__":
    gdf = gpd.read_file(file_path)

    dissolved_gdf = gdf.to_crs(mars_lonlat)

    # Define the bounding box coordinates
    bbox = [-179.754215560711, -65.4460986614572, 179.755664515299, 65.9419173856088]

    # Filter the dissolved GeoDataFrame based on the bounding box
    filtered_dissolved_gdf = filter_within_bbox(dissolved_gdf, bbox)

    # Optionally, save the filtered results to a new shapefile
    filtered_dissolved_gdf.to_file("filtered_polygons_bboxed.shp")