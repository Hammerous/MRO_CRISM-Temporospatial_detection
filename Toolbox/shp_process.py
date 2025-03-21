import os, shapely
import geopandas as gpd
import pandas as pd
from osgeo import gdal, ogr
from shapely.geometry import Polygon
from shapely.wkt import loads
from shapely.ops import unary_union
from .file_manage import convert_pid

def open_shp(file_path, cols):
    return gpd.read_file(file_path, columns=cols)

def open_csv(file_path):
    return pd.read_csv(file_path)

def load_and_merge_shp_files(directory, cols):
    # 初始化一个空列表来存储每个GeoDataFrame
    gdfs = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.shp'):
            # 将shapefile加载到GeoDataFrame中
            gdf = open_shp(os.path.join(directory, filename), cols)
            #gdf = gpd.read_file(os.path.join(directory, filename))
            # 将GeoDataFrame添加到列表中
            gdfs.append(gdf)

    # 将所有GeoDataFrame合并为一个大的GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    return merged_gdf

def find_relation_lst(gdf):
    # Perform a spatial join to find intersecting polygons
    joined = gdf.sjoin(gdf, predicate='intersects', how='left', lsuffix='left', rsuffix='right')
    # Remove self-intersections
    joined = joined[joined['ProductId_left'] != joined['ProductId_right']]
    edges = set(map(tuple, joined[['ProductId_left', 'ProductId_right']].values))
    # Convert to list of tuples
    return list(edges)

def com_overlap(gdf_geom):
    return shapely.intersection_all(gdf_geom)

def length_to_width(polygon, threshold):
    """
    Compute the Length-to-Width ratio of a polygon's bounding box.
    Returns False if the value falls outside the range [0.5, 2].
    """
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    length = bounds[2] - bounds[0]
    width = bounds[3] - bounds[1]
    
    if width == 0:
        return False  # Avoid division by zero
    
    ratio = length / width if length > width else width / length
    return ratio if ratio <= threshold else False

def raster_to_polygons(img_path, band_num=1):
    """
    Converts the first band of a raster (.img) file into polygons, excluding NoData values.
    Returns the geometries as Shapely MultiPolygon.
    
    :param img_path: Path to the input raster (.img) file.
    :param band_num: Band number to process (default is 1).
    :return: Shapely MultiPolygon geometry and CRS as WKT.
    """
    # Open the raster dataset
    raster_dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
    if raster_dataset is None:
        raise ValueError("Unable to open raster file.")
    
    band = raster_dataset.GetRasterBand(band_num)  # Use only the first band
    mask_band = band.GetMaskBand()  # Get mask to exclude NoData values
    
    # Create an in-memory vector dataset
    driver = ogr.GetDriverByName("Memory")
    output_ds = driver.CreateDataSource("out")
    srs = ogr.osr.SpatialReference()
    srs.ImportFromWkt(raster_dataset.GetProjection())
    layer = output_ds.CreateLayer("polygonized", srs=srs, geom_type=ogr.wkbPolygon)
    
    # Polygonize the raster
    gdal.Polygonize(band, mask_band, layer, -1, options=["8CONNECTED"], callback=None)

    # Extract all geometries efficiently
    polygons = [loads(feature.GetGeometryRef().ExportToWkt()) for feature in layer if feature.GetGeometryRef()]
    
    # Perform union for simplification
    dissolved_polygon = unary_union(polygons) if polygons else Polygon()

    # Clean up
    del band, raster_dataset, output_ds

    return dissolved_polygon, srs.ExportToWkt()

def find_convert_raster(folder_path, img_Id):
    img_Id = convert_pid(img_Id)
    img_path = os.path.join(folder_path, img_Id+".img")
    return raster_to_polygons(img_path, 1)

def convex_filter(gdf, threshold=0.75):
    # Compute convex hulls for all geometries in one step
    convex_hulls = gdf.geometry.apply(shapely.convex_hull)
    # Compute intersection areas in one step
    intersection_areas = gdf.geometry.intersection(convex_hulls).area
    # Filter based on overlap condition
    keep_mask = (intersection_areas / convex_hulls.area.replace(0, float('inf'))) >= threshold
    passed = gdf[keep_mask]
    # Print warning messages for dropped geometries
    dropped = gdf[~keep_mask]
    return passed.reset_index(), dropped.reset_index()

def merge_gdfs(gdf_list):
    return pd.concat(gdf_list).reset_index()

def df2gdf(df, crs):
    return gpd.GeoDataFrame(df, geometry='geometry', crs=crs)