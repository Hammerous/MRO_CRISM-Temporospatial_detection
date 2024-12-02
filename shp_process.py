import os
import geopandas as gpd
import itertools
import pandas as pd

def print_crs_of_shp_file(file_path):
    # 将shapefile加载到GeoDataFrame中
    gdf = gpd.read_file(file_path)
    # 打印坐标参考系统（CRS）信息
    print(gdf.crs)
    return gdf

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
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    return merged_gdf

def find_overlapping_polygons(gdf):
    # 初始化一个字典来存储重叠多边形的索引集合
    intsec_idx = set()
    # 遍历GeoDataFrame中的每个多边形
    for idx, poly in gdf.iterrows():
        print(f"\r{idx}         ", end='')
        intsec_polys = gdf[gdf.intersects(poly['geometry'])].index.values   # 找到与当前多边形相交的多边形，排除自身
        # 如果存在重叠多边形，将集合添加到字典中
        if intsec_polys.shape[0] > 1:
            intsec_idx.add(frozenset(intsec_polys))
    return intsec_idx

directory = r'mars_mro_crism_mtrdr_c0a'
merged_gdf = load_and_merge_shp_files(directory)
print(merged_gdf.shape)
intsec_idxs = find_overlapping_polygons(merged_gdf)
all_combinations = []
for set_sqc in intsec_idxs:
    all_combinations.extend(itertools.combinations(set_sqc, 3))
    for cmb_sqc in all_combinations:
        intsec_gpf = merged_gdf[cmb_sqc]
        # 计算所有多边形的交集
        intersection = intsec_gpf.geometry.unary_union  # 合并所有多边形为一个几何对象
        common_intersection = intersection  # 公共交集可以直接取合并几何对象的 self-intersection