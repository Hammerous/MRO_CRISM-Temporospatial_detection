import os
import geopandas as gpd
import pandas as pd
from itertools import combinations

def print_crs_of_shp_file(file_path, cols):
    # 将shapefile加载到GeoDataFrame中
    gdf = gpd.read_file(file_path, columns=cols)
    # 打印坐标参考系统（CRS）信息
    print(gdf.crs)
    mars_equal_area_crs = (
    "+proj=cea +lon_0=0 +lat_ts=0 +a=3396190 +b=3396190 +units=m +no_defs"
    )
    gdf_projected = gdf.to_crs(mars_equal_area_crs)
    print(gdf_projected.crs)
    return gdf_projected

def load_and_merge_shp_files(directory, cols):
    # 初始化一个空列表来存储每个GeoDataFrame
    gdfs = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.shp'):
            # 将shapefile加载到GeoDataFrame中
            gdf = print_crs_of_shp_file(os.path.join(directory, filename), cols)
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
        intsec_polys = gdf[gdf.intersects(poly['geometry'])]   # 找到与当前多边形相交的多边形，包括自身
        product_Ids = set(intsec_polys['ProductId'].values)
        # 如果存在重叠多边形，将集合添加到字典中
        if len(product_Ids) > 1:
            intsec_idx.add(frozenset(intsec_polys.index.tolist()))
    return intsec_idx


def compute_max_overlap(gdf, max_num):
    """
    计算最大公共相交区域，限制组合计算的最大数量。
    
    参数:
        gdf (GeoDataFrame): 包含多边形和时间属性的 GeoDataFrame。
        max_num (int): 组合的最大数量。
        
    返回:
        tuple: (最大相交区域, 对应的组合数量, 最大时间跨度)
    """
    max_overlap_area = None
    max_overlap_size = 0
    max_time_span = None

    for n in range(2, max_num + 1):
        for indices in combinations(range(len(gdf)), n):
            selected_geometries = gdf.iloc[list(indices)].geometry
            overlap = selected_geometries.iloc[0]
            for geom in selected_geometries.iloc[1:]:
                overlap = overlap.intersection(geom)
                if overlap.is_empty:
                    break  # 如果没有交集，跳出当前组合
            else:
                # 如果有有效的交集
                if max_overlap_area is None or overlap.area > max_overlap_area.area:
                    max_overlap_area = overlap
                    max_overlap_size = n
                    max_time_span = gdf.iloc[list(indices)]["UTCstart"].max() - gdf.iloc[list(indices)]["UTCstart"].min()
        
        # 如果在当前组合大小下无交集，则返回上一次的结果
        if max_overlap_area is None or max_overlap_area.is_empty:
            break

    return max_overlap_area, max_overlap_size, max_time_span

if __name__ == "__main__":
    directory = r'mars_mro_crism_mtrdr_c0a'
    merged_gdf = load_and_merge_shp_files(directory, ['ProductId', 'LabelURL', 'UTCstart', 'geometry'])
    # 将字符串列转换为 datetime 格式
    merged_gdf['UTCstart'] = pd.to_datetime(merged_gdf['UTCstart'])
    print("Shp File Loaded !!!\nFinding intersecting polygons ...")
    intsec_idxs = find_overlapping_polygons(merged_gdf)
    assessment = {'ProductIds':[], 'ProductURLs':[], 'View Num': [], 'Area(km^2)':[], 'Time Range': []}
    print("\nAssessing Intersecting Area and View Numbers")
    count_num = 0
    work_num = len(intsec_idxs)
    for set_sqc in intsec_idxs:
        print(f"\r {count_num}/{work_num}         ", end='')
        assessment['View Num'].append(len(set_sqc))
        # 计算所有多边形的交集
        intsec_gdf = merged_gdf.iloc[list(set_sqc)]
        # Step 2: Compute the intersection of all polygons
        max_overlap, max_overlap_size, max_time_span = compute_max_overlap(intsec_gdf, 3)
        # Step 3: Calculate the area of the intersected region
        assessment['Area(km^2)'].append(max_overlap.area/1e6)
        # 将时间跨度转换为天数
        assessment['Time Range'].append(max_time_span.days)
        assessment['ProductIds'].append(intsec_gdf['ProductId'].values)
        assessment['ProductURLs'].append(intsec_gdf['LabelURL'].values)
        count_num += 1

    # 将字典转换为DataFrame
    assessment = pd.DataFrame(assessment)
    # 保存为CSV文件
    assessment.to_csv('assessment.csv', index=False, encoding='utf-8-sig')
    print("\nDictionary has been saved to assessment.csv")