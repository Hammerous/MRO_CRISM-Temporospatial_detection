import pandas as pd
import re
from shp_process import print_crs_of_shp_file

ass_df = pd.read_csv('assessment.csv')
productId_set = set()
productURL_set = set()
for idx, row in ass_df.iterrows():
    if row['Area(km^2)'] > 3.24:
        # 使用正则表达式将缺少逗号的字符串分割为列表
        item_list = re.findall(r'\w+', row['ProductIds'])
        # Extract URLs from ProductURLs
        url_list = re.findall(r'https?://[^\s,]+', row['ProductURLs'])  # Updated regex to handle space-separated URLs
        # Clean up trailing characters from URLs
        url_list = [url.rstrip("]'\"") for url in url_list]  # Remove trailing ], ' or "
        # 将list中的元素添加到set中
        productId_set.update(item_list)
        productURL_set.update(url_list)

# 生成文件链接的函数
def generate_file_links(url_set, suffixes):
    if_links = []
    sr_links = []
    for url in url_set:
        base_url = url[:-4]  # 去掉.xml部分
        
        for suffix in suffixes:
            if_links.append(f"{base_url}.{suffix}")

        # 替换第三项元素中的'if'为'sr'的函数
        parts = base_url.split('_')
        if len(parts) > 2 and parts[-2][:2] == 'if':
            parts[-2] = parts[-2].replace('if', 'sr')
            new_base_url = '_'.join(parts)
            for suffix in suffixes:
                sr_links.append(f"{new_base_url}.{suffix}")
    return if_links, sr_links

# 定义后缀
suffixes = ['hdr', 'img', 'lbl', 'xml']

# 生成文件链接并保存到txt文件
if_links, sr_links = generate_file_links(productURL_set, suffixes)

# 将原始链接写入txt文件
with open('if_links.txt', 'w') as f:
    for link in if_links:
        f.write(f"{link}\n")

# 将修改后的链接写入另一个txt文件
with open('sr_links.txt', 'w') as f:
    for link in sr_links:
        f.write(f"{link}\n")
print("链接已成功写入if_links.txt和sr_links.txt")

# print(len(productId_set))
exit()
# 读取 shapefile 并设置索引
path = r'mars_mro_crism_mtrdr_c0a\mars_mro_crism_mtrdr_c0a.shp'
gdf = print_crs_of_shp_file(path)
gdf.set_index('ProductId', inplace=True)
# 创建一个副本以便动态剔除已经遍历到的多边形
remaining_gdf = gdf.copy()
dropped_set = set()
# 遍历 GeoDataFrame 并剔除不在 productId_set 中的多边形
for idx, poly in gdf.iterrows():
    if idx in productId_set and idx not in dropped_set:
        dropped_set.add(idx)
        remaining_gdf.drop(idx, inplace=True)
# 将 remaining_gdf 保存为新的 shapefile
new_path = r'new_shapefile.shp'
remaining_gdf.to_file(new_path)
print(f"Remaining GeoDataFrame has been saved to {new_path}")


'''
import os
import geopandas as gpd
import pandas as pd
from shp_process import print_crs_of_shp_file

def filter_polygons(gdf):
    intsec_idx = []
    # 创建一个副本以便动态剔除已经遍历到的多边形
    remaining_gdf = gdf.copy()
    # 遍历GeoDataFrame中的每个多边形
    for idx, poly in gdf.iterrows():
        if idx not in remaining_gdf.index:
            continue  # 如果当前多边形已经被剔除，则跳过
        print(f"\r{idx}         ", end='')
        intsec_polys = remaining_gdf[remaining_gdf.intersects(poly['geometry'])]  # 找到与当前多边形相交的多边形，包括自身
        product_Ids = set(intsec_polys['ProductId'].values)
        # 如果存在重叠多边形，将集合添加到字典中
        if len(product_Ids) > 1:
            intsec_idx.extend(intsec_polys.index.tolist())
        # 从剩余的GeoDataFrame中剔除已经遍历到的多边形
        remaining_gdf = remaining_gdf.drop(intsec_polys.index)
    print()
    return intsec_idx

path = r'mars_mro_crism_mtrdr_c0a\mars_mro_crism_mtrdr_c0a.shp'
gdf = print_crs_of_shp_file(path)
print("Shp File Loaded !!!\nFiltering polygons ...")
intsec_idxs = filter_polygons(gdf)
print(len(intsec_idxs))
'''