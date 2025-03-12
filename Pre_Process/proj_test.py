import os
from osgeo import ogr, osr

def find_overlapping_polygons(layer):
    # 初始化一个字典来存储重叠多边形的索引集合
    intsec_idx = set()
    
    # 获取图层中的所有要素
    layer.ResetReading()
    features = [feature.Clone() for feature in layer]
    
    # 遍历图层中的每个多边形
    for i, feature in enumerate(features):
        print(f"\r{i}         ", end='')
        geom1 = feature.GetGeometryRef()
        
        # 查找与当前多边形相交的多边形
        intsec_polys = []
        for j, other_feature in enumerate(features):
            if i != j:
                geom2 = other_feature.GetGeometryRef()
                if geom1.Intersects(geom2):
                    intsec_polys.append(j)
        
        # 如果存在重叠多边形，将集合添加到字典中
        if intsec_polys:
            intsec_polys.append(i)
            intsec_idx.add(frozenset(intsec_polys))
    
    return intsec_idx

# 打开shapefile
driver = ogr.GetDriverByName("ESRI Shapefile")
merged_ds = driver.Open(r"mars_mro_crism_mtrdr_c0a\mars_mro_crism_mtrdr_c0a.shp", 0)  # 0 means read-only
merged_layer = merged_ds.GetLayer()

print("Shp File Loaded !!!\nFinding intersecting polygons ...")
intsec_idxs = find_overlapping_polygons(merged_layer)
print(len(intsec_idxs))