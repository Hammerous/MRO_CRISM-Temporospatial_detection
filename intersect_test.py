import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# 创建示例多边形
data = {
    "geometry": [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),  # 多边形 1
        Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),  # 多边形 2
        Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])  # 多边形 3
    ]
}

# 创建 GeoDataFrame
gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

# 计算最大公共相交区域
def compute_max_overlap(geometries):
    """递归计算最大公共相交区域"""
    if len(geometries) == 0:
        return None
    overlap = geometries[0]
    for geom in geometries[1:]:
        overlap = overlap.intersection(geom)
        if overlap.is_empty:
            return None
    return overlap

max_overlap = compute_max_overlap(list(gdf.geometry))

# 可视化
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制单个多边形
for i, row in gdf.iterrows():
    gpd.GeoSeries(row.geometry).plot(ax=ax, color=f"C{i}", alpha=0.5, label=f"Polygon {i + 1}")

# 绘制最大公共相交区域
if max_overlap and not max_overlap.is_empty:
    gpd.GeoSeries(max_overlap).plot(ax=ax, color="red", alpha=0.7, label="Max Overlap")

# 添加图例和设置
ax.legend()
ax.set_title("Polygons and Their Maximum Overlap Area")
plt.show()
