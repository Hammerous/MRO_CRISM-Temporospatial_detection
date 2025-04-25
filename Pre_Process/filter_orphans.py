import geopandas as gpd

# Replace 'your_input_file.shp' with the path to your shapefile
gdf = gpd.read_file('InAngle65_poly.shp')

# Perform self spatial join using intersects
joined = gpd.sjoin(gdf, gdf, how="inner", predicate="intersects")

# Remove self-intersections (same geometry matched to itself)
joined = joined[joined.index != joined["index_right"]]

# Get unique indices of polygons that intersect others
intersecting_indices = joined.index.unique()

# Filter original GeoDataFrame
intersecting_gdf = gdf.loc[intersecting_indices]

intersecting_gdf.to_file("InAngle65_intsct.shp")