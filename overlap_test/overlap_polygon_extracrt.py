import os
from osgeo import gdal, ogr

# Input folder and output paths
input_folder = r'Playground'
output_vector = "output_polygons.shp"

# Get all .img files in the folder
img_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.img')]

if not img_files:
    print("No .img files found in the folder.")
    exit()

# Create the output shapefile for storing intersected polygons
driver = ogr.GetDriverByName("ESRI Shapefile")
output_ds = driver.CreateDataSource(output_vector)
srs = None  # Placeholder for spatial reference
layer = None

# Function to polygonize a raster
def polygonize_raster(img_file):
    # Open the raster dataset
    raster_dataset = gdal.Open(img_file)
    band = raster_dataset.GetRasterBand(1)
    mask_band = band.GetMaskBand()

    # Temporary shapefile for vectorized output
    temp_vector = f"{img_file}_temp.shp"
    temp_ds = driver.CreateDataSource(temp_vector)
    global srs
    if not srs:
        srs = ogr.osr.SpatialReference()
        srs.ImportFromWkt(raster_dataset.GetProjection())
    temp_layer = temp_ds.CreateLayer("temp_layer", srs=srs, geom_type=ogr.wkbPolygon)

    # Polygonize
    gdal.Polygonize(band, mask_band, temp_layer, -1, options=["8CONNECTED"], callback=None)
    temp_ds = None
    raster_dataset = None
    return temp_vector

# Vectorize each raster and compute the intersection
intersect_geom = None

for img_file in img_files:
    print(f"Processing: {img_file}")
    temp_vector = polygonize_raster(img_file)

    # Open the temporary shapefile
    temp_ds = ogr.Open(temp_vector)
    temp_layer = temp_ds.GetLayer()

    # Merge all polygons in the temporary shapefile
    temp_union = None
    for feature in temp_layer:
        geom = feature.GetGeometryRef()
        if temp_union is None:
            temp_union = geom.Clone()
        else:
            temp_union = temp_union.Union(geom)

    # Compute intersection with the accumulated geometry
    if intersect_geom is None:
        intersect_geom = temp_union.Clone()
    else:
        intersect_geom = intersect_geom.Intersection(temp_union)

    temp_ds = None
    os.remove(temp_vector)
    os.remove(temp_vector.replace(".shp", ".shx"))
    os.remove(temp_vector.replace(".shp", ".dbf"))

# Save the resulting intersection geometry
if intersect_geom is not None:
    print("Saving the largest intersecting polygon.")
    output_layer = output_ds.CreateLayer("intersected", srs=srs, geom_type=ogr.wkbPolygon)
    output_feature = ogr.Feature(output_layer.GetLayerDefn())
    output_feature.SetGeometry(intersect_geom)
    output_layer.CreateFeature(output_feature)

output_ds = None
print(f"Intersected polygons saved at: {output_vector}")
