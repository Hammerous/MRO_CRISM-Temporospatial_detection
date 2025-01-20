from osgeo import gdal, ogr
import numpy as np

# Input and output paths
input_raster = r'Playground\frt000161ef_07_sr167j_mtr3.img'
output_vector = "output_polygons.shp"

# Open the raster dataset
raster_dataset = gdal.Open(input_raster)

# Get the raster band
band = raster_dataset.GetRasterBand(1)

# Create the output shapefile
driver = ogr.GetDriverByName("ESRI Shapefile")
output_ds = driver.CreateDataSource(output_vector)
srs = ogr.osr.SpatialReference()
srs.ImportFromWkt(raster_dataset.GetProjection())
layer = output_ds.CreateLayer("polygonized", srs=srs, geom_type=ogr.wkbPolygon)

# Create a mask band to exclude NoData pixels
mask_band = band.GetMaskBand()

# Polygonize the raster, ignoring NaN values (nodata)
gdal.Polygonize(band, mask_band, layer, -1, options=["8CONNECTED"], callback=None)

# Close the datasets
layer = None
output_ds = None
raster_dataset = None

print(f"Polygonized shapefile saved at: {output_vector}")
