from osgeo import gdal

def print_band_index_by_name(dataset):
    for bidx in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(bidx)
        band_name = band.GetDescription()
        print("{0}: {1}".format(bidx, band_name))
    return None  # Not found

ds = gdal.Open(r'G:\MTRDR_filtered\3-0\frt0000a8ce_07_sr166j_mtr3.tif')
index = print_band_index_by_name(ds)
