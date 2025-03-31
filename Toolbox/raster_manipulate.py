import cv2, os
import numpy as np

# Set GDAL memory buffer to 2GB
os.environ['GDAL_CACHEMAX'] = str(2048 * 1024 * 1024)  # in bytes
from osgeo import gdal, ogr

def open_img(filepath):
    # Open the dataset using GDAL.
    dataset = gdal.Open(filepath)
    if not dataset:
        raise ValueError(f"Failed to open the image file at {filepath}.")
    return dataset

def save_gdal_dataset(dataset, out_filepath):
    """
    Saves an in-memory GDAL dataset to a file on disk (e.g., GeoTIFF).

    :param dataset:       An in-memory GDAL dataset (e.g., from apply_affinetransform).
    :param out_filepath:  Path to the output file (e.g., 'output.tif').
    """
    driver = gdal.GetDriverByName('GTiff')
    # CreateCopy will clone the entire dataset (bands, geotransform, projection, etc.)
    out_ds = driver.CreateCopy(out_filepath, dataset, strict=0, options=['COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS'],)
    # Properly close / flush
    out_ds = None

def affine_trans(homography, points1, points2, in_dataset, target_projection, target_geotransform):
    x_size = in_dataset.RasterXSize
    y_size = in_dataset.RasterYSize
    num_bands = in_dataset.RasterCount
    datatype = in_dataset.GetRasterBand(1).DataType

    corners = np.array([
        [0,       0],
        [x_size,  0],
        [x_size,  y_size],
        [0,       y_size]
    ], dtype=np.float32)

    corners_h = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])  # shape = [4, 3]
    new_corners = (homography @ corners_h.T).T  # shape = [4, 3]

    # Normalize homogeneous coordinates
    new_corners[:, 0] /= new_corners[:, 2]
    new_corners[:, 1] /= new_corners[:, 2]

    min_x = np.min(new_corners[:, 0])
    max_x = np.max(new_corners[:, 0])
    min_y = np.min(new_corners[:, 1])
    max_y = np.max(new_corners[:, 1])

    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))

    # Translation matrix to shift the result so that the new min_x/min_y -> (0,0)
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0,      1]
    ], dtype=np.float64)

    # Compose final transform for warpPerspective
    final_transform = T @ homography
    
    # Compute transformation accuracy via residuals (RMSE in pixel space)
    points2_h = np.concatenate([points2, np.ones((points2.shape[0], 1))], axis=1)  # shape: (N, 3)
    transformed_points2 = (final_transform @ points2_h.T).T  # shape: (N, 3)
    # Normalize by the third (homogeneous) coordinate
    transformed_points2[:, :2] /= transformed_points2[:, 2:3]
    residuals = np.linalg.norm(points1 - transformed_points2[:, :2], axis=1)
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"Transformation Accuracy (RMSE): {rmse:.4f} pixels")

    if rmse > 10:
        return False

    driver = gdal.GetDriverByName('MEM')  # In-memory dataset for output
    out_dataset = driver.Create('',new_width, new_height, num_bands, datatype)

    if rmse > 10:
        return False
    new_geotransform = list(target_geotransform)
    # Shift in pixel space -> shift in georeferenced units
    # new X origin:
    new_geotransform[0] += min_x * target_geotransform[1] + min_y * target_geotransform[2]
    # new Y origin:
    new_geotransform[3] += min_x * target_geotransform[4] + min_y * target_geotransform[5]
    out_dataset.SetGeoTransform(new_geotransform)

    out_dataset.SetProjection(target_projection)
    # Warp each band using OpenCV's warpPerspective and write to output
    for band_index in range(num_bands):
        in_band = in_dataset.GetRasterBand(band_index+1)
        warped_band = cv2.warpPerspective(
            in_band.ReadAsArray(),
            final_transform,
            (new_width, new_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE
        )
        out_band = out_dataset.GetRasterBand(band_index + 1)
        out_band.WriteArray(warped_band)
        out_band.SetNoDataValue(in_band.GetNoDataValue())
        out_band.SetDescription(in_band.GetDescription())
        out_band.FlushCache()
    return out_dataset

def BGR2GRAY(dataset, rgb_serial):
    # Retrieve the first band for nodata info and (if needed) for grayscale data.
    rgb_image = dataset.GetRasterBand(1)
    nodata = rgb_image.GetNoDataValue()
    rgb_image = rgb_image.ReadAsArray()
    # Create nodata (alpha) mask from the first band.
    if nodata is not None:
        alpha_mask = np.where(rgb_image == nodata, 0, 255).astype(np.uint8)
    else:
        alpha_mask = np.full(rgb_image.shape, 255, dtype=np.uint8)

    # Get the shape (height, width) from the first band.
    height, width = rgb_image.shape
    # Initialize an empty RGB image array.
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    # Assign each channel in the RGB image based on the provided band names.
    for i, band_i in enumerate(rgb_serial):
        this_band = dataset.GetRasterBand(band_i)
        rgb_image[:, :, i] = cv2.normalize(this_band.ReadAsArray(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, mask = alpha_mask)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)    ### default as grayscaled in uint8
    img_equalized = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6)).apply(gray_image)
    return img_equalized, alpha_mask

def shp_cut_raster(shp_path, raster_path, output_path):
    # Ensure output format is .img
    if not output_path.lower().endswith(".tif"):
        raise ValueError("Output file must have a .tif extension.")

    # Open raster
    raster_ds = gdal.Open(raster_path)
    if raster_ds is None:
        raise ValueError(f"Failed to open raster file: {raster_path}")

    # Clip raster and save in .img format with lossless compression
    result = gdal.Warp(
        output_path,
        raster_ds,
        format='GTIFF',
        cutlineDSName=shp_path,
        cropToCutline=True,
        options=['COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS']  # No-loss compression
    )

    # Verify and clean up
    if result is None:
        raise ValueError(f"Clipping failed: {raster_path} - {shp_path}")

    raster_ds = None  # Close dataset

def align_raster(base_rst, resample_rst, output):
    # Open the reference raster
    reference_ds = gdal.Open(base_rst)
    reference_proj = reference_ds.GetProjection()  # Projection
    reference_geotrans = reference_ds.GetGeoTransform()  # Geotransform
    reference_width = reference_ds.RasterXSize  # Width in pixels
    reference_height = reference_ds.RasterYSize  # Height in pixels

    # Align the target raster using gdal.Warp
    gdal.Warp(
        output,  # Output file
        resample_rst,  # Input file
        dstSRS=reference_proj,  # Set projection to match reference
        outputBounds=[reference_geotrans[0], 
                    reference_geotrans[3] + reference_geotrans[5] * reference_height, 
                    reference_geotrans[0] + reference_geotrans[1] * reference_width, 
                    reference_geotrans[3]],  # Match extent
        xRes=reference_geotrans[1],  # Match pixel width
        yRes=abs(reference_geotrans[5]),  # Match pixel height
        resampleAlg=gdal.GRA_Lanczos,
        options=['COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS']
    )

def change_detect(base_img_path, img2_path, output_path, method="PER"):
    """
    Perform change detection between two multi-band images and save the percentage difference.
    Only pixels with valid values in both images (and where img1 != 0) are computed.
    
    Parameters:
        base_img_path (str): Path to the first image (baseline).
        img2_path (str): Path to the second image (comparison).
        output_path (str): Path to save the output raster.
    
    Returns:
        None (saves output raster to file)
    """
    # Open datasets
    ds1 = gdal.Open(base_img_path)
    ds2 = gdal.Open(img2_path)
    
    if ds1 is None or ds2 is None:
        raise IOError("One of the images could not be opened.")
    
    # Ensure both images have the same number of bands
    num_bands1 = ds1.RasterCount
    num_bands2 = ds2.RasterCount
    
    if num_bands1 != num_bands2:
        raise ValueError("Images must have the same number of bands.")
    
    # Get metadata for output raster
    driver = gdal.GetDriverByName("GTiff")
    cols = ds1.RasterXSize
    rows = ds1.RasterYSize
    geotransform = ds1.GetGeoTransform()
    projection = ds1.GetProjection()
    
    # Create output raster
    out_ds = driver.Create(output_path, cols, rows, num_bands1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    scale = 1.0
    if method == "PER":
        scale = 100.0
    
    for band_idx in range(1, num_bands1 + 1):  # GDAL bands are 1-based
        band1 = ds1.GetRasterBand(band_idx)
        band2 = ds2.GetRasterBand(band_idx)
        
        arr1 = band1.ReadAsArray().astype(np.float32)
        arr2 = band2.ReadAsArray().astype(np.float32)
        
        nodata1 = band1.GetNoDataValue()
        nodata2 = band2.GetNoDataValue()
        
        # Create valid data mask
        valid_mask = np.ones(arr1.shape, dtype=bool)
        if nodata1 is not None:
            valid_mask &= (arr1 != nodata1)
        if nodata2 is not None:
            valid_mask &= (arr2 != nodata2)
        
        valid_mask &= (arr1 != 0)  # Avoid division by zero
        
        # Initialize output array with NaN
        percentage_change = np.full(arr1.shape, np.nan, dtype=np.float32)
        
        # Compute change where valid
        percentage_change[valid_mask] = scale * (arr2[valid_mask] - arr1[valid_mask]) / arr1[valid_mask]
        
        # Write to output raster
        out_band = out_ds.GetRasterBand(band_idx)
        out_band.WriteArray(percentage_change)
        out_band.SetNoDataValue(np.nan)  # Set NoData value
        out_band.SetDescription(band1.GetDescription())
        out_band.FlushCache()
    
    # Close datasets
    ds1, ds2, out_ds = None, None, None

def freq_summary(img_path):
    # Open dataset
    ds = gdal.Open(img_path)
    if ds is None:
        raise IOError("Input image could not be opened.")
    stats_list = []
    # Iterate over each band
    for band_idx in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(band_idx)
        band_data = band.ReadAsArray()
        band_name = band.GetDescription()
        if band_data is None:
            stats_list.append([None] * 8)
            continue
        nodata_value = band.GetNoDataValue()
        if nodata_value is not None:
            band_data = band_data[band_data != nodata_value]  # Remove NoData values
        else:
            band_data = band_data[~np.isnan(band_data)]  # Remove NaN values
        if band_data.size == 0:
            stats_list.append([None] * 8)
            continue
        # Compute all statistics efficiently in one pass
        # min_val = np.min(band_data)
        # max_val = np.max(band_data)
        mean_val = np.mean(band_data)
        # median_val = np.median(band_data)
        std_val = np.std(band_data)
        min_val, first_quantile, median_val, third_quantile, max_val = np.percentile(band_data, [0, 25, 50, 75, 100])
        stats = (band_name, mean_val, std_val, min_val, first_quantile, median_val, third_quantile, max_val)
        stats_list.append(stats)
    ds = None  # Close dataset
    return stats_list