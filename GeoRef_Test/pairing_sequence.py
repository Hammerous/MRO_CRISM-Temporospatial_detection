import cv2, os
import numpy as np
from osgeo import gdal

# Optionally set GDAL cache size and warp memory (e.g., half of available free RAM)
# For example, suppose you calculate free_mem_mb dynamically:
os.environ['GDAL_CACHEMAX'] = str(2048 * 1024 * 1024)  # in bytes

def process_geotiff(image_path, rgb_bands=None):
    """
    Process an image file (.img or .tif) and return a grayscale image, a nodata (alpha) mask,
    and the original GDAL dataset.

    For .img files:
      - The image is assumed to have multiple bands.
      - The `rgb_bands` parameter may be provided as a tuple/list of three elements.
        Each element should be:
            a string specifying the band name. In that case the function will search the header
            information (the band description) to determine the appropriate band index.
      - The selected bands are read and combined into an RGB image.
      - The resulting grayscale image is then normalized (using only valid data as defined by the
        nodata mask) to the 0–255 range and converted to uint8.

    For .tif files:
      - The image is assumed to be single-band.
      - The band data is normalized (using only valid data as defined by the nodata mask)
        to the 0–255 range and converted to uint8.

    In both cases, a nodata mask is created based on the nodata value of the first band:
      - Pixels equal to the nodata value are set to 0 (transparent).
      - All other pixels are set to 255 (opaque).
      - If no nodata value is defined, the mask is set to fully opaque.

    Parameters:
        image_path (str): Path to the input image file (.img or .tif).
        rgb_bands (tuple of length 3, optional): For .img files, specifies either the band indices
            (as integers) or the band names (as strings) for the Red, Green, and Blue channels.
            This parameter is ignored for .tif files.

    Returns:
        gray_image (np.ndarray): The processed grayscale image in uint8 format.
        alpha_mask (np.ndarray): The nodata mask as an 8-bit image (0 for nodata, 255 for valid data).
        dataset (gdal.Dataset): The original GDAL dataset.
    """
    # Open the dataset using GDAL.
    dataset = gdal.Open(image_path)
    if not dataset:
        raise ValueError(f"Failed to open the image file at {image_path}.")

    # Determine file extension.
    ext = os.path.splitext(image_path)[-1].lower()

    # Retrieve the first band for nodata info and (if needed) for grayscale data.
    rgb_image = dataset.GetRasterBand(1)
    nodata = rgb_image.GetNoDataValue()
    rgb_image = rgb_image.ReadAsArray()
    # Create nodata (alpha) mask from the first band.
    if nodata is not None:
        alpha_mask = np.where(rgb_image == nodata, 0, 255).astype(np.uint8)
    else:
        alpha_mask = np.full(rgb_image.shape, 255, dtype=np.uint8)

    if ext == '.img':
        # Get the shape (height, width) from the first band.
        height, width = rgb_image.shape
        # Initialize an empty RGB image array.
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        # Assign each channel in the RGB image based on the provided band names.
        for i in range(1, dataset.RasterCount + 1):
            this_band = dataset.GetRasterBand(i)
            desc = this_band.GetDescription()
            if desc in rgb_bands:
                rgb_image[:, :, rgb_bands.index(desc)] = cv2.normalize(this_band.ReadAsArray(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, mask = alpha_mask)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)    ### default as grayscaled in uint8
    elif ext == '.tif':
        # For .tif files, assume the image is single-band.
        gray_image = cv2.normalize(rgb_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, mask = alpha_mask).astype(np.uint8)
    else:
        raise ValueError("Unsupported file format. Only .img and .tif files are supported.")
    
    img_equalized = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6,6)).apply(gray_image)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # tophat = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
    # bottomhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
    # img_corrected = cv2.add(gray_image, tophat) - bottomhat

    return img_equalized, alpha_mask, dataset

def pixel_to_geographic(points, geotransform):
    """
    Converts pixel coordinates to geographical coordinates using the geotransform.
    Args:
        points (numpy.ndarray): Pixel coordinates as an array of shape (n, 2) where each row is (x, y).
        geotransform (tuple): Geotransform of the image.
    Returns:
        numpy.ndarray: Geographical coordinates as an array of shape (n, 2) where each row is (longitude, latitude).
    """
    # Extract geotransform components
    x_origin, pixel_width, x_rotation, y_origin, y_rotation, pixel_height = geotransform

    # Convert pixel coordinates to geographic coordinates using vectorized operations
    geo_x = x_origin + points[:, 0] * pixel_width + points[:, 1] * x_rotation
    geo_y = y_origin + points[:, 0] * y_rotation + points[:, 1] * pixel_height

    # Stack the results into an (n, 2) array
    geo_coords = np.column_stack((geo_x, geo_y))

    return geo_coords

def apply_georeferencing(in_ds, points_image, points_map, new_prj, result_path, resample=gdal.GRA_Lanczos, compression='LZW', num_threads='ALL_CPUS'):
    """
    Use GDAL's GCP-based warping to georeference an image based on matched control points.
    Splits GCPs into 80% for warping, and 20% for verification, ensuring an approximately even
    spatial distribution across the image.

    :param in_ds: GDAL dataset (from gdal.Open()) representing the raster image.
    :param points_image: An Nx2 array (or list) of pixel coordinates [(px, py), ...] in the image.
    :param points_map: An Nx2 array (or list) of map coordinates [(mx, my), ...] corresponding
                       to the pixel coordinates.
    :param new_prj: The projection for the output dataset (e.g., 'EPSG:4326').
    :param result_path: Path to save the georeferenced result.
    :param resample: Resampling algorithm (eg. gdal.GRA_NearestNeighbour/gdal.GRA_Lanczos)
    :param compression: The compression method (default is 'LZW') for output file.
    :param num_threads: Number of threads for parallel processing, or 'ALL_CPUS' to use all 
                        available CPUs (default is 'ALL_CPUS').
    :param grid_size: The number of grid cells along each axis for splitting points into 
                      warp/verify subsets evenly across the image (default = 5).
    :return: None
    """
    points_image = np.asarray(points_image, dtype=np.float64)  # shape: (N, 2) with columns [px, py]
    points_map   = np.asarray(points_map, dtype=np.float64)
    combined = np.hstack([points_image, points_map])
    gcps_warp = [gdal.GCP(mx, my, 0.0, px, py) for px, py, mx, my in combined]
    # gcps_warp = [gdal.GCP(float(mx), float(my), 0.0, float(px), float(py))
    #         for (px, py), (mx, my) in zip(points_image, points_map)]

    # Assign GCPs (warp subset) and set the coordinate system
    print(in_ds.GetGeoTransform())
    in_ds.SetGCPs(gcps_warp, new_prj)

    # Define warp options
    warp_options = gdal.WarpOptions(
        dstSRS=new_prj,                     # Output spatial reference system
        format='GTiff',                     # Output as GeoTIFF
        tps=True,                           # Use thin plate spline transformation
        errorThreshold=1,                 # Lower error threshold for better accuracy (in pixels)
        warpMemoryLimit=1024,               # Working buffer size (in MB)
        resampleAlg=resample,       # Resampling algorithm
        creationOptions=['COMPRESS=' + compression, f'NUM_THREADS={num_threads}'],
        multithread=True
    )

    # Perform the warp and save the result
    try:
        gdal.Warp(result_path, in_ds, options=warp_options)
    except Exception as e:
        print(f"Error during warp operation: {e}")
        return
    
    print(in_ds.GetGeoTransform())
    output_ds = gdal.Open(result_path)
    geo_transform = output_ds.GetGeoTransform()
    print(geo_transform)
    if geo_transform is None:
        print("GeoTransform could not be retrieved from the warped dataset.")
        return

    residuals = []
    for gcp in gcps_warp:
        # The original map coords
        mx, my = gcp.GCPX, gcp.GCPY

        # Convert map coords -> pixel/line in the warped image
        #   px = (mx - gt[0]) / gt[1]
        #   py = (my - gt[3]) / gt[5]
        px_warped = (mx - geo_transform[0]) / geo_transform[1]
        py_warped = (my - geo_transform[3]) / geo_transform[5]

        # The original pixel coords for this check GCP
        px_orig = gcp.GCPPixel
        py_orig = gcp.GCPLine

        # Residual in pixel space
        px_diff = px_orig - px_warped
        py_diff = py_orig - py_warped
        residuals.append(np.sqrt(px_diff**2 + py_diff**2))

    if len(residuals) == 0:
        print("No points in the verification subset. Cannot compute accuracy.")
        return

    avg_residual = np.mean(residuals)
    max_residual = np.max(residuals)

    print(f"Number of GCPs used for warping: {len(gcps_warp)}")
    #print(f"Number of GCPs used for checking: {len(gcps_check)}")
    print(f"Average residual error (check subset) = {avg_residual:.2f} pixels")
    print(f"Max residual error (check subset) = {max_residual:.2f} pixels")

    # Cleanup
    del in_ds, output_ds

def feature_matching_SIFT(img1, valid_mask1, img2, valid_mask2, output_file='feature_matches_SIFT.png',
                          ratio_threshold=0.75, ransac_reproj_threshold=10.0):
    """
    Extracts features from both images using the SIFT descriptor
    and matches the features using a FLANN-based matcher with improved parameter selection.
    Removes keypoints near NoData areas and applies advanced filtering techniques, including RANSAC.
    """
    # Initialize the SIFT detector with optimized parameters
    sift = cv2.SIFT_create(
        nfeatures=0,              # Auto-detect optimal number of features
        nOctaveLayers=30,          # Increase from default (3) for better scale-space representation
        contrastThreshold=0.01,   # Prevent too many weak keypoints
        #edgeThreshold=10,         # Avoid unreliable edge responses
        sigma=1.0,                # Standard value for scale-space pyramid
        #descriptorType=cv2.CV_32F # Use 32-bit float descriptors for better precision
    )

    # Detect keypoints and descriptors in both images after masking
    kp1, des1 = sift.detectAndCompute(img1, valid_mask1)
    kp2, des2 = sift.detectAndCompute(img2, valid_mask2)
    print(f"Keypoints detected: {len(kp1)} in image 1, {len(kp2)} in image 2")

    # FLANN parameters for efficient matching
    index_params = dict(algorithm=1, trees=8)  # Use KDTree for SIFT descriptors
    search_params = dict(checks=500)          # Increase checks for better accuracy

    # Use FLANN-based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)  # Find the 2 nearest neighbors for each descriptor

    # Apply Lowe's ratio test to filter good matches
    filtered_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]

    # Sort the filtered matches by distance (lower distance = better match)
    filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)

    # Extract matched keypoint coordinates
    points1 = np.array([kp1[m.queryIdx].pt for m in filtered_matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in filtered_matches], dtype=np.float32)

    # Apply RANSAC-based filtering for robustness
    if len(points1) >= 4:  # RANSAC requires at least 4 points
        homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_reproj_threshold)
        if mask is not None:
            mask = mask.ravel().astype(bool)
            points1, points2 = points1[mask], points2[mask]
            filtered_matches = [m for i, m in enumerate(filtered_matches) if mask[i]]
    else:
        raise ValueError(f"Insufficient points to perform RANSAC filtering: {points1}.")

    # Remove duplicate points
    unique_points1, unique_indices1 = np.unique(points1, axis=0, return_index=True)
    points1, points2 = points1[unique_indices1], points2[unique_indices1]

    unique_points2, unique_indices2 = np.unique(points2, axis=0, return_index=True)
    points1, points2 = points1[unique_indices2], points2[unique_indices2]

    # Draw the matches between the images if output_file is specified
    if output_file:
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, filtered_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(output_file, img_matches)

    print(f"Final number of matches after RANSAC filtering: {points1.shape[0]}       ")
    return points1, points2

def georeferencing(result_path, base_img_path, warp_img_path, warp_rgb):
    # Step 1: Read the images
    img1_gray, img1_alpha, base_img = process_geotiff(base_img_path, warp_rgb)
    img2_gray, img2_alpha, img2prj = process_geotiff(warp_img_path, warp_rgb)
    #cv2.imwrite('img1_gray.png', img1_gray)
    #cv2.imwrite('img2_gray.png', img2_gray)

    # Step 2: Feature matching
    points1, points2 = feature_matching_SIFT(img1_gray, img1_alpha, img2_gray, img2_alpha)
    del img1_gray, img1_alpha, img2_gray, img2_alpha

    # Step 3: Apply georeferencing and save
    points_map = pixel_to_geographic(points1, base_img.GetGeoTransform())
    apply_georeferencing(img2prj, points2,\
                        points_map, base_img.GetProjection(),\
                        result_path)

if __name__ == "__main__":
    base_img_path = r'CTX_DEM_Retrieve\ortho_clipped.tif'
    warp_img_path = r'Playground\frt000161ef_07_sr167j_mtr3.img'

    base_img_path = r'CTX_DEM_Retrieve\ortho_clipped_2.tif'
    warp_img_path = r'Playground\frt000174f4_07_sr166j_mtr3.img'

    # base_img_path = r'Playground\frt000161ef_07_sr167j_mtr3.img'
    # warp_img_path = r'Playground\frt000174f4_07_sr166j_mtr3.img'

    warp_rgb = ('R2529', 'R1506', 'R1080')
    result_path = r'CTX_DEM_Retrieve\orthorecitified_test2_2.tif'
    georeferencing(result_path, base_img_path, warp_img_path, warp_rgb)