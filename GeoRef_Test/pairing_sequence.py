import cv2, os
import numpy as np
from osgeo import gdal

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

    return gray_image, alpha_mask, dataset

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

def apply_georeferencing(in_ds, points_image, points_map, new_prj, result_path, compression='LZW', num_threads='ALL_CPUS'):
    """
    Use GDAL's GCP-based warping to georeference an image based on matched control points.
    
    The input image is provided as a GDAL dataset (e.g., from gdal.Open()). This function 
    creates an in-memory copy of the dataset, assigns the provided ground control points (GCPs),
    and then applies a warp using a thin plate spline interpolation.
    
    :param in_ds: GDAL dataset (from gdal.Open()) representing the raster image.
    :param points_image: An Nx2 array (or list) of pixel coordinates [(px, py), ...] in the image.
    :param points_map: An Nx2 array (or list) of map coordinates [(mx, my), ...] corresponding to the pixel coordinates (e.g., in EPSG:XXXX).
    :param new_prj: The projection for the output dataset (e.g., 'EPSG:4326').
    :param result_path: Path to save the georeferenced result.
    :param compression: The compression method (default is 'LZW') for output file.
    :param num_threads: Number of threads for parallel processing, or 'ALL_CPUS' to use all available CPUs (default is 'ALL_CPUS').
    :return: None
    """
    # Build the list of GCPs from the matched points
    # points_image -> pixel/line
    # points_map -> map X, Y
    gcps = [gdal.GCP(float(mx), float(my), 0.0, float(px), float(py))
            for (px, py), (mx, my) in zip(points_image, points_map)]

    # Assign GCPs and set the coordinate system that these map coords belong to
    in_ds.SetGCPs(gcps, new_prj)

    # Define warp options with compression, multithreading, and NUM_THREADS
    warp_options = gdal.WarpOptions(
        dstSRS=new_prj,                     # Output spatial reference system
        format='GTiff',                     # Output as GeoTIFF
        tps=True,                           # Use thin plate spline transformation
        errorThreshold = 1,               # error threshold for approximation transformer (in pixels)
        warpMemoryLimit = 1024,             # size of working buffer in MB
        #resampleAlg=gdal.GRA_Lanczos,       # Resampling algorithm
        resampleAlg=gdal.GRA_NearestNeighbour,
        creationOptions=['COMPRESS=' + compression, f'NUM_THREADS={num_threads}'],  # Set compression method and thread usage
    )

    # Perform the warp and save the result to the provided path
    gdal.Warp(result_path, in_ds, options=warp_options)

    # Cleanup
    del in_ds

def feature_matching_SIFT(img1, valid_mask1, img2, valid_mask2, output_file='feature_matches_SIFT.png',
                          ratio_threshold=0.7, distance_percentile=0.2, ransac_reproj_threshold=10.0):
    """
    Extracts features from both images using the SIFT descriptor
    and matches the features using a FLANN-based matcher with improved parameter selection.
    Removes keypoints near NoData areas and applies advanced filtering techniques, including RANSAC.
    """
    # Initialize the SIFT detector with optimized parameters
    sift = cv2.SIFT_create(
        nfeatures=10000,       # Increase the number of features detected
        contrastThreshold=0.01,  # Adjust to capture more relevant keypoints
        edgeThreshold=10,        # Lower value to detect finer features
        sigma=1.2                # Gaussian kernel standard deviation
    )

    # Detect keypoints and descriptors in both images after masking
    kp1, des1 = sift.detectAndCompute(img1, valid_mask1)
    kp2, des2 = sift.detectAndCompute(img2, valid_mask2)
    print(f"Keypoints detected: {len(kp1)} in image 1, {len(kp2)} in image 2")

    # FLANN parameters for efficient matching
    index_params = dict(algorithm=1, trees=5)  # Use KDTree for SIFT descriptors
    search_params = dict(checks=200)          # Increase checks for better accuracy

    # Use FLANN-based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)  # Find the 2 nearest neighbors for each descriptor

    # Apply Lowe's ratio test to filter good matches
    filtered_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]

    # Further filter matches based on a distance threshold
    distances = np.array([m.distance for m in filtered_matches])
    if len(distances) > 0:  # Ensure there are matches to process
        distance_threshold = np.percentile(distances, distance_percentile * 100)  # Calculate threshold based on percentile
        filtered_matches = [m for m in filtered_matches if m.distance <= distance_threshold]

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

    print(f"\rFinal number of matches after RANSAC filtering: {points1.shape[0]}       ", end='')
    return points1, points2

def georeferencing(result_path, base_img_path, warp_img_path, warp_rgb):
    # Step 1: Read the images
    #img1_gray, img1_alpha, base_img = process_geotiff(base_img_path)
    img1_gray, img1_alpha, base_img = process_geotiff(base_img_path, warp_rgb)
    img2_gray, img2_alpha, img2prj = process_geotiff(warp_img_path, warp_rgb)
    cv2.imwrite('img1_gray_1.png', img1_gray)
    cv2.imwrite('img2_gray_1.png', img2_gray)

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

    base_img_path = r'Playground\frt000161ef_07_sr167j_mtr3.img'
    warp_img_path = r'Playground\frt000174f4_07_sr166j_mtr3.img'

    warp_rgb = ('R2529', 'R1506', 'R1080')
    result_path = r'CTX_DEM_Retrieve\orthorecitified_test1_1.tif'
    georeferencing(result_path, base_img_path, warp_img_path, warp_rgb)