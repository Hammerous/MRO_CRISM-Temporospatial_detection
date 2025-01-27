import cv2
import numpy as np
from osgeo import osr, gdal

def read_geotiff(image_path):
    """
    Reads a GeoTIFF image and returns the image, geotransform, CRS, and projection information.

    Parameters:
    - image_path: Path to the GeoTIFF file.

    Returns:
    - rgb_image: The RGB image data as a NumPy array. For single-band images, this will contain the band data.
    - alpha_channel: The alpha channel data if present, otherwise None.
    - geotransform: The geotransform of the image, defining its spatial resolution and origin.
    - crs: The spatial reference system as an OSR SpatialReference object.
    - projection: The projection string in WKT format.
    """
    # Open the GeoTIFF file
    dataset = gdal.Open(image_path)
    if not dataset:
        raise ValueError(f"Failed to open the GeoTIFF file at {image_path}.")

    # Retrieve geotransform and projection information
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    crs = osr.SpatialReference()
    crs.ImportFromWkt(projection)

    # Read the image data
    image = dataset.ReadAsArray()

    # Separate RGBA channels if applicable
    if image.ndim == 3 and image.shape[0] == 4:  # If the image has 4 bands (RGBA)
        rgba_image = image
        rgb_image = rgba_image[:3, :, :]  # Keep only the RGB channels
        alpha_channel = rgba_image[3, :, :]  # Separate the alpha channel
    else:
        rgb_image = image
        alpha_channel = None

    return rgb_image, alpha_channel, geotransform, crs, projection

def apply_georeferencing_cv(points1, points2, image_to_adjust, geotransform):
    """
    Applies affine transformation to all bands of the reprojected image based on the matched points.
    Georeferencing is updated accordingly.
    """
    print("{0} Points for Georeference Detected".format(points1.shape[0]))
    # Get the number of bands and image dimensions
    if image_to_adjust.ndim == 3:
        num_bands, height, width = image_to_adjust.shape
    elif image_to_adjust.ndim == 2:  # Single-band (grayscale)
        num_bands = 1
        height, width = image_to_adjust.shape
        image_to_adjust = image_to_adjust[np.newaxis, :, :]  # Add a band dimension for uniformity

    # Find the homography matrix using the matched points
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Compute the residuals for accuracy assessment
    points2_h = np.concatenate([points2, np.ones((points2.shape[0], 1))], axis=1).T
    transformed_points = np.dot(h, points2_h).T
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2:3]  # Normalize by the third coordinate
    residuals = np.linalg.norm(points1 - transformed_points, axis=1)
    rmse = np.sqrt(np.mean(residuals**2))
    print("Transformation Accuracy (RMSE): {:.4f} pixels".format(rmse))

    # Initialize a container for the adjusted image
    adjusted_image = np.zeros_like(image_to_adjust)

    # Apply the homography to each band
    for band in range(num_bands):
        adjusted_image[band] = cv2.warpPerspective(image_to_adjust[band], h, (width, height), \
                                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)

    # Update the geotransform
    new_geotransform = list(geotransform)
    new_geotransform[0] = geotransform[0] + h[0, 2]  # Update x-origin
    new_geotransform[3] = geotransform[3] + h[1, 2]  # Update y-origin

    return adjusted_image, new_geotransform

def apply_georeferencing_poly2(points1, points2, image_to_adjust, geotransform, polynomial_order=2):
    from scipy.interpolate import griddata
    """
    Applies polynomial transformation to all bands of the reprojected image 
    based on the matched points. Georeferencing is updated accordingly.
    """    
    # Validate input points
    if len(points1) < 6 or len(points2) < 6:
        raise ValueError("At least 6 points are required for a robust polynomial transformation.")
    else:
        print("{0} Points for Georeference Detected".format(points1.shape[0]))
    
    # Get the number of bands and image dimensions
    if image_to_adjust.ndim == 3:
        num_bands, height, width = image_to_adjust.shape
    elif image_to_adjust.ndim == 2:  # Single-band (grayscale)
        num_bands = 1
        height, width = image_to_adjust.shape
        image_to_adjust = image_to_adjust[np.newaxis, :, :]  # Add a band dimension for uniformity
    
    # Fit polynomial transformation model
    coeff_x = np.polyfit(points2[:, 0], points1[:, 0], polynomial_order)
    coeff_y = np.polyfit(points2[:, 1], points1[:, 1], polynomial_order)
    
    # Generate grid for the output image
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    flat_grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    
    # Apply the polynomial transformation to map output coordinates back to input
    transformed_x = (
        coeff_x[0] * flat_grid[:, 0]**2 +
        coeff_x[1] * flat_grid[:, 0] +
        coeff_x[2]
    )
    transformed_y = (
        coeff_y[0] * flat_grid[:, 1]**2 +
        coeff_y[1] * flat_grid[:, 1] +
        coeff_y[2]
    )
    transformed_coords = np.stack([transformed_x, transformed_y], axis=-1)
    
    # Initialize a container for the adjusted image
    adjusted_image = np.zeros_like(image_to_adjust, dtype=image_to_adjust.dtype)
    
    # Interpolate each band using griddata
    for band in range(num_bands):
        flat_image = image_to_adjust[band].ravel()
        adjusted_image[band] = griddata(
            transformed_coords,
            flat_image,
            (xx, yy),
            method='nearest',
            fill_value=0
        )
    
    # Update the geotransform based on the polynomial offsets
    new_geotransform = list(geotransform)
    new_geotransform[0] += coeff_x[2]  # Update x-origin
    new_geotransform[3] += coeff_y[2]  # Update y-origin

    return adjusted_image, new_geotransform

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

def apply_georeferencing(points_image, 
                              points_map, 
                              image_array, 
                              original_geotransform, 
                              projection, 
                              transform_type='tps'):
    """
    Use GDAL's GCP-based warping to georeference an image based on matched control points.

    :param points_image: Nx2 array of pixel coordinates [(px, py), ...] in `image_array`.
    :param points_map: Nx2 array of map coordinates [(mx, my), ...] (e.g., in EPSG:XXXX).
    :param image_array: 2D or 3D numpy array (bands, rows, cols) of the raster data.
    :param original_geotransform: A 6-element geotransform for the unreferenced image 
                                  (often placeholder).
    :param projection: Well-Known Text or EPSG code for the coordinate system 
                       of `points_map`.
    :param transform_type: Transformation type for warping. Options:
                           'tps' = thin plate spline (flexible, non-linear)
                           'polynomial' = polynomial of a certain order (rigid, linear)
                           Default: 'tps'.

    :return: (warped_array, new_geotransform)
    """
    # Ensure image_array is in (bands, rows, cols) format
    if image_array.ndim == 2:
        # Single band: reshape to (1, rows, cols)
        image_array = image_array[np.newaxis, ...]
    num_bands, height, width = image_array.shape

    # Create an in-memory GDAL dataset to hold the input image
    mem_driver = gdal.GetDriverByName('MEM')
    in_ds = mem_driver.Create('', width, height, num_bands, gdal.GDT_Float32)

    # Set the (placeholder) geotransform and projection
    in_ds.SetGeoTransform(original_geotransform)
    in_ds.SetProjection(projection)  # Set projection (or keep empty if unknown)

    # Write the image_array into the GDAL dataset
    for b in range(num_bands):
        in_band = in_ds.GetRasterBand(b + 1)
        in_band.WriteArray(image_array[b])

    # Build the list of GCPs from the matched points
    # points_image -> pixel/line
    # points_map   -> map X, Y
    gcps = []
    for (px, py), (mx, my) in zip(points_image, points_map):
        gcps.append(gdal.GCP(float(mx), float(my), 0.0, float(px), float(py)))

    # Assign GCPs and set the coordinate system that these map coords belong to
    in_ds.SetGCPs(gcps, projection)

    # Handle transformation method selection
    if transform_type == 'tps':
        warp_options = gdal.WarpOptions(
            dstSRS=projection,             # Output spatial reference
            format='MEM',                  # In-memory output
            tps=True,                      # Thin plate spline method
            resampleAlg=gdal.GRA_NearestNeighbour  # Interpolation (can be adjusted)
        )
    elif transform_type == 'polynomial':
        warp_options = gdal.WarpOptions(
            dstSRS=projection,             # Output spatial reference
            format='MEM',                  # In-memory output
            polynomialOrder=2,             # Polynomial order (can be adjusted)
            resampleAlg=gdal.GRA_NearestNeighbour  # Interpolation (can be adjusted)
        )
    else:
        raise ValueError("Invalid transform_type. Choose 'tps' or 'polynomial'.")

    # Perform the warp in memory
    out_ds = gdal.Warp('', in_ds, options=warp_options)

    # Extract the warped raster as a numpy array
    warped_width = out_ds.RasterXSize
    warped_height = out_ds.RasterYSize
    warped_bands = out_ds.RasterCount

    warped_array = np.zeros((warped_bands, warped_height, warped_width), dtype=np.float32)
    for b in range(warped_bands):
        out_band = out_ds.GetRasterBand(b + 1)
        warped_array[b, :, :] = out_band.ReadAsArray()

    # Get the new geotransform
    new_geotransform = out_ds.GetGeoTransform()

    # Cleanup
    in_ds = None
    out_ds = None

    return warped_array, new_geotransform

def save_georeferenced_image(output_path, adjusted_image, geotransform, crs, nodata_value=0):
    """
    Saves the georeferenced image as a GeoTIFF with the same CRS and geotransform.
    Dynamically generates an alpha channel based on nodata areas in the processed image.
    """
    # Get the image dimensions
    bands, rows, cols = adjusted_image.shape

    # Dynamically create an alpha channel based on nodata areas
    # A pixel is set to 0 in the alpha channel if all bands at that pixel have the nodata_value
    alpha_channel = np.all(adjusted_image == nodata_value, axis=0).astype(np.uint8)
    alpha_channel = np.where(alpha_channel == 1, 0, 255)  # Convert to alpha format (0 for transparent, 255 for opaque)

    # Create the output GeoTIFF file
    driver = gdal.GetDriverByName('GTiff')
    dataset_out = driver.Create(output_path, cols, rows, bands + 1, gdal.GDT_Byte)  # Add 1 for the alpha channel

    # Set geotransform and CRS
    dataset_out.SetGeoTransform(geotransform)
    dataset_out.SetProjection(crs.ExportToWkt())

    # Write each band to the output
    for i in range(bands):
        dataset_out.GetRasterBand(i + 1).WriteArray(adjusted_image[i, :, :])  # Bands are 1-indexed in GDAL

    # Write the alpha channel as the last band
    dataset_out.GetRasterBand(bands + 1).WriteArray(alpha_channel)

    # Flush and close the dataset
    dataset_out.FlushCache()
    dataset_out = None  # Close file and release resources

def mask_image(image, valid_mask):
    """
    Apply a valid mask to the image, setting invalid areas (NoData or margin) to black.
    """
    masked_image = np.copy(image)
    masked_image[~valid_mask] = 0  # Set the invalid area (NoData) to black (0)
    return masked_image

def feature_matching_SIFT(img1, valid_mask1, img2, valid_mask2, output_file='feature_matches_SIFT.png', ratio_threshold=0.75, max_matches=50, distance_percentile=0.25):
    """
    Extracts features from both images using the SIFT descriptor
    and matches the features using a FLANN-based matcher with improved parameter selection.
    Removes keypoints near NoData areas and applies advanced filtering techniques.
    """
    # Initialize the SIFT detector with optimized parameters
    sift = cv2.SIFT_create(
        nfeatures=5000,       # Increase the number of features detected
        contrastThreshold=0.04,  # Adjust to capture more relevant keypoints
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
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:  # Ratio test as per Lowe's paper
            good_matches.append(m)

    print(f"Matches after Lowe's Ratio Test: {len(good_matches)}")

    '''
    # Use BFMatcher (NOTE: this is AWFUL)
    bf = cv2.BFMatcher(crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort matches by distance (lower distance = better match)
    good_matches = sorted(matches, key=lambda x: x.distance)
    '''

    # Further filter matches based on a distance threshold
    distances = np.array([m.distance for m in good_matches])
    distance_threshold = np.percentile(distances, distance_percentile * 100)  # Calculate threshold based on percentile
    filtered_matches = [m for m in good_matches if m.distance <= distance_threshold]

    print(f"Matches after distance filtering: {len(filtered_matches)}")

    # Sort the filtered matches by distance (lower distance = better match)
    filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)[:max_matches]

    print(f"Final number of matches after applying max_matches: {len(filtered_matches)}")

    # Extract points from the filtered matches
    points1 = np.array([kp1[m.queryIdx].pt for m in filtered_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in filtered_matches])

    # Draw the matches between the images
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save the feature matching result to a file
    cv2.imwrite(output_file, img_matches)

    return points1, points2

def feature_matching_ORB(img1, valid_mask1, img2, valid_mask2, output_file='feature_matches_ORB.png', ratio_threshold=0.75, max_matches=30):
    """
    Feature matching using ORB (Oriented FAST and Rotated BRIEF).
    Optimized parameter selection and pairing points using Lowe's Ratio Test and sorting by distance.
    """
    # Initiate ORB detector with optimized parameters
    orb = cv2.ORB_create(
        nfeatures=500,         # Max number of features to retain
        scaleFactor=1.2,        # Scale factor for the image pyramid
        nlevels=8,              # Number of pyramid levels
        # edgeThreshold=31,       # Size of the border where features are not detected
        firstLevel=0,           # Pyramid layer to start from
        WTA_K=2,                # Number of points to produce each descriptor element
        scoreType=cv2.ORB_HARRIS_SCORE,  # Scoring algorithm
        patchSize=24,           # Size of the patch used for computing descriptors
        fastThreshold=20        # FAST detection threshold
    )

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, valid_mask1)
    kp2, des2 = orb.detectAndCompute(img2, valid_mask2)
    print(f"Keypoints detected: {len(kp1)} in image 1, {len(kp2)} in image 2")

    # Create a BFMatcher object with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors using KNN
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower distance = better match)
    good_matches = sorted(matches, key=lambda x: x.distance)

    # Limit the number of matches to max_matches
    good_matches = good_matches[:max_matches]

    print(f"Good matches after applying max_matches threshold: {len(good_matches)}")

    # Extract points from good matches
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    # Draw the matches between the images
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save the feature matching result to a file
    cv2.imwrite(output_file, img_matches)

    return points1, points2

def main():
    # Step 1: Read the images
    img_dtm, alpha_dtm, geotransform_dtm, crs_dtm, prj_dtm = read_geotiff(r'CTX_DEM_Retrieve\clipped.tif')
    img_crism, alpha_crism, geotransform_crism, crs_crism, prj_dtm = read_geotiff(r'CTX_DEM_Retrieve\reprojected_old.tif')

    # Step 2: Convert reprojected image to grayscale (using only RGB channels if RGBA)
    img_dtm_gray = cv2.cvtColor(img_dtm.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    img_crism_gray = cv2.cvtColor(img_crism.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY) # Change shape to (1091, 981, 3)
    #cv2.imwrite('reproject_gray.png', img_crism_gray)

    # Step 3: Feature matching
    points1, points2 = feature_matching_SIFT(img_dtm_gray, alpha_dtm, img_crism_gray, alpha_crism)
    #points1, points2 = feature_matching_ORB(img_dtm_gray, alpha_dtm, img_crism_gray, alpha_crism)
    # ORB performs AWFUL when applied to orthoimages

    # Step 4: Apply georeferencing
    #orthorectified_image, new_geotransform = apply_georeferencing_cv(points1, points2, img_crism, geotransform_dtm)
    points_map = pixel_to_geographic(points1, geotransform_dtm)
    orthorectified_image, new_geotransform = apply_georeferencing(points2, points_map, img_crism, geotransform_dtm, prj_dtm)

    # Step 5: Save the newly orthorectified image (with or without alpha channel)
    save_georeferenced_image(r'CTX_DEM_Retrieve\orthorecitified_new.tif', orthorectified_image, new_geotransform, crs_dtm)

if __name__ == "__main__":
    main()