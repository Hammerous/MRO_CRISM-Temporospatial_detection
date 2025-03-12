import cv2, os
import numpy as np
from osgeo import osr, gdal
from GeoRef_Test.dense_pairing import feature_matching_denseSIFT

def process_geotiff(image_path, rgb_bands=None):
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
    return img_equalized, alpha_mask, dataset

def apply_affinetransform(points1, points2, in_dataset, target_projection, target_geotransform):
    """
    Applies affine (homography) transformation to the input GDAL dataset based on matched pixel coordinates.

    :param points1:            Nx2 array of matched 'destination' pixel coordinates (float32 or float64).
    :param points2:            Nx2 array of matched 'source' pixel coordinates (float32 or float64).
    :param in_dataset:         GDAL dataset to be transformed.
    :param target_projection:  WKT or PROJ4 string specifying the desired projection for the output dataset.
    :return:                   (out_dataset, rmse, transformed_points2)
                               out_dataset:         new GDAL in-memory dataset with updated geotransform and projection
                               rmse:                Root Mean Square Error (in pixels) for the transform
                               transformed_points2: Nx2 array of transformed source points2 in pixel coordinates
    """
    # ------------------------------------------------------------------------
    # 1) Extract basic info from the input dataset
    # ------------------------------------------------------------------------
    #geotransform = in_dataset.GetGeoTransform()
    # original_projection = in_dataset.GetProjection()  # Not strictly needed if we are using target_projection
    
    x_size = in_dataset.RasterXSize
    y_size = in_dataset.RasterYSize
    num_bands = in_dataset.RasterCount
    datatype = in_dataset.GetRasterBand(1).DataType

    # ------------------------------------------------------------------------
    # 3) Compute the homography (3x3) that maps points2 -> points1
    # ------------------------------------------------------------------------
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # ------------------------------------------------------------------------
    # 4) Compute transformation accuracy via residuals (RMSE in pixel space)
    # ------------------------------------------------------------------------
    points2_h = np.concatenate([points2, np.ones((points2.shape[0], 1))], axis=1).T
    transformed_points2 = (h @ points2_h).T  # shape: (N, 3)
    # Normalize by the third (homogeneous) coordinate
    transformed_points2[:, :2] /= transformed_points2[:, 2:3]

    residuals = np.linalg.norm(points1 - transformed_points2[:, :2], axis=1)
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"Transformation Accuracy (RMSE): {rmse:.4f} pixels")
    print(f"{points1.shape[0]} Points for Georeference Detected.")

    # ------------------------------------------------------------------------
    # 5) Compute the new bounding rectangle after the transform
    #    by transforming the corners of the original image
    # ------------------------------------------------------------------------
    corners = np.array([
        [0,       0],
        [x_size,  0],
        [x_size,  y_size],
        [0,       y_size]
    ], dtype=np.float32)

    corners_h = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])  # shape = [4, 3]
    new_corners = (h @ corners_h.T).T  # shape = [4, 3]

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
    final_transform = T @ h
    
    # ------------------------------------------------------------------------
    # 5) Compute the transformed pixel coordinates of points2 in the new image space
    # ------------------------------------------------------------------------
    # Recompute in homogeneous coordinates and apply the full final transform
    points2_h = np.concatenate([points2, np.ones((points2.shape[0], 1))], axis=1)  # shape: (N, 3)
    transformed_points2_new = (final_transform @ points2_h.T).T  # shape: (N, 3)
    transformed_points2_new[:, :2] /= transformed_points2_new[:, 2:3]

    # ------------------------------------------------------------------------
    # 2) Read all bands into a NumPy array (shape = [num_bands, height, width])
    # ------------------------------------------------------------------------
    driver = gdal.GetDriverByName('MEM')  # In-memory dataset for output
    # ------------------------------------------------------------------------
    # 6) Create an in-memory GDAL dataset for the output
    # ------------------------------------------------------------------------
    out_dataset = driver.Create(
        '',            # no filename for in-memory
        new_width,
        new_height,
        num_bands,
        datatype
    )
    # ------------------------------------------------------------------------
    # 7) Compute the updated geotransform
    #    If we assume the original geotransform has the form:
    #       [topLeftX, xRes, 0, topLeftY, 0, yRes]
    #    If skew terms exist (gt[2], gt[4]), we incorporate them as well.
    # ------------------------------------------------------------------------
    new_geotransform = list(target_geotransform)
    # Shift in pixel space -> shift in georeferenced units
    # new X origin:
    new_geotransform[0] += min_x * target_geotransform[1] + min_y * target_geotransform[2]
    # new Y origin:
    new_geotransform[3] += min_x * target_geotransform[4] + min_y * target_geotransform[5]
    out_dataset.SetGeoTransform(new_geotransform)

    # ------------------------------------------------------------------------
    # 8) Set the desired (target) projection
    # ------------------------------------------------------------------------
    out_dataset.SetProjection(target_projection)
    # ------------------------------------------------------------------------
    # 9) Warp each band using OpenCV's warpPerspective and write to output
    # ------------------------------------------------------------------------
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
    del in_dataset
    # ------------------------------------------------------------------------
    # 10) Return the output dataset, RMSE, and transformed points2
    # ------------------------------------------------------------------------
    return out_dataset, transformed_points2_new[:, :2]

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

def feature_matching_SIFT(img1, valid_mask1, img2, valid_mask2, output_file='feature_matches_v2_SIFT.png',
                          ransac_reproj_threshold=10.0, ratio_threshold=0.7):
    """
    Extracts features from both images using the SIFT descriptor
    and matches the features using a FLANN-based matcher with improved parameter selection.
    Removes keypoints near NoData areas and applies advanced filtering techniques, including RANSAC.
    """
    # Initialize the SIFT detector with optimized parameters
    sift = cv2.SIFT_create(
        nfeatures=0,              # Auto-detect optimal number of features
        nOctaveLayers=10,          # Increase from default (3) for better scale-space representation
        contrastThreshold=0.01,   # Prevent too many weak keypoints
        #edgeThreshold=10,         # Avoid unreliable edge responses
        sigma=1.6,                # Standard value for scale-space pyramid
        #descriptorType=cv2.CV_32F # Use 32-bit float descriptors for better precision
    )

    # Detect keypoints and descriptors in both images after masking
    kp1, des1 = sift.detectAndCompute(img1, valid_mask1)
    kp2, des2 = sift.detectAndCompute(img2, valid_mask2)
    print(f"Keypoints detected: {len(kp1)} in image 1, {len(kp2)} in image 2")

    # Use FLANN-based matcher
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),  dict(checks=200))        # Use KDTree for SIFT descriptors
    matches = flann.knnMatch(des1, des2, k=2)  # Find the 2 nearest neighbors for each descriptor

    # Apply Lowe's ratio test to filter good matches
    matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]

    # Extract matched keypoint coordinates
    points1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    # Apply RANSAC-based filtering for robustness
    if len(points1) >= 4:  # RANSAC requires at least 4 points
        homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_reproj_threshold)
        if mask is not None:
            mask = mask.ravel().astype(bool)
            points1, points2 = points1[mask], points2[mask]
            matches = [m for i, m in enumerate(matches) if mask[i]]
    else:
        raise ValueError(f"Insufficient points to perform RANSAC filtering: {points1}.")

    # Draw the matches between the images if output_file is specified
    if output_file:
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(output_file, img_matches)

    # Remove duplicate points
    unique_points1, unique_indices1 = np.unique(points1, axis=0, return_index=True)
    points1, points2 = points1[unique_indices1], points2[unique_indices1]

    unique_points2, unique_indices2 = np.unique(points2, axis=0, return_index=True)
    points1, points2 = points1[unique_indices2], points2[unique_indices2]

    print(f"Final number of matches after RANSAC filtering: {points1.shape[0]}       ")
    return points1, points2

def apply_georeferencing(in_ds, points_image, points_map, result_path, resample=gdal.GRA_Lanczos, compression='LZW', num_threads='ALL_CPUS'):
    points_image = np.asarray(points_image, dtype=np.float64)  # shape: (N, 2) with columns [px, py]
    points_map   = np.asarray(points_map, dtype=np.float64)
    combined = np.hstack([points_image, points_map])
    gcps_warp = [gdal.GCP(mx, my, 0.0, px, py) for px, py, mx, my in combined]
    # gcps_warp = [gdal.GCP(float(mx), float(my), 0.0, float(px), float(py))
    #         for (px, py), (mx, my) in zip(points_image, points_map)]

    # Assign GCPs (warp subset) and set the coordinate system
    print(in_ds.GetGeoTransform())
    in_ds.SetGCPs(gcps_warp, in_ds.GetProjection())

    # Define warp options
    warp_options = gdal.WarpOptions(
        dstSRS=in_ds.GetProjection(),                     # Output spatial reference system
        format='GTiff',                     # Output as GeoTIFF
        tps=True,                           # Use thin plate spline transformation
        errorThreshold=0.0,                 # Lower error threshold for better accuracy (in pixels)
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
    print(f"Dataset saved to {out_filepath}")

def visualize_distances(points1, points2, top_num=20, num_directions=16, dist_interval = 10, output_dir='./'):
    import matplotlib.pyplot as plt
    # Check that both arrays have the same length
    if len(points1) != len(points2):
        raise ValueError("The number of points in both arrays must be the same.")
    
    # Calculate distances between corresponding pairs
    distances = np.linalg.norm(points1 - points2, axis=1)

    # Plot and save the histogram
    plt.figure(figsize=(10, 5))
    plt.hist(distances, bins=50, color='skyblue', edgecolor='black')
    plt.title('Frequency Histogram of Point Pair Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distance_histogram.png")
    plt.close()

    # Get the indices of the top largest distances
    top_indices = np.argsort(distances)[-top_num:]

    # Plot and save the top `top_num` point pairs with largest distances
    plt.figure(figsize=(10, 10))
    plt.scatter(points1[:, 0], points1[:, 1], color='blue', label='Points 1', s=5)
    plt.scatter(points2[:, 0], points2[:, 1], color='orange', label='Points 2', s=5)

    for i in top_indices:
        plt.plot([points1[i, 0], points2[i, 0]], [points1[i, 1], points2[i, 1]], 
                 color='red', linestyle='-', linewidth=2, alpha=0.7)

    plt.title(f'Top {top_num} Point Pairs with Largest Distances')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_point_pairs.png", dpi=300)
    plt.close()

    # Radar plot: Frequency of distances in multiple directions
    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
      # Define the distance interval
    direct_dist_count = {}  # This dict will accumulate distances for each direction

    # Initialize a dict with keys as the range of distances (dist//dist_interval) and values as lists of counts for each direction
    direct_dist_count = np.zeros((int(np.max(distances)//dist_interval + 1), num_directions))        

    # Vectorize the angle and distance calculations
    dx = points2[:, 0] - points1[:, 0]
    dy = points2[:, 1] - points1[:, 1]
    distances = np.sqrt(dx**2 + dy**2)
    angles_vec = np.arctan2(dy, dx) % (2 * np.pi)

    # Find closest direction for each point
    direction_indices = np.argmin(np.abs(angles - angles_vec[:, None]), axis=1)

    # Initialize the direct_dist_count array
    direct_dist_count = np.zeros((direct_dist_count.shape[0], num_directions))

    # Update the direct_dist_count using vectorization
    dist_ranges = (distances // dist_interval).astype(int)

    # Original logic: For each distance, update all counts up to dist_range
    for i, dist_range in enumerate(dist_ranges):
        for d in range(dist_range + 1):
            direct_dist_count[d, direction_indices[i]] += 1

    # Plot and save the radar chart
    plt.figure(figsize=(7, 7))
    angles = np.concatenate((angles, [angles[0]]))  # To close the radar plot
    plt.subplot(111, polar=True)

    color_map = plt.cm.Spectral(np.linspace(0, 1, direct_dist_count.shape[0]))[::-1]  # Create a color map
    for dist_idx in range(direct_dist_count.shape[0]):
        radar_data = direct_dist_count[dist_idx, :]
        radar_data = np.concatenate((radar_data, radar_data[:1]))  # To close the radar plot
        plt.plot(angles, radar_data, linewidth=2, linestyle='-', alpha=0.7, color=color_map[dist_idx])
        plt.fill(angles, radar_data, alpha=0.4, color=color_map[dist_idx])

    plt.title('Frequency of Distances in All Directions')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distance_radar_chart.png")
    plt.close()

def georeferencing(result_path, base_img_path, warp_img_path, warp_rgb):
    # Step 1: Read the images
    img1_gray, img1_alpha, base_img = process_geotiff(base_img_path, warp_rgb)
    img2_gray, img2_alpha, img2prj = process_geotiff(warp_img_path, warp_rgb)
    #cv2.imwrite('img1_gray.png', img1_gray)
    #cv2.imwrite('img2_gray.png', img2_gray)

    # Step 2: Feature matching
    # points1, points2 = feature_matching_SIFT(img1_gray, img1_alpha, img2_gray, img2_alpha)
    points1, points2 = feature_matching_denseSIFT(img1_gray, img1_alpha, img2_gray, img2_alpha,\
                                                output_file_name='feature_matches_v2_SIFT', ratio_threshold=0.70, ransac_reproj_threshold=5.0\
                                                ,Layers=10, grid_spacing=4)
    del img1_gray, img1_alpha, img2_gray, img2_alpha
    img2prj, points2 = apply_affinetransform(points1, points2, img2prj, base_img.GetProjection(), base_img.GetGeoTransform())
    # visualize_distances(pixel_to_geographic(points1, base_img.GetGeoTransform()),\
    #                     pixel_to_geographic(points2, img2prj.GetGeoTransform()),\
    #                     output_dir=r'CTX_DEM_Retrieve\fig', num_directions=36, top_num=50)
    save_gdal_dataset(img2prj, result_path+'_Affine.tif')
    save_gdal_dataset(base_img, result_path+'_Base.tif')
    # Step 3: Apply georeferencing and save
    # points_map = pixel_to_geographic(points1, base_img.GetGeoTransform())
    # points_map2 = pixel_to_geographic(points2, img2prj.GetGeoTransform())
    # residuals = np.linalg.norm(points_map - points_map2, axis=1)
    # rmse = np.sqrt(np.mean(residuals**2))
    # print(rmse)
    # apply_georeferencing(img2prj, points2, pixel_to_geographic(points1, base_img.GetGeoTransform()), result_path+'.tif')

if __name__ == "__main__":
    base_img_path = r'CTX_DEM_Retrieve\ortho_clipped.tif'
    warp_img_path = r'Playground\frt000161ef_07_sr167j_mtr3.img'

    # base_img_path = r'CTX_DEM_Retrieve\ortho_clipped_2.tif'
    # warp_img_path = r'Playground\frt000174f4_07_sr166j_mtr3.img'

    # base_img_path = r'Playground\frt000174f4_07_sr166j_mtr3.img'
    # warp_img_path = r'Playground\frt000161ef_07_sr167j_mtr3.img'

    warp_rgb = ('R2529', 'R1506', 'R1080')
    result_path = r'CTX_DEM_Retrieve\orthorecitified_test4_1'
    georeferencing(result_path, base_img_path, warp_img_path, warp_rgb)