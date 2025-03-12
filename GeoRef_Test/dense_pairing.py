import cv2
import numpy as np
from osgeo import gdal
import GeoRef_Test.pairing_sequence as pair

def feature_matching_SIFT_dense(
    img1, 
    valid_mask1, 
    img2, 
    valid_mask2, 
    output_file_name='feature_matches_SIFT_dense',
    step=2,
    ratio_threshold=0.75,
    ransac_reproj_threshold=10.0
):
    """
    Performs dense feature matching between two images using SIFT descriptors.
    Keypoints are sampled on a regular grid (controlled by 'step') for each image, 
    and only locations allowed by the valid_mask are used.
    
    :param img1: First input image (grayscale or color)
    :param valid_mask1: Mask for valid pixels in img1 (non-zero = valid, zero = invalid)
    :param img2: Second input image (grayscale or color)
    :param valid_mask2: Mask for valid pixels in img2 (non-zero = valid, zero = invalid)
    :param output_file: Path to save the visualization of matches. Set None to skip saving.
    :param step: Spacing in pixels for the dense keypoint grid.
    :param ratio_threshold: Lowe's ratio test threshold.
    :param ransac_reproj_threshold: RANSAC reprojection threshold for outlier removal.
    :return: (points1, points2) - N x 2 arrays of matched points in each image.
    """
    # Initialize SIFT with custom parameters (mostly relevant for descriptor calculation).
    sift = cv2.SIFT_create(
        #nfeatures=0,              # Not used for 'dense' approach
        nOctaveLayers=20,          # Increase from default (3) to capture more scales
        contrastThreshold=0.01,
        sigma=1.6
    )
    
    # 1) Create dense keypoints for img1
    kp1 = [cv2.KeyPoint(x, y, step) for y in range(0, img1.shape[0], step) for x in range(0, img1.shape[1], step)]

    # 2) Create dense keypoints for img2
    kp2 = [cv2.KeyPoint(x, y, step) for y in range(0, img2.shape[0], step) for x in range(0, img2.shape[1], step)]

    print(f"Dense keypoints created: {len(kp1)} in image 1, {len(kp2)} in image 2")

    # 3) Compute descriptors for the dense keypoints
    #    Note: sift.compute() expects a list of keypoints and returns (kp, descriptors).
    _, des1 = sift.compute(img1, kp1)
    _, des2 = sift.compute(img2, kp2)

    # If there are no descriptors, we cannot proceed
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        raise ValueError("No descriptors found for one or both images in the dense approach.")

    # 4) FLANN parameters for matching
    index_params = dict(algorithm=1, trees=8)  # KD-Tree for SIFT descriptors
    search_params = dict(checks=500)           # Increase checks for better accuracy
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 5) KNN match and apply Lowe's ratio test
    matches = flann.knnMatch(des1, des2, k=2)
    filtered_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            filtered_matches.append(m)
    del matches, flann
    print(f"Matches after ratio test: {len(filtered_matches)}")

    # 6) Extract matched point coordinates
    points1 = np.array([kp1[m.queryIdx].pt for m in filtered_matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in filtered_matches], dtype=np.float32)

    # 7) RANSAC filtering to remove outliers
    if len(points1) >= 4:  # Need at least 4 points
        homography, inliers_mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_reproj_threshold)
        if inliers_mask is not None:
            inliers_mask = inliers_mask.ravel().astype(bool)
            points1 = points1[inliers_mask]
            points2 = points2[inliers_mask]
            filtered_matches = [m for i, m in enumerate(filtered_matches) if inliers_mask[i]]
        print(f"Matches after RANSAC: {len(points1)}")
    else:
        raise ValueError("Insufficient points to perform RANSAC (need >= 4).")

    # 8) Remove potential duplicates (optional, but can be useful)
    #    a) Uniqueness on image1 points
    unique_points1, unique_indices1 = np.unique(points1, axis=0, return_index=True)
    points1, points2 = points1[unique_indices1], points2[unique_indices1]

    #    b) Uniqueness on image2 points
    unique_points2, unique_indices2 = np.unique(points2, axis=0, return_index=True)
    points1, points2 = points1[unique_indices2], points2[unique_indices2]

    # 9) (Optional) Draw matches for visualization
    if output_file_name:
        # We need a subset of matches that survived RANSAC and uniqueness filtering
        # Build a quick mapping to figure out which matches are still valid.
        final_valid_indices = set()
        # Re-check against final lists of points. An easy way:
        final_kp1_map = {tuple(pt): idx for idx, pt in enumerate(points1)}
        final_kp2_map = {tuple(pt): idx for idx, pt in enumerate(points2)}

        new_matches = []
        for m in filtered_matches:
            pt1 = tuple(np.round(kp1[m.queryIdx].pt, decimals=2))
            pt2 = tuple(np.round(kp2[m.trainIdx].pt, decimals=2))
            # Rounding helps match floating point, but you could also store them exactly
            if pt1 in final_kp1_map and pt2 in final_kp2_map:
                new_matches.append(m)

        # # Draw
        # img_matches = cv2.drawMatches(
        #     img1, kp1,
        #     img2, kp2,
        #     new_matches, None,
        #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        # )
        # cv2.imwrite(output_file, img_matches)
        # Draw keypoints only
        img1_keypoints = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2_keypoints = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Save the images with keypoints
        cv2.imwrite(output_file_name+'_1.png', img1_keypoints)
        cv2.imwrite(output_file_name+'_2.png', img2_keypoints)

    print(f"Final number of matches after all filtering: {points1.shape[0]}")

    return points1, points2

def feature_matching_DAISY(
    img1_gray, valid_mask1,
    img2_gray, valid_mask2,
    output_file='feature_matches_DAISY.png',
    ratio_threshold=0.75,
    ransac_reproj_threshold=10.0,
    step=4,
    radius=15,
    rings=4,
    histograms=16,
    orientations=8
):
    """
    Performs dense feature extraction using DAISY and matches descriptors between two images.
    Includes robust outlier filtering with RANSAC and optional result visualization.
    
    Parameters
    ----------
    img1, img2 : np.ndarray
        Input images (grayscale or color). If color, they will be converted internally to grayscale.
    valid_mask1, valid_mask2 : np.ndarray
        Binary masks indicating valid (non-NoData) regions for each image.
        Must be the same height/width as the respective image.
    output_file : str
        If provided, an image illustrating the matches is saved to this path.
    ratio_threshold : float
        Lowe's ratio test threshold for filtering matches.
    ransac_reproj_threshold : float
        Reprojection threshold for cv2.findHomography RANSAC outlier removal.
    step : int
        Sampling step for the DAISY descriptor in both row and column direction.
    radius : int
        Outer radius (in pixels) for the DAISY descriptor.
    rings : int
        Number of concentric circles (rings) in the DAISY descriptor.
    histograms : int
        Number of histograms sampled per ring.
    orientations : int
        Number of orientations (bins) for the histograms.

    Returns
    -------
    points1, points2 : np.ndarray
        Matched keypoint coordinates (x,y) in each image after RANSAC and duplicate removal.
    """
    from skimage.feature import daisy
    # 2. Compute DAISY descriptors for each image
    descs1 = daisy(
        img1_gray,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations
    )
    descs2 = daisy(
        img2_gray,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations
    )

    # DAISY returns a 3D array: (rows_out, cols_out, descriptor_size)
    rows_desc1, cols_desc1, descriptor_size = descs1.shape
    rows_desc2, cols_desc2, _ = descs2.shape

    # 3. Map descriptor positions to image coordinates.
    #    For each row i, col j in the DAISY output, the center in the original image is roughly:
    #        (y, x) = (radius + i * step, radius + j * step)
    #    We'll flatten these into coordinate arrays and descriptor arrays.
    row_coords1 = radius + step * np.arange(rows_desc1)
    col_coords1 = radius + step * np.arange(cols_desc1)
    grid_y1, grid_x1 = np.meshgrid(row_coords1, col_coords1, indexing='ij')  # shape (rows_desc1, cols_desc1)

    # Flatten
    coords1 = np.column_stack([grid_x1.ravel(), grid_y1.ravel()])  # shape (rows_desc1*cols_desc1, 2)
    descs1_flat = descs1.reshape(-1, descriptor_size)              # same number of positions

    # Apply valid mask for image1
    # Make sure coordinates are within the image bounds and are valid in the mask.
    h1, w1 = img1_gray.shape
    in_bounds_1 = (
        (coords1[:, 0] >= 0) & (coords1[:, 0] < w1) &
        (coords1[:, 1] >= 0) & (coords1[:, 1] < h1)
    )
    valid_in_mask_1 = valid_mask1[coords1[:, 1].astype(int), coords1[:, 0].astype(int)]
    keep_1 = in_bounds_1 & (valid_in_mask_1 > 0)

    coords1 = coords1[keep_1]
    descs1_flat = descs1_flat[keep_1]

    # Repeat for image2
    row_coords2 = radius + step * np.arange(rows_desc2)
    col_coords2 = radius + step * np.arange(cols_desc2)
    grid_y2, grid_x2 = np.meshgrid(row_coords2, col_coords2, indexing='ij')

    coords2 = np.column_stack([grid_x2.ravel(), grid_y2.ravel()])
    descs2_flat = descs2.reshape(-1, descriptor_size)

    h2, w2 = img2_gray.shape
    in_bounds_2 = (
        (coords2[:, 0] >= 0) & (coords2[:, 0] < w2) &
        (coords2[:, 1] >= 0) & (coords2[:, 1] < h2)
    )
    valid_in_mask_2 = valid_mask2[coords2[:, 1].astype(int), coords2[:, 0].astype(int)]
    keep_2 = in_bounds_2 & (valid_in_mask_2 > 0)

    coords2 = coords2[keep_2]
    descs2_flat = descs2_flat[keep_2]

    # 4. Use FLANN or BF for descriptor matching; here we use FLANN
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=8)
    search_params = dict(checks=500)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Convert DAISY (float64) -> float32 for OpenCV
    descs1_float = descs1_flat.astype(np.float32)
    descs2_float = descs2_flat.astype(np.float32)

    # KNN match
    knn_matches = flann.knnMatch(descs1_float, descs2_float, k=2)

    # 5. Lowe's Ratio Test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # Extract matched coordinates
    matched_pts1 = coords1[[m.queryIdx for m in good_matches]]
    matched_pts2 = coords2[[m.trainIdx for m in good_matches]]

    # 6. RANSAC-based filtering to remove outliers
    if len(matched_pts1) < 4:
        raise ValueError("Not enough matched points for RANSAC. Found only {}".format(len(matched_pts1)))

    H, inlier_mask = cv2.findHomography(matched_pts1, matched_pts2, cv2.RANSAC, ransac_reproj_threshold)
    if inlier_mask is not None:
        inlier_mask = inlier_mask.ravel().astype(bool)
        matched_pts1 = matched_pts1[inlier_mask]
        matched_pts2 = matched_pts2[inlier_mask]

    # 7. Remove any duplicates (rarely an issue, but good to clean up)
    unique_p1, idx1 = np.unique(matched_pts1, axis=0, return_index=True)
    matched_pts1 = matched_pts1[idx1]
    matched_pts2 = matched_pts2[idx1]

    unique_p2, idx2 = np.unique(matched_pts2, axis=0, return_index=True)
    matched_pts1 = matched_pts1[idx2]
    matched_pts2 = matched_pts2[idx2]

    # 8. (Optional) Visualize and save match results
    # Because DAISY is dense, there can be thousands of matches. Drawing them all may be cluttered.
    # We synthesize KeyPoint objects for the matched points so we can use `drawMatches`.
    if output_file:        
        # Create KeyPoints for final matches (index i in points1 -> index i in points2)
        final_kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in matched_pts1]
        final_kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in matched_pts2]

        final_matches = [cv2.DMatch(i, i, 0) for i in range(len(final_kp1))]
        match_vis = cv2.drawMatches(
            img1_gray, final_kp1,
            img2_gray, final_kp2,
            final_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(output_file, match_vis)

    print(f"Final number of dense DAISY matches after RANSAC: {matched_pts1.shape[0]}")
    return matched_pts1, matched_pts2

def generate_dense_keypoints(valid_mask, grid_spacing, size=8):
    """
    Generate keypoints on a dense grid, restricted by a valid mask.
    """
    h, w = valid_mask.shape[:2]
    ys, xs = np.mgrid[0:h:grid_spacing, 0:w:grid_spacing]
    xs = xs.flatten()
    ys = ys.flatten()

    # Only keep coordinates where the valid_mask is non-zero.
    valid_indices = valid_mask[ys, xs] != 0
    xs, ys = xs[valid_indices], ys[valid_indices]

    # Create keypoints with the provided size.
    keypoints = [cv2.KeyPoint(float(x), float(y), size) for x, y in zip(xs, ys)]
    return keypoints

def feature_matching_denseSIFT(img1, valid_mask1, img2, valid_mask2,
                               grid_spacing=8, Layers=10,
                               output_file_name='feature_matches_SIFT_dense',
                               ratio_threshold=0.7, ransac_reproj_threshold=10.0):
    """
    Extracts dense SIFT features from both images using a regular grid and matches
    the features using a FLANN-based matcher with improved parameter selection.
    Invalid keypoints (e.g., near NoData areas) are filtered out via the valid_mask.
    
    Parameters:
        img1 (np.array): First image.
        valid_mask1 (np.array): Binary mask for image1 (non-zero indicates valid data).
        img2 (np.array): Second image.
        valid_mask2 (np.array): Binary mask for image2 (non-zero indicates valid data).
        grid_spacing (int): Step size (in pixels) for dense sampling of keypoints.
        output_file (str): Filename to save the match visualization (if provided).
        ratio_threshold (float): Lowe's ratio threshold for filtering matches.
        ransac_reproj_threshold (float): RANSAC reprojection threshold.
    
    Returns:
        points1, points2 (np.array): Arrays of matched coordinates in image1 and image2.
    """

    # Initialize the SIFT detector with optimized parameters.
    # Note: In Dense SIFT, detection is replaced by grid sampling.
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=Layers,
        contrastThreshold=0.01,
        sigma=1.6
    )
    
    # Generate dense keypoints using vectorized operations.
    kp1 = generate_dense_keypoints(valid_mask1, grid_spacing, size=grid_spacing)
    kp2 = generate_dense_keypoints(valid_mask2, grid_spacing, size=grid_spacing)
    
    print(f"Total keypoints: {len(kp1)} in image 1, {len(kp2)} in image 2")
    
    # Compute SIFT descriptors.
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)
    
    if des1 is None or des2 is None:
        raise ValueError("Descriptor computation failed for one or both images.")
    
    # Set up FLANN-based matcher.
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=500)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors using k-NN (k=2).
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test.
    filtered_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]
    
    # Sort the matches by distance.
    filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)
    
    # Extract matched coordinates.
    points1 = np.array([kp1[m.queryIdx].pt for m in filtered_matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in filtered_matches], dtype=np.float32)
    
    # RANSAC filtering: only if sufficient matches exist.
    if len(points1) >= 4:
        homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_reproj_threshold)
        if mask is not None:
            mask = mask.ravel().astype(bool)
            points1, points2 = points1[mask], points2[mask]
            filtered_matches = [m for i, m in enumerate(filtered_matches) if mask[i]]
    else:
        raise ValueError("Insufficient matches for RANSAC filtering.")
    
    # Remove duplicates
    _, unique_indices = np.unique(points1, axis=0, return_index=True)
    points1, points2 = points1[unique_indices], points2[unique_indices]
    _, unique_indices = np.unique(points2, axis=0, return_index=True)
    points1, points2 = points1[unique_indices], points2[unique_indices]
    
    # Optionally draw and save the matches.
    if output_file_name:
        # img1_keypoints = cv2.drawKeypoints(img1, kp1, filtered_matches, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # img2_keypoints = cv2.drawKeypoints(img2, kp2, filtered_matches, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # # Save the images with keypoints
        # cv2.imwrite(output_file_name+'_1.png', img1_keypoints)
        # cv2.imwrite(output_file_name+'_2.png', img2_keypoints)
    
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, filtered_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(output_file_name+'.png', img_matches)

    print(f"Final number of matches after RANSAC filtering: {points1.shape[0]}")
    return points1, points2

def feature_matching_dense_multi(img1, valid_mask1, img2, valid_mask2,
                                 grid_spacing=8,
                                 output_file_name='feature_matches_dense',
                                 ratio_threshold=0.7, ransac_reproj_threshold=10.0):
    """
    Perform dense feature matching between two images using multiple descriptors
    (SIFT and ORB). For a given pixel, if multiple descriptors provide matches,
    the average displacement vector is computed for the final pairing.
    """
    # --- 1. Generate dense keypoints using grid sampling ---
    kp1 = generate_dense_keypoints(valid_mask1, grid_spacing, size=grid_spacing)
    kp2 = generate_dense_keypoints(valid_mask2, grid_spacing, size=grid_spacing)
    
    print(f"Total grid keypoints: {len(kp1)} in image 1, {len(kp2)} in image 2")
    
    # --- 2. Initialize detectors/descriptors ---
    # SIFT: more robust to scale and illumination changes.
    sift = cv2.SIFT_create(nOctaveLayers=10, contrastThreshold=0.01, sigma=1.6)
    # ORB: fast and complementary; note that ORB produces binary descriptors.
    orb = cv2.ORB_create(nfeatures=0)
    
    # --- 3. Compute descriptors with SIFT and ORB ---
    kp1_sift, des1_sift = sift.compute(img1, kp1)
    kp2_sift, des2_sift = sift.compute(img2, kp2)
    
    kp1_orb, des1_orb = orb.compute(img1, kp1)
    kp2_orb, des2_orb = orb.compute(img2, kp2)
    
    if des1_sift is None or des2_sift is None or des1_orb is None or des2_orb is None:
        raise ValueError("Descriptor computation failed for one or both images.")
    
    # --- 4. Match descriptors for both SIFT and ORB ---
    # SIFT matching with FLANN matcher.
    index_params = dict(algorithm=1, trees=5)  # KD-Tree index for SIFT.
    search_params = dict(checks=200)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_sift = flann.knnMatch(des1_sift, des2_sift, k=2)
    good_matches_sift = [m for m, n in matches_sift if m.distance < ratio_threshold * n.distance]

    # ORB matching with BFMatcher using Hamming distance.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_orb = bf.knnMatch(des1_orb, des2_orb, k=2)
    good_matches_orb = [m for m, n in matches_orb if m.distance < ratio_threshold * n.distance]
    
    print(f"SIFT matches: {len(good_matches_sift)}, ORB matches: {len(good_matches_orb)}")
    
    # --- 5. Fuse matches: group by the pixel (keypoint) in image 1 ---
    # We use a dictionary keyed by integer pixel coordinates.
    match_dict = {}  # key: (x,y) from image1, value: list of (pt1, pt2)
    
    def add_matches(matches, kp1_list, kp2_list):
        for m in matches:
            pt1 = np.array(kp1_list[m.queryIdx].pt, dtype=np.float32)
            pt2 = np.array(kp2_list[m.trainIdx].pt, dtype=np.float32)
            # Round pixel coordinates to group matches by pixel.
            key = (int(round(pt1[0])), int(round(pt1[1])))
            if key not in match_dict:
                match_dict[key] = []
            match_dict[key].append((pt1, pt2))
    
    add_matches(good_matches_sift, kp1_sift, kp2_sift)
    add_matches(good_matches_orb, kp1_orb, kp2_orb)
    
    # --- 6. For each pixel key, compute the average displacement vector ---
    final_points1 = []
    final_points2 = []

    for key, matches in match_dict.items():
        pts1, pts2 = zip(*matches)  # separate (pt1, pt2) pairs
        final_points1.append(np.mean(pts1, axis=0))
        final_points2.append(np.mean(pts2, axis=0))

    final_points1 = np.array(final_points1, dtype=np.float32)
    final_points2 = np.array(final_points2, dtype=np.float32)
        
    print(f"Combined matches (before RANSAC): {final_points1.shape[0]}")
    
    # --- 7. RANSAC filtering ---
    if final_points1.shape[0] >= 4:
        homography, inlier_mask = cv2.findHomography(final_points1, final_points2,
                                                     cv2.RANSAC, ransac_reproj_threshold)
        if inlier_mask is not None:
            inlier_mask = inlier_mask.ravel().astype(bool)
            final_points1 = final_points1[inlier_mask]
            final_points2 = final_points2[inlier_mask]
        else:
            raise ValueError("RANSAC failed to find a valid homography.")
    else:
        raise ValueError("Insufficient matches for RANSAC filtering.")
    
    # --- 8. Remove any duplicates (if present) ---
    # Remove duplicates from image1 matches.
    _, unique_indices = np.unique(final_points1, axis=0, return_index=True)
    final_points1 = final_points1[unique_indices]
    final_points2 = final_points2[unique_indices]
    
    # Remove duplicates from image2 matches.
    _, unique_indices = np.unique(final_points2, axis=0, return_index=True)
    final_points1 = final_points1[unique_indices]
    final_points2 = final_points2[unique_indices]
    
    # --- 9. Optionally draw and save keypoints and matches ---
    if output_file_name:
        # Draw all keypoints (for visualization of grid sampling).
        img1_keypoints = cv2.drawKeypoints(img1, kp1, None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img2_keypoints = cv2.drawKeypoints(img2, kp2, None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(output_file_name+'_img1_keypoints.png', img1_keypoints)
        cv2.imwrite(output_file_name+'_img2_keypoints.png', img2_keypoints)
    
    print(f"Final number of matches after RANSAC filtering: {final_points1.shape[0]}")
    return final_points1, final_points2

def get_elevations_gdal(coords: np.ndarray, dtm_path: str) -> np.ndarray:
    """
    Retrieve elevation values from a DTM using vectorized processing.
    
    Parameters:
    dtm_path (str): Path to the DTM .tif file.
    coords (numpy.ndarray): Nx2 array of geographic coordinates (longitude, latitude).
    
    Returns:
    numpy.ndarray: Array of elevation values corresponding to input coordinates.
    """
    # Open the dataset
    dataset = gdal.Open(dtm_path)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open DTM file: {dtm_path}")
    
    # Get geotransform and raster band
    transform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    
    # Read the raster data as a NumPy array
    elevation_data = band.ReadAsArray()
    
    # Compute pixel indices
    x_coords = ((coords[:, 0] - transform[0]) / transform[1]).astype(int)
    y_coords = ((coords[:, 1] - transform[3]) / transform[5]).astype(int)
    
    # Ensure indices are within bounds
    rows, cols = elevation_data.shape
    valid_mask = (x_coords >= 0) & (x_coords < cols) & (y_coords >= 0) & (y_coords < rows)
    
    # Retrieve elevation values using vectorized indexing
    elevations = np.full((coords.shape[0],1), np.nan)  # Default NaN for out-of-bounds points
    elevations[valid_mask,0] = elevation_data[y_coords[valid_mask], x_coords[valid_mask]]
    
    return elevations

def apply_georeferencing(in_ds, points_image, points_map, elevation, new_prj, result_path, resample=gdal.GRA_Lanczos, compression='LZW', num_threads='ALL_CPUS'):
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
    combined = np.hstack([points_image, points_map, elevation])
    gcps_warp = [gdal.GCP(mx, my, mz, px, py) for px, py, mx, my, mz in combined]
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

def georeferencing(result_path, base_img_path, warp_img_path, base_dtm_path, warp_rgb):
    # Step 1: Read the images
    img1_gray, img1_alpha, base_img = pair.process_geotiff(base_img_path, warp_rgb)
    img2_gray, img2_alpha, img2prj = pair.process_geotiff(warp_img_path, warp_rgb)
    #cv2.imwrite('img1_gray.png', img1_gray)
    #cv2.imwrite('img2_gray.png', img2_gray)

    # Step 2: Feature matching
    #points1, points2 = feature_matching_SIFT_dense(img1_gray, img1_alpha, img2_gray, img2_alpha)
    #points1, points2 = feature_matching_DAISY(img1_gray, img1_alpha, img2_gray, img2_alpha)
    points1, points2 = feature_matching_denseSIFT(img1_gray, img1_alpha, img2_gray, img2_alpha, grid_spacing=2)
    #points1, points2 = feature_matching_dense_multi(img1_gray, img1_alpha, img2_gray, img2_alpha, grid_spacing=4)
    del img1_gray, img1_alpha, img2_gray, img2_alpha

    # Step 3: Apply georeferencing and save
    points_map = pair.pixel_to_geographic(points1, base_img.GetGeoTransform())
    import pandas as pd
    df = pd.DataFrame(np.hstack((points2, pair.pixel_to_geographic(points2, img2prj.GetGeoTransform()), points1, points_map)),\
                                 columns=['p0x','p0y', 'm0x','m0y', 'p1x','p1y', 'm1x','m1y'])
    df.to_csv("GCPs.csv", encoding='UTF8', index=False)
    pair.apply_georeferencing(img2prj, points2, points_map,\
                    base_img.GetProjection(), result_path,\
                    resample=gdal.GRA_NearestNeighbour)
    # elevation = get_elevations_gdal(points_map,base_dtm_path)
    # apply_georeferencing(img2prj, points2, points_map, elevation,\
    #                     base_img.GetProjection(), result_path,\
    #                     resample=gdal.GRA_NearestNeighbour)

if __name__ == "__main__":
    base_img_path = r'CTX_DEM_Retrieve\ortho_clipped.tif'
    base_dtm_path = r'CTX_DEM_Retrieve\tif\K06_055567_1983_XN_18N283W__K05_055501_1983_XN_18N283W_dtm.tif'
    warp_img_path = r'Playground\frt000161ef_07_sr167j_mtr3.img'

    # base_img_path = r'CTX_DEM_Retrieve\ortho_clipped_2.tif'
    # warp_img_path = r'Playground\frt000174f4_07_sr166j_mtr3.img'

    # base_img_path = r'Playground\frt000161ef_07_sr167j_mtr3.img'
    # warp_img_path = r'Playground\frt000174f4_07_sr166j_mtr3.img'

    warp_rgb = ('R2529', 'R1506', 'R1080')
    result_path = r'CTX_DEM_Retrieve\orthorecitified_test3_1.tif'
    georeferencing(result_path, base_img_path, warp_img_path, base_dtm_path, warp_rgb)