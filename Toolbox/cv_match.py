import cv2
import numpy as np

def _dense_keypts(valid_mask, grid_spacing, size=8):
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

def denseSIFT(img1, valid_mask1, img2, valid_mask2,
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
    kp1 = _dense_keypts(valid_mask1, grid_spacing, size=grid_spacing)
    kp2 = _dense_keypts(valid_mask2, grid_spacing, size=grid_spacing)
    
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
        return None,None,None,False
        raise ValueError("Insufficient matches for RANSAC filtering.")
    
    # Remove duplicates
    _, unique_indices = np.unique(points1, axis=0, return_index=True)
    points1, points2 = points1[unique_indices], points2[unique_indices]
    _, unique_indices = np.unique(points2, axis=0, return_index=True)
    points1, points2 = points1[unique_indices], points2[unique_indices]
    
    # Optionally draw and save the matches.
    if output_file_name:
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, filtered_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(output_file_name+'.png', img_matches)

    return homography, points1, points2, True

