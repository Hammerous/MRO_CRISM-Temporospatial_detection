import cv2
import numpy as np
import pairing_sequence as pair

def resize_and_center(src, target_shape):
    """
    Resizes 'src' so it fits (without distorting aspect ratio) within 'target_shape',
    then places it in the center of a black canvas (for images) of size 'target_shape'.
    
    :param src:           Input image (numpy array).
    :param target_shape:  (height, width) tuple.
    :return:              Centered, resized image of size 'target_shape'.
    """
    th, tw = target_shape  # target (height, width)
    h, w = src.shape[:2]

    # Compute an aspect-ratio-preserving scale
    scale = min(float(tw)/w, float(th)/h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with preserved aspect ratio
    resized = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a black canvas
    if len(src.shape) == 2:  # grayscale
        canvas = np.zeros((th, tw), dtype=src.dtype)
    else:                    # multi-channel
        canvas = np.zeros((th, tw, src.shape[2]), dtype=src.dtype)

    # Compute centering offsets
    top = (th - new_h) // 2
    left = (tw - new_w) // 2

    # Paste the resized image into the center of the canvas
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas

def compute_false_color_disparity(disparity_float):
    """
    Given a floating-point disparity map, produce an 8-bit false-color visualization.
    
    :param disparity_float:  Disparity map in float32.
    :return:                 BGR color image (uint8) with a colormap.
    """
    # Replace negative or invalid disparities with 0 (optional)
    # disparity_float[disparity_float < 0] = 0

    disp_min = np.min(disparity_float)
    disp_max = np.max(disparity_float)

    if disp_max - disp_min < 1e-5:
        # Avoid division by zero if disparity is constant
        disp_vis_gray = np.zeros_like(disparity_float, dtype=np.uint8)
    else:
        # Normalize to 0..255
        disp_vis_gray = 255 * (disparity_float - disp_min) / (disp_max - disp_min)
        disp_vis_gray = disp_vis_gray.astype(np.uint8)

    # Apply a colormap for false color (e.g. JET or INFERNO)
    disp_vis_color = cv2.applyColorMap(disp_vis_gray, cv2.COLORMAP_JET)
    return disp_vis_color

def warp_right_image_using_disparity(left_img, right_img, disparity_float):
    """
    Warp the right image into the left image's coordinate space, using the disparity map.
    
    This is a naive per-pixel shift:
      - The disparity tells us how many pixels to shift (x-coordinate).
      - If disparity[y,x] = d, then the pixel in right_img at (y, x - d) 
        should match left_img at (y, x).
    
    :param left_img:        Grayscale left image (H x W).
    :param right_img:       Grayscale right image (same shape as left_img).
    :param disparity_float: float32 disparity map aligned to left_img. 
    :return:                A new grayscale image (H x W) which is the right image 
                            warped onto the left's coordinates.
    """
    h, w = left_img.shape[:2]
    matched_img = np.zeros_like(left_img, dtype=left_img.dtype)

    for y in range(h):
        for x in range(w):
            d = disparity_float[y, x]
            # Round or int() the shift
            x_right = int(round(x - d))
            if 0 <= x_right < w:
                matched_img[y, x] = right_img[y, x_right]

    return matched_img


def stereo_match_original_vs_scaled(
    left_img, left_mask,
    right_img, right_mask,
    output_disparity_path,
    output_matched_image_path
):
    """
    1. Load left (original) image and its mask (grayscale).
    2. Load right (to-match) image and its mask (grayscale).
    3. Resize+center the right image and mask to match the left's shape.
    4. Compute disparity with StereoSGBM on (left_img, right_img_resized).
    5. Combine masks, set invalid regions to 0 in the disparity.
    6. Produce a false-color disparity image and save it.
    7. Warp the right image to the left's coordinate space using the disparity
       and save the matched image.
    """
    # 2. Determine the "fixed" shape based on the left (original) image
    h_left, w_left = left_img.shape

    # 3. Resize + center the right image/mask to match the left's shape
    right_img_resized  = resize_and_center(right_img,  (h_left, w_left))
    right_mask_resized = resize_and_center(right_mask, (h_left, w_left))

    # 4. Create StereoSGBM object (you may need to tune these parameters)
    min_disp = 0
    num_disp = 64  # must be multiple of 16
    block_size = 5
    P1 = 8 * block_size * block_size
    P2 = 32 * block_size * block_size

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        preFilterCap=31,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.MODE_HH 
    )

    # Compute the disparity in 16-bit signed format, then convert to float
    disparity_16S = stereo.compute(left_img, right_img_resized)
    disparity_float = disparity_16S.astype(np.float32) / 16.0

    # 5. Combine masks: only keep pixels valid in both
    combined_mask = cv2.bitwise_and(left_mask, right_mask_resized)
    # Mask out invalid areas
    disparity_float[combined_mask == 0] = 0

    # Save a grayscale version of disparity (optional).
    # But the user specifically wants false color, so let's produce that.

    # 6. Produce a false-color disparity image
    disp_vis_color = compute_false_color_disparity(disparity_float)
    cv2.imwrite(output_disparity_path, disp_vis_color)
    print(f"False-color disparity saved to: {output_disparity_path}")

    # # Also, if you still want a single-channel disparity for further analysis:
    # # Normalize & save a plain grayscale disparity
    # disp_min = disparity_float.min()
    # disp_max = disparity_float.max()
    # if disp_max - disp_min < 1e-5:
    #     disp_vis_gray = np.zeros_like(disparity_float, dtype=np.uint8)
    # else:
    #     disp_vis_gray = 255 * (disparity_float - disp_min) / (disp_max - disp_min)
    #     disp_vis_gray = disp_vis_gray.astype(np.uint8)
    # cv2.imwrite(output_disparity_path, disp_vis_gray)
    # print(f"Grayscale disparity saved to: {output_disparity_path}")

    # 7. Warp the right image to produce a "matched" image
    #    We use the same shape as left_img, so right_img was already resized
    matched_img = warp_right_image_using_disparity(left_img, right_img_resized, disparity_float)
    cv2.imwrite(output_matched_image_path, matched_img)
    print(f"Matched (warped) right image saved to: {output_matched_image_path}")


if __name__ == "__main__":
    base_img_path = r'CTX_DEM_Retrieve\ortho_clipped.tif'
    base_dtm_path = r'CTX_DEM_Retrieve\tif\K06_055567_1983_XN_18N283W__K05_055501_1983_XN_18N283W_dtm.tif'
    warp_img_path = r'Playground\frt000161ef_07_sr167j_mtr3.img'
    warp_rgb = ('R2529', 'R1506', 'R1080')
    img1_gray, img1_alpha, base_img = pair.process_geotiff(base_img_path, warp_rgb)
    img2_gray, img2_alpha, img2prj = pair.process_geotiff(warp_img_path, warp_rgb)
    # Example usage
    stereo_match_original_vs_scaled(img1_gray, img1_alpha, img2_gray, img2_alpha,\
                             output_disparity_path="disparity_result.png",
                            output_matched_image_path="matched_right.png")
