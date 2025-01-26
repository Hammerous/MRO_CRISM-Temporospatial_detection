import cv2
import numpy as np
from osgeo import osr, gdal

def read_geotiff(image_path):
    """
    Reads a GeoTIFF image and returns the image and its geotransform.
    """
    dataset = gdal.Open(image_path)
    geotransform = dataset.GetGeoTransform()
    crs = osr.SpatialReference()
    crs.ImportFromWkt(dataset.GetProjection())
    image = dataset.ReadAsArray()

    # Check if image has 4 channels (RGBA), if so, we separate it
    if image.shape[0] == 4:
        rgba_image = image
        rgb_image = rgba_image[:3, :, :]  # Keep only the RGB channels
        alpha_channel = rgba_image[3, :, :]  # Separate the alpha channel
    else:
        rgb_image = image
        alpha_channel = None

    return rgb_image, alpha_channel, geotransform, crs, dataset

def feature_matching(img1, img2):
    """
    Extracts features from both images using Canny edge detector
    and matches the features using ORB.
    """
    # Rescale images to uint8 type if necessary (normalize to 0-255 range)
    img1_scaled = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img2_scaled = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Canny edge detector to extract features
    edges1 = cv2.Canny(img1_scaled, 100, 200)
    edges2 = cv2.Canny(img2_scaled, 100, 200)

    # Use ORB detector to find keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(edges1, None)
    kp2, des2 = orb.detectAndCompute(edges2, None)

    # Use BFMatcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    # Extract matching points
    points1 = np.array([kp1[m.queryIdx].pt for m in matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in matches])

    return points1, points2

def apply_georeferencing(points1, points2, image_to_adjust, geotransform, crs):
    """
    Applies affine transformation to the reprojected image based on the matched points.
    """
    # Ensure image has 3 channels (RGB)
    if image_to_adjust.ndim == 3 and image_to_adjust.shape[0] == 3:
        # Split the image into individual RGB channels
        r_channel = image_to_adjust[0, :, :]
        g_channel = image_to_adjust[1, :, :]
        b_channel = image_to_adjust[2, :, :]
        
        # Find the homography matrix using the points
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Get the image size (height, width)
        height, width = r_channel.shape

        # Apply the homography to each channel separately
        r_adjusted = cv2.warpPerspective(r_channel, h, (width, height))
        g_adjusted = cv2.warpPerspective(g_channel, h, (width, height))
        b_adjusted = cv2.warpPerspective(b_channel, h, (width, height))

        # Stack the adjusted channels back into a 3-channel image
        adjusted_image = np.stack([r_adjusted, g_adjusted, b_adjusted], axis=0)

    else:
        # If the image is already grayscale (1 channel), apply the homography to it
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
        adjusted_image = cv2.warpPerspective(image_to_adjust, h, (image_to_adjust.shape[1], image_to_adjust.shape[0]))

    # Apply the transformation to the geotransform
    new_geotransform = list(geotransform)
    new_geotransform[0] = geotransform[0] + h[0, 0] * geotransform[1] + h[0, 1] * geotransform[5]
    new_geotransform[3] = geotransform[3] + h[1, 0] * geotransform[1] + h[1, 1] * geotransform[5]

    return adjusted_image, new_geotransform, crs

def save_georeferenced_image(output_path, adjusted_image, geotransform, crs, alpha_channel=None):
    """
    Saves the georeferenced image as a GeoTIFF with the same CRS and geotransform.
    If alpha channel is provided, it will be added back to the output.
    """
    driver = gdal.GetDriverByName('GTiff')
    band, rows, cols = adjusted_image.shape
    # Create the output dataset (1 band for RGB and 1 band for Alpha if provided)
    dataset_out = driver.Create(output_path, cols, rows, 3 if alpha_channel is None else 4, gdal.GDT_Byte)

    dataset_out.SetGeoTransform(geotransform)
    dataset_out.SetProjection(crs.ExportToWkt())

    dataset_out.GetRasterBand(1).WriteArray(adjusted_image[0, :, :])  # Red channel
    dataset_out.GetRasterBand(2).WriteArray(adjusted_image[1, :, :])  # Green channel
    dataset_out.GetRasterBand(3).WriteArray(adjusted_image[2, :, :])  # Blue channel

    if alpha_channel is not None:
        dataset_out.GetRasterBand(4).WriteArray(alpha_channel)  # Write the alpha channel

def main():
    # Step 1: Read the images
    img_clipped, alpha_clipped, geotransform_clipped, crs_clipped, dataset_clipped = read_geotiff(r'CTX_DEM_Retrieve\clipped.tif')
    img_reprojected, alpha_reprojected, geotransform_reprojected, crs_reprojected, dataset_reprojected = read_geotiff(r'CTX_DEM_Retrieve\reprojected.tif')

    # Step 2: Convert reprojected image to grayscale (using only RGB channels if RGBA)
    img_reprojected_gray = cv2.cvtColor(img_reprojected.transpose(1, 2, 0), cv2.COLOR_BGR2GRAY) # Change shape to (1091, 981, 3)

    # Step 3: Feature matching
    points1, points2 = feature_matching(img_clipped, img_reprojected_gray)

    # Step 4: Apply georeferencing
    orthorectified_image, new_geotransform, new_crs = apply_georeferencing(
        points1, points2, img_reprojected, geotransform_clipped, crs_clipped)

    # Step 5: Save the newly orthorectified image (with or without alpha channel)
    save_georeferenced_image(r'CTX_DEM_Retrieve\orthorecitified.tif', orthorectified_image, new_geotransform, new_crs, alpha_channel=alpha_reprojected)

if __name__ == "__main__":
    main()
