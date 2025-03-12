import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np
from osgeo import gdal

# Define the folder containing the .img files
input_folder = "Playground"
output_file = "overlapping_area_geotiff.tif"

# Step 1: Read in all files and extract their geographic coverage in the first band
def read_rasters(input_folder):
    raster_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if (f.endswith('.img') or f.endswith('.tif'))]
    rasters = []
    for file in raster_files:
        src = rasterio.open(file)
        rasters.append(src)
    return rasters

# Step 2: Find the common overlapping area of all images
def find_common_extent(rasters):
    common_extent = rasters[0].bounds
    for raster in rasters[1:]:
        common_extent = (
            max(common_extent[0], raster.bounds.left),
            max(common_extent[1], raster.bounds.bottom),
            min(common_extent[2], raster.bounds.right),
            min(common_extent[3], raster.bounds.top)
        )
    return common_extent

# Step 3: Stack all trimmed layers with the same spatial extent
def stack_trimmed_layers(rasters, common_extent):
    data_stack = []
    band_names = []  # List to store band descriptions
    meta = rasters[0].meta.copy()

    # Calculate new dimensions and transform for the common extent
    transform = rasterio.transform.from_bounds(
        *common_extent, 
        width=int((common_extent[2] - common_extent[0]) / meta['transform'][0]),
        height=int((common_extent[3] - common_extent[1]) / -meta['transform'][4])
    )

    # Process each raster
    for raster in rasters:
        # Define the window to clip the raster to the common extent
        window = rasterio.windows.from_bounds(*common_extent, transform=raster.transform)
        
        # Read all bands at once with masking for no-data and out-of-bounds areas
        data = raster.read(window=window, boundless=True, masked=True)
        
        # Convert masked array to a regular array with NaN for masked values
        data = data.filled(np.nan)
        
        # Append the 3D array (bands x height x width) to the stack
        data_stack.append(data)
        
        raster_affix = os.path.basename(raster.name).split(".")[0].split("_")[-1]
        descriptions = raster.descriptions if raster.descriptions and any(raster.descriptions) else [f"Band {i+1}" for i in range(raster.count)]
        band_names.extend([f"{desc}_{raster_affix}" for desc in descriptions])

    # Combine all bands from all rasters into a single 3D array
    stacked_data = np.concatenate(data_stack, axis=0)

    # Update metadata for the output GeoTIFF
    meta.update({
        'driver': 'GTiff',
        'height': stacked_data.shape[1],
        'width': stacked_data.shape[2],
        'transform': transform,
        'count': stacked_data.shape[0],  # Total number of bands across all rasters
    })
    return stacked_data, meta, band_names

# Step 4: Trim the commonly covered pixels (set all no-data pixels to no-data in the stack)
def trim_no_data(stack, nodata_value):
    mask = np.any(np.isnan(stack), axis=0)  # Find locations where any layer has NaN
    stack[:, mask] = nodata_value  # Set all no-data areas
    return stack

# Main function
def main(input_folder, output_file):
    rasters = read_rasters(input_folder)
    common_extent = find_common_extent(rasters)
    stack, meta, band_names = stack_trimmed_layers(rasters, common_extent)  # Now returns band_names

    # Set nodata value for all bands (preserve consistency)
    nodata_value = meta.get('nodata', 65535)  # Default to 65535 if not defined
    stack = trim_no_data(stack, nodata_value=nodata_value)

    # Save the trimmed and stacked layers to a new GeoTIFF
    meta.update({})
    # Update metadata with compression and multi-threading options
    meta.update({
        'compress': 'LZW',       # Use a supported compression method
        'NUM_THREADS': 'ALL_CPUS',    # Use all available CPU cores for compression
        'nodata': nodata_value
    })
    with rasterio.open(output_file, 'w', **meta) as dst:
        for i in range(stack.shape[0]):
            dst.write(stack[i], i + 1)
            dst.set_band_description(i + 1, band_names[i])  # Set the band description

    print(f"GeoTIFF saved to {output_file}")

if __name__ == "__main__":
    main(input_folder, output_file)