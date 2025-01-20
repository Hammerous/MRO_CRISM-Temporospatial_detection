import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np

# Define the folder containing the .img files
input_folder = "Playground"
output_file = "overlapping_area_geotiff.tif"

# Step 1: Read in all files and extract their geographic coverage in the first band
def read_rasters(input_folder):
    raster_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.img')]
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
    meta = rasters[0].meta.copy()

    # Calculate new dimensions and transform for the common extent
    transform = rasterio.transform.from_bounds(
        *common_extent, 
        width=int((common_extent[2] - common_extent[0]) / meta['transform'][0]),
        height=int((common_extent[3] - common_extent[1]) / -meta['transform'][4])
    )

    for raster in rasters:
        # Clip each raster to the common extent
        window = from_bounds(*common_extent, transform=raster.transform)
        data = raster.read(1, window=window, boundless=True)
        
        # Replace no-data values with NaN
        nodata_value = raster.nodatavals[0] if raster.nodatavals else None
        if nodata_value is not None:
            data[data == nodata_value] = np.nan

        data_stack.append(data)

    # Update metadata for the output GeoTIFF
    meta.update({
        'driver': 'GTiff',
        'height': data_stack[0].shape[0],
        'width': data_stack[0].shape[1],
        'transform': transform,
        'count': len(data_stack),  # Number of bands
    })

    return np.stack(data_stack, axis=0), meta

# Step 4: Trim the commonly covered pixels (set all no-data pixels to no-data in the stack)
def trim_no_data(stack, nodata_value):
    mask = np.any(np.isnan(stack), axis=0)  # Find locations where any layer has NaN
    stack[:, mask] = nodata_value  # Set all no-data areas
    return stack

# Main function
def main(input_folder, output_file):
    rasters = read_rasters(input_folder)
    common_extent = find_common_extent(rasters)
    stack, meta = stack_trimmed_layers(rasters, common_extent)

    # Set nodata value for all bands (preserve consistency)
    nodata_value = meta.get('nodata', 65535)  # Default to 65535 if not defined
    stack = trim_no_data(stack, nodata_value=nodata_value)

    # Save the trimmed and stacked layers to a new GeoTIFF
    meta.update({'nodata': nodata_value})
    with rasterio.open(output_file, 'w', **meta) as dst:
        for i in range(stack.shape[0]):
            dst.write(stack[i], i + 1)

    print(f"GeoTIFF saved to {output_file}")

if __name__ == "__main__":
    main(input_folder, output_file)