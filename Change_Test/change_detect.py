import os
import json
import rasterio
import numpy as np

def compute_stats(data, nodata):
    """
    Compute summary statistics (count, min, max, mean, std) over valid pixels.
    
    Parameters:
        data (np.ndarray): Input array of shape (bands, height, width).
        nodata (numeric or None): The nodata value to ignore in calculations.
    
    Returns:
        dict: Statistics for valid data.
    """
    if nodata is not None:
        # Create a boolean mask for valid pixels.
        valid_data = data[data != nodata]
    else:
        valid_data = data.flatten()
    
    if valid_data.size > 0:
        return {
            'count': int(valid_data.size),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data))
        }
    else:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'std': None}

def process_geotiff(input_file, output_file):
    """
    Process a GeoTIFF with at least 120 bands by computing the difference between bands 61-120 and bands 1-60.
    
    The function reads the first 120 bands, computes summary statistics for bands 1-60 and bands 61-120,
    then computes a difference (band_{i+60} - band_i) only in the valid data region (i.e. where both bands are valid).
    The difference is saved to a new GeoTIFF with 60 bands (dtype float32) and summary statistics for the difference
    are computed and returned.
    
    Parameters:
        input_file (str): Path to the input GeoTIFF.
        output_file (str): Path to the output GeoTIFF.
    
    Returns:
        dict: A dictionary containing statistics for the first set of bands, the second set, and the difference.
              Example:
              {
                  'bands1_stats': { ... },
                  'bands2_stats': { ... },
                  'diff_stats': { ... }
              }
    """
    with rasterio.open(input_file) as src:
        if src.count < 120:
            raise ValueError("Input file must have at least 120 bands")
        
        # Read bands 1 to 120 into a 3D array of shape (120, height, width)
        all_bands = src.read(range(1, 121))
        
        # Split into the first 60 bands and the subsequent 60 bands
        bands1 = all_bands[:60]
        bands2 = all_bands[60:120]
        
        # Get the nodata value (could be None)
        nodata = src.nodata
        
        # Compute statistics for each set before the difference
        stats_bands1 = compute_stats(bands1, nodata)
        stats_bands2 = compute_stats(bands2, nodata)
        
        # Compute the difference only on valid data regions
        if nodata is not None:
            valid_mask = (bands1 != nodata) & (bands2 != nodata)
            diff = np.full(bands1.shape, nodata, dtype='float32')
            diff[valid_mask] = bands2[valid_mask] - bands1[valid_mask]
        else:
            diff = bands2 - bands1
        
        # Compute statistics for the difference image (ignoring nodata)
        stats_diff = compute_stats(diff, nodata)
        
        # Update metadata for the output file
        meta = src.meta.copy()
        meta.update({
            'count': 60,         # Output file will have 60 bands
            'dtype': 'float32'   # Use float32 to store differences
        })
    
    # Write the difference image to a new GeoTIFF file
    with rasterio.open(output_file, 'w', **meta, compress='LZW', num_threads="ALL_CPUS") as dst:
        dst.write(diff.astype('float32'))
    
    return {
        'bands1_stats': stats_bands1,
        'bands2_stats': stats_bands2,
        'diff_stats': stats_diff
    }

def main(input_folder, output_folder):
    """
    Process all .tif files in the input folder by applying the process_geotiff function.
    
    For each input file:
      - A change detection image (output GeoTIFF) is saved to the output folder with the same file name.
      - A corresponding JSON file containing statistics is saved with a '_stats.json' suffix.
    
    Parameters:
        input_folder (str): Path to the folder containing input .tif files.
        output_folder (str): Path to the folder where output files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process only files in the given folder (non-recursive)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tif'):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            
            try:
                stats = process_geotiff(input_file_path, output_file_path)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                continue
            
            # Save the statistics as a JSON file paired with the image file
            stats_filename = os.path.splitext(filename)[0] + '_stats.json'
            stats_file_path = os.path.join(output_folder, stats_filename)
            with open(stats_file_path, 'w') as f:
                json.dump(stats, f, indent=4)
            
            print(f"Processed {filename}: change detection image and stats saved.")

if __name__ == '__main__':
    input_folder = "Round1"
    output_folder = "Round1_CD"
    main(input_folder, output_folder)