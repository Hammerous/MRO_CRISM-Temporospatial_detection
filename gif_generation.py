import os
import subprocess
import pandas as pd
from osgeo import gdal
import numpy as np
from Toolbox.raster_manipulate import DS2RGB

# --- Parameters ---
#red_band_name   = "R3920"
red_band_name   = "BD1900_2"
green_band_name = "BD1500_2"
blue_band_name  = "BD1435"

folder_name = "7-0"
gif_name = "ICE_animated.gif"

csv_file = os.path.join("CRISM_Metadata_Database", f"{folder_name}.csv")
input_folder = os.path.join("Round2_trimed", folder_name)
output_folder = os.path.join(input_folder, "output")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

## ---------------- Band definitions ------------------
# red_band_name   = "SINDEX2"
# green_band_name = "BD2100_2"
# blue_band_name  = "BD1900_2" 
rgb_lst = [red_band_name, green_band_name, blue_band_name]

# ----------------------------------------------------
def find_band_indices(ds, rgb_lst):
    """
    Return a list of band indices (1-based) corresponding to band_names.
    If any band is missing, returns None.
    """
    name_to_index = {}
    for i in range(1, ds.RasterCount + 1):
        desc = ds.GetRasterBand(i).GetDescription()
        #print(desc)
        if desc in rgb_lst:
            name_to_index[desc] = i

    # Preserve the order in band_names
    indices = [name_to_index.get(name) for name in rgb_lst]
    return None if any(idx is None for idx in indices) else indices

# ---------------- Main processing -------------------
df = pd.read_csv(csv_file)
product_ids = df["ProductId"].tolist()
saved_files = []
from PIL import Image, ImageDraw, ImageFont 
for pid in product_ids:
    tif_path = os.path.join(input_folder, f"{pid}.tif")
    if not os.path.exists(tif_path):
        print(f"[WARN] {tif_path} not found; skipping.")
        continue

    ds = gdal.Open(tif_path)
    if ds is None:
        print(f"[WARN] Could not open {tif_path}; skipping.")
        continue

    band_idxs = find_band_indices(ds, rgb_lst)
    if band_idxs is None:
        print(f"[WARN] Missing at least one RGB band in {pid}; skipping.")
        continue

   # --- Build RGB + Alpha ---
    rgb, alpha = DS2RGB(ds, band_idxs)          # alpha is a 2‑D uint8 array (0‑255)

    # --- 1. grab the UTCstart string that belongs to this ProductId ---
    row = df.loc[df["ProductId"] == pid].iloc[0]          # row is a Series
    label_text = f"UTCstart: {row['UTCstart'].split('T')[0]}\nSolLong: {row['SolLong']:.2f}\nInAngle: {row['InAngle']:.2f}"

    # --- 2. draw it on the RGB image ---
    rgb_img = Image.fromarray(rgb, mode="RGB")
    draw    = ImageDraw.Draw(rgb_img)

    # pick a font; fall back to the default if the TTF isn’t found
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    # put a tiny shadow behind the text so it stays legible on bright pixels
    xy = (10, 10)                                 # upper‑left corner, 10‑px padding
    draw.text(xy, label_text, fill=(0, 0, 0),  font=font, stroke_width=3, stroke_fill=(0, 0, 0))
    draw.text(xy, label_text, fill=(255, 255, 255), font=font)

    # convert back to numpy so the rest of your pipeline is unchanged
    rgb = np.asarray(rgb_img, dtype=np.uint8)

    # --- Save 4‑band LZW‑compressed GeoTIFF ---
    out_file = os.path.join(output_folder, f"{pid}_falsecolor.tif")
    driver   = gdal.GetDriverByName("GTiff")

    out_ds = driver.Create(
        out_file,
        ds.RasterXSize,
        ds.RasterYSize,
        3,
        #4,                     # <‑‑ 4 bands: R, G, B, A
        gdal.GDT_Byte,
        options=[
            "COMPRESS=LZW",
            "PHOTOMETRIC=RGB",  # tell readers this is RGB
            "ALPHA=YES"         # advertise the 4th band as alpha (GDAL ≥3.3)
        ]
    )

    # Write RGB planes
    for i in range(3):
        out_ds.GetRasterBand(i + 1).WriteArray(rgb[:, :, i])

    # Write Alpha plane
    #out_ds.GetRasterBand(4).WriteArray(alpha)

    # Optional: set colour interpretations for clarity
    out_ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
    out_ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
    out_ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
    #out_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)

    out_ds.FlushCache()
    out_ds = None
    ds     = None

    print(f"[INFO] Saved {out_file}")
    saved_files.append(out_file)

# ---------------- Assemble GIF with custom ffmpeg pipeline -----------
if saved_files:
    gif_path = os.path.join(input_folder, gif_name)
    print("[INFO] Creating high-quality GIF with palette optimization...")

    # Build ffmpeg command dynamically
    ffmpeg_cmd = ["ffmpeg", "-y"]
    
    # Add each image as input with 1s display time
    for img in saved_files:
        ffmpeg_cmd += ["-loop", "1", "-t", "1", "-i", img]
    
    # Build filter_complex part
    input_count = len(saved_files)
    stream_tags = "".join(f"[{i}:v]" for i in range(input_count))
    filter_complex = (
        f"{stream_tags}concat=n={input_count}:v=1:a=0,split[s0][s1];"
        "[s0]palettegen=stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    
    ffmpeg_cmd += [
        "-filter_complex", filter_complex,
        "-r", "10",        # Output frame rate
        "-loop", "0",      # Infinite loop
        gif_path
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    print(f"[DONE] Animated GIF saved to {gif_path}")
else:
    print("[WARN] No images were generated; GIF not created.")