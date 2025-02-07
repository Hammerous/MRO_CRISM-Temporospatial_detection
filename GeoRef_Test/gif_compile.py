import subprocess

# Configuration
input_image1 = r'CTX_DEM_Retrieve\Download Strategy\GIFs\effect_3.png'
input_image2 = r'CTX_DEM_Retrieve\Download Strategy\GIFs\effect_4.png'
output_gif = r'D:\2024Fall\Undergraduate Discourse\CTX_DEM_Retrieve\Download Strategy\GIFs\output_6.gif'

# FFmpeg command
ffmpeg_cmd = [
    "ffmpeg",
    "-y",  # Overwrite output file without asking
    "-loop", "1", "-t", "1", "-i", input_image1,
    "-loop", "1", "-t", "1", "-i", input_image2,
    "-filter_complex",
    "[0:v][1:v]concat=n=2:v=1:a=0,split[s0][s1];"
    "[s0]palettegen=stats_mode=diff[p];"
    "[s1][p]paletteuse=dither=bayer:bayer_scale=3",
    "-r", "10",  # Output frame rate (10 frames per second)
    "-loop", "0",  # Infinite looping
    output_gif
]

# Run the command
try:
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Successfully created GIF: {output_gif}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")
except FileNotFoundError:
    print("FFmpeg not found. Please install FFmpeg and ensure it's in your system PATH.")