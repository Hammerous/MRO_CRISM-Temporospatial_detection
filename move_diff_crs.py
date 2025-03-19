import os
import shutil

# Define source and destination folders
source_folder = r"IMG2SHP\Less Views"
destination_folder = r"IMG2SHP\Diff CRS"

# Ensure destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Loop through files in the source folder
for file_name in os.listdir(source_folder):
    if "crsCRS" in os.path.splitext(file_name)[0]:  # Check if "crs" is in the base name
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        
        # Move file to the new folder
        shutil.move(source_path, destination_path)
        print(f"Moved: {file_name}")

print("File transfer complete.")
