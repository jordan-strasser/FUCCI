import os
from PIL import Image
import re
from pathlib import Path

# Define source and destination directories
source_dir = Path(input('paste root directory: '))
destination_base_dir = Path(input('paste output directory for stacks: '))

# Ensure destination directory exists
destination_base_dir.mkdir(parents=True, exist_ok=True)

# Regular expressions to extract well and time info
well_pattern = re.compile(r'_([A-Z]\d{2})_')
time_pattern = re.compile(r't(\d+)_')

# Define channel mappings
channels = {'w2': 'green.tif', 'w3': 'red.tif'}

# Get all files in the source directory, skipping hidden files
files = [f for f in source_dir.iterdir() if f.is_file() and not f.name.startswith('.')]

# Extract and sort unique well IDs
wells = sorted({well_pattern.search(f.name).group(1) for f in files if well_pattern.search(f.name)})

# Process each well
for well_id in wells:
    well_dir = destination_base_dir / well_id
    
    # Check if the well folder already exists, and skip if it does
    if well_dir.exists():
        print(f"Skipping well {well_id} - folder already exists.")
        continue

    well_dir.mkdir(parents=True, exist_ok=True)  # Create well directory

    print(f"Processing well: {well_id}")

    # Create TIFF stacks for each channel
    for channel, output_name in channels.items():
        images = []
        well_files = sorted(
            [f for f in files if f"_{well_id}_" in f.name and f"_s1_{channel}_" in f.name],
            key=lambda x: int(time_pattern.search(x.name).group(1))
        )

        print(f"Channel: {channel}, Files: {well_files[:5]}... ({len(well_files)} total)")

        for file_path in well_files:
            with Image.open(file_path) as img:
                images.append(img.copy())

        if images:
            output_path = well_dir / output_name
            
            # Save as an uncompressed TIFF stack
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                compression=None,  # No compression
                bits=16  # Ensure 16-bit saving if applicable
            )
            print(f"Saved {output_name} as a TIFF stack in {well_dir}.")
        else:
            print(f"No images found for channel {channel} in well {well_id}.")

print("All TIFF stacks have been created.")
