#!/usr/bin/env /cluster/home/jstras02/cf/python
import sys
sys.path.append('/cluster/home/jstras02/.conda/envs/rhel8/2024.06-py311/extra/lib/python3.10/site-packages')
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def getdrugs(base_path):

    """
    Returns a list of drugs based on the files in the 'aggregated' folder
    within the 'Day0' subdirectory.
    
    Args:
        base_path (str or Path): The base directory containing the 'Day0/aggregated' folder.
    
    Returns:
        list: A list of unique drug identifiers (e.g., 'A', 'B', 'C').
    """
    drugs = set()
    aggregated_folder = Path(base_path) / 'Day0' / 'aggregated'
    
    if not aggregated_folder.exists():
        return []  # Return an empty list if the folder does not exist
    
    for file in aggregated_folder.iterdir():
        if file.is_file() and file.name[0].isalpha():  # Check if it's a file and starts with a letter
            drugs.add(file.name[0].upper())  # Add the first character as the drug identifier

    return sorted(drugs)  # Return a sorted list of drugs

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def roseplot(base_path):
    """
    Generates a rose plot of cell migration trajectories and 
    enforces consistent x and y axis limits (±150 µm) for comparison.

    - base_path: Base path to save the plots.

    Returns:
    - A DataFrame with shifted X and Y positions for each track.
    """
    days = ['Day0', 'Day1', 'Day2']
    drugs = ['A', 'B', 'C']
    concs = ['a', 'b', 'c', 'd', 'Ctrl']
    combos = [d + c for c in concs for d in drugs]
    filepaths = [c + '_roseplot_data.csv' for c in combos]
    total_paths = [d + '/aggregated/' + f for f in filepaths for d in days]

    for t in total_paths:
        day = t.split('/')[0]
        roseplot_path = base_path / t
        if roseplot_path.is_file():
            df = pd.read_csv(roseplot_path)

            # Set figure size back to (10, 10) for quality
            fig, ax = plt.subplots(figsize=(10, 10))

            # Loop through each track ID and plot its trajectory
            for track_id in df['all_ids'].unique():
                df_filtered = df[df['all_ids'] == track_id].copy()
                ax.plot(df_filtered['shifted_POSITION_X'], df_filtered['shifted_POSITION_Y'], label=f'Track {track_id}')

            # Enforce consistent axis limits
            ax.set_xlim(-300, 300)  # ±300 µm
            ax.set_ylim(-300, 300)  # ±300 µm
            tick_interval = 100
            ax.set_xticks(np.arange(-300, 301, tick_interval))
            ax.set_yticks(np.arange(-300, 301, tick_interval))

            font_size = 36  # Adjust as needed

            ax.set_title('Cell Migration Rose Plot', fontsize=font_size, pad=30)
            ax.set_xlabel('X-Distance (µm)', fontsize=font_size, labelpad=20)
            ax.set_ylabel('Y-Distance (µm)', fontsize=font_size, labelpad=20)
    

            ax.tick_params(axis='both', which='major', labelsize=font_size, pad=10)
            ax.grid(True)

            plt.tight_layout()

            # Save the rose plot in 'output/roseplots/[day]'
            output_folder = base_path / 'output' / 'roseplots' / day
            output_folder.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            name_parts = roseplot_path.name.split('_')
            graph_filename = "_".join(name_parts[:2]) + '.png'
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
            fig.savefig(output_folder / graph_filename, bbox_inches='tight', pad_inches=0.5)
            plt.show()
            plt.close(fig)
        else: 
            print(f'{t} does not exist')
            

import pandas as pd
from pathlib import Path

def process_velocity_data(base_path, drugs, column_name):
    """
    Process velocity data by concatenating specific columns from multiple files across days.
    
    Parameters:
    - base_path (str or Path): Base path to the data directory.
    - column_name (str): Name of the column to extract ('max' or 'avg').
    
    Saves:
    - A CSV file for each day in the 'collective_results' directory.
    """
    # Convert base_path to a Path object if it's a string
    base_path = Path(base_path)

    # Days to process
    days = ['Day0', 'Day1', 'Day2']
    concs = ['a', 'b', 'c', 'd', 'Ctrl']
    filename = '_velocity_data.csv'


    # Directory to save results
    results_path = base_path / 'output'/ 'velocity_tables'
    results_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    for drug in drugs: 
        names = [drug + conc for conc in concs]
        filenames = [name + filename for name in names]
        dataframes_per_day = []

        # Process each day
        for day in days:
            # Define the path for the current day
            day_path = base_path / day / 'aggregated'

            # Load DataFrames and extract the relevant column ('max_velocity' or 'avg_velocity')
            mf = [
                pd.read_csv(day_path / filename)[f'{column_name}_velocity'] 
                for filename in filenames
                if (day_path / filename).exists()
            ]

            # Concatenate the columns side by side
            velocity_df = pd.concat(mf, axis=1).reset_index(drop=True)

            # Assign column names for clarity
            velocity_df.columns = names

            # Save the result to the collective_results directory
            output_file = results_path / f"{day}_{drug}_{column_name}_velocity.csv"
            velocity_df.to_csv(output_file, index=False)

            print(f"Saved: {output_file}")

from PIL import Image
import cv2
import numpy as np
from pathlib import Path

def makeVideo(df, base_path, red_path, green_path, red_segmented_path, green_segmented_path):
    
    red_stack = Image.open(red_path)
    green_stack = Image.open(green_path)
    red_mask_stack = Image.open(red_segmented_path)
    green_mask_stack = Image.open(green_segmented_path)

    # Load the merged tracks DataFrame
    merged_tracks_df = df.copy()
    print(f'columns: {merged_tracks_df.columns}')

    if merged_tracks_df.empty:
        print("Warning: merged_tracks_df is empty. Skipping video generation.")
        return
    
    # Get the dimensions and number of frames
    width, height = red_stack.size
    num_frames = red_stack.n_frames

    # Create a VideoWriter object using OpenCV, outputting an MP4 video
    output_video_path = str(base_path / "trackmovie.mp4")  # Save as .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for mp4
    writer = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

    for i in range(num_frames):
        # Seek to the current frame in each stack
        red_stack.seek(i)
        green_stack.seek(i)
        red_mask_stack.seek(i)
        green_mask_stack.seek(i)
        
        # Convert the current frames to numpy arrays
        red_img = np.array(red_stack, dtype=np.float32)
        green_img = np.array(green_stack, dtype=np.float32)
        red_mask = np.array(red_mask_stack)
        green_mask = np.array(green_mask_stack)

        # Normalize the intensity images for visualization
        red_img_norm = red_img / red_img.max() if red_img.max() > 0 else red_img
        green_img_norm = green_img / green_img.max() if green_img.max() > 0 else green_img

        # Create an RGB image from the base red and green images
        overlay = np.zeros((height, width, 3), dtype=np.float32)
        overlay[:, :, 0] = red_img_norm  # Red channel
        overlay[:, :, 1] = green_img_norm  # Green channel

        # Create a transparent overlay for the masks
        transparent_overlay = np.zeros((height, width, 3), dtype=np.float32)

        # Apply semi-transparent red for red mask regions
        transparent_overlay[:, :, 0][red_mask > 0] = 1.0  # Fully color red mask areas in red
        
        # Apply semi-transparent green for green mask regions
        transparent_overlay[:, :, 1][green_mask > 0] = 1.0  # Fully color green mask areas in green
        
        # Where both masks overlap, color it yellow
        both_mask = (red_mask > 0) & (green_mask > 0)
        transparent_overlay[:, :, 0][both_mask] = 1.0  # Red channel (yellow = red + green)
        transparent_overlay[:, :, 1][both_mask] = 1.0  # Green channel (yellow = red + green)

        # Blend the transparent overlay with the base image
        blended_image = overlay + 0.5 * transparent_overlay  # Overlay with 50% transparency
        
        # Clip the blended image to make sure values are within the valid range [0, 1]
        blended_image = np.clip(blended_image, 0, 1)

        # Convert the overlay to 8-bit image for video writing
        overlay_uint8 = np.uint8(blended_image * 255)
        
        # Get the track data for the current frame
        frame_data = merged_tracks_df[merged_tracks_df['frame'] == i]
        track_ids = frame_data['all_ids'].values
        x_coords = frame_data['POSITION_X'].values
        y_coords = frame_data['POSITION_Y'].values
        emt_status = frame_data['EMT'].values  # Get the 'EMT' column values
        
        # Draw the track IDs and EMT status on the frame
#         for j in range(len(track_ids)):
#             text = f'ID: {track_ids[j]} ({emt_status[j]})'  # Include EMT status
#             cv2.putText(overlay_uint8, text, 
#                         (int(x_coords[j]), int(y_coords[j])), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Write the frame to the video using OpenCV
        writer.write(cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    # Release the writer object
    writer.release()
    print(f"Video with tracks saved at: {output_video_path}")
import pandas as pd

def safe_load_csv(filepath):
    try:
        if filepath.stat().st_size == 0:
            print(f"Skipping {filepath}: File is empty.")
            return None  # Return None for empty files
        
        df = pd.read_csv(filepath, dtype=str, low_memory=False)  # Read as strings to avoid mixed type issues

        # Convert necessary columns to numeric (handle errors)
        for col in ['POSITION_X', 'POSITION_Y', 'EMT']:  # Adjust based on your real column names
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert while handling errors
        
        return df

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None  # Return None for problematic files

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <plate_path> <function_name>")
        sys.exit(1)
    
    root_path = Path(sys.argv[1]) # root which contains Plates
    function = sys.argv[2]  # Second argument: Function to execute

    plates = ['NS', 'P1', 'P2', 'P3', 'P4', 'P5']
    days = ['Day0', 'Day1', 'Day2']
    wells = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11',
             'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11',
             'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11',
             'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11',
             'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11',
             'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11']

    for p in plates: 
        base_path = root_path / p
        
        drugs = getdrugs(base_path)
        print("Identified drugs:", drugs)

        if function == 'roseplot':
            print(f'Working on plate {p} at {base_path}')
            roseplot(base_path)
        elif function == 'velocity':
            print(f'Working on plate {p} at {base_path}')
            process_velocity_data(base_path, drugs, column_name='max')
            process_velocity_data(base_path, drugs, column_name='average')
        elif function == 'video':
            print(f'Generating videos for plate {p} at {base_path}')
            for day in days:
                for well in wells:
                    well_path = base_path / day / well
                    df_path = well_path / 'data' / 'all_tracks_df.csv'

                    if not df_path.exists():
                        print(f"Skipping {well_path}: No all_tracks_df.csv found.")
                        continue  # Skip if file doesn't exist

                    df = safe_load_csv(df_path)
                    if df is None:
                        continue  # Skip processing if CSV is empty or invalid


                    red_path = well_path / 'red.tif'
                    green_path = well_path / 'green.tif'
                    red_segmented_path = well_path / 'red_segmented.tiff'
                    green_segmented_path = well_path / 'green_segmented.tiff'

                    # Check if required files exist before processing
                    if not all([red_path.exists(), green_path.exists(), red_segmented_path.exists(), green_segmented_path.exists()]):
                        print(f"Skipping {well_path}: Missing one or more TIFF files.")
                        continue

                    print(f'Processing {well_path}...')
                    makeVideo(df, well_path, red_path, green_path, red_segmented_path, green_segmented_path)

        else:
            print(f"Function '{function}' not recognized.")

if __name__ == "__main__":
    main()