#!/usr/bin/env /cluster/home/jstras02/cf/python
import sys
# USER INPUT TO PYTHON PATH
sys.path.append('/cluster/home/jstras02/.conda/envs/rhel8/2024.06-py311/extra/lib/python3.10/site-packages')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.measure import label
import os
import subprocess
import psutil
import confluentfucci as cf
from pathlib import Path
import pandas as pd
print(f'GPU is enabled: {cf.gui.check_cellpose_gpu()}')

# segmentation step to bypass Cellpose
import numpy as np
import h5py
import tifffile
from aicsimageio.writers import OmeTiffWriter
from tqdm import trange
from pathlib import Path


def threshold_segment_frame(image, threshold):
    """Simple threshold-based segmentation."""
    # Apply the threshold to create a binary mask
    masks = np.where(image > threshold, 1, 0)
    
    # Optionally, you can compute the center of mass for further tracking, but it's not necessary here
    # center_of_mass = None  # Implement if needed

    return masks

def segment_stack(
    path: Path,
    threshold: float,
    export_tiff: bool = True,
    panel_red_tqdm_instance=None,
    panel_green_tqdm_instance=None,
):
    """Segment stack frame by frame using simple pixel thresholding."""
    print(f"Segmenting stack at {path} using threshold {threshold}")
    stack = cf.utils.read_stack(path)  # Assuming this is a function that reads the TIFF stack
    print(stack.shape)
    frames, Y, X = stack.shape
    

    new_file_path = path.parent / f"{path.stem}_segmented.h5"
    dataset_name = "data"

    with h5py.File(new_file_path, "w") as f:
        dataset = f.create_dataset(
            dataset_name,
            shape=(frames, Y, X),
            dtype=np.uint16,
            chunks=True,
        )

        # Set up progress bar
        if panel_red_tqdm_instance:
            frame_iterator = panel_red_tqdm_instance(
                range(frames), desc="Red", colour="#ff0000"
            )
        elif panel_green_tqdm_instance:
            frame_iterator = panel_green_tqdm_instance(
                range(frames), desc="Green", colour="#008000"
            )
        else:
            frame_iterator = trange(frames)

        # Loop through each frame in the stack and apply threshold segmentation
        for frame in frame_iterator:
            masks = threshold_segment_frame(stack[frame, ...], threshold)
            dataset[frame, :, :] = masks

        # Export the segmentation as a TIFF file
        if export_tiff:
            new_tiff_path = path.parent / f"{path.stem}_segmented.tiff"
            print(f"Exporting to TIFF at {new_tiff_path}")
            OmeTiffWriter.save(f.get(dataset_name).__array__(), new_tiff_path, dim_order="TYX")

    print("Segmentation complete")


def _run_local_trackmate(settings_path: Path, data_path: Path):
    print("RUN LOCAL DOCKER")
    # cmd = f"/opt/fiji/ImageJ-linux64 --ij2 --headless --console --memory=$MEMORY --run read_settings_and_process_tiff_stack.py"
    cmd = [
        "/opt/Fiji.app/ImageJ-linux64",
        "--ij2",
        "--headless",
        "--console",
        f"--memory={int(psutil.virtual_memory().total // 1024 ** 3 * 0.75)}G",
        "--run",
        str('/cluster/home/jstras02/alpha/read_settings_and_process_tiff_stack.py'),
    ]

    env = {
        **os.environ,
        "DOCKER": str(True),
        "DOCKER_SETTINGS_XML": str(settings_path.absolute()),
        "DOCKER_TIFF_STACK": str(data_path.absolute()),
        # "MEMORY": f"{int(psutil.virtual_memory().total // 1024 ** 3 * 0.5)}G",
    }
    subprocess.run(cmd, env=env)


def addEMTColumn(merged_tracks_df, base_path, vmax_threshold, conv):
    """
    Classify tracks as attached or detached based on max velocity, 
    and return an additional DataFrame with max and average velocity per track.

    Parameters:
    - df: The DataFrame containing the track data.
    - base_path: Path for saving plots and other outputs.
    - vmax_threshold: Velocity threshold to classify tracks as attached or detached.

    Modifies:
    - Adds a column 'EMT' to the input DataFrame ('Attached' or 'Detached' based on velocity threshold).

    Returns:
    - The modified DataFrame with the new 'EMT' column.
    - A DataFrame with max and average velocity for each track.
    """

    if merged_tracks_df.empty:
        print("Warning: merged_tracks_df is empty. Skipping EMT calculations.")
        return pd.DataFrame(), pd.DataFrame()

    if 'all_ids' not in merged_tracks_df.columns:
        raise KeyError("Error: 'all_ids' column is missing in merged_tracks_df.")


    time_between_frames = conv # Time interval between frames
    um_per_pixel = 1.3556  # Conversion factor from pixels to micrometers

    # Calculate velocity between consecutive frames
    merged_tracks_df['dx'] = merged_tracks_df.groupby('all_ids')['POSITION_X'].diff().fillna(0)
    merged_tracks_df['dy'] = merged_tracks_df.groupby('all_ids')['POSITION_Y'].diff().fillna(0)

    # Calculate distance and velocity
    merged_tracks_df['distance'] = np.sqrt(merged_tracks_df['dx']**2 + merged_tracks_df['dy']**2) * um_per_pixel
    merged_tracks_df['velocity'] = merged_tracks_df['distance'] / time_between_frames

    # Calculate max and average velocity for each track
    velocity_summary = merged_tracks_df.groupby('all_ids')['velocity'].agg(
        max_velocity='max', average_velocity='mean'
    ).reset_index()

    # Add 'EMT' column based on the velocity threshold
    merged_tracks_df['EMT'] = merged_tracks_df['all_ids'].map(
        lambda track_id: 'Attached' if velocity_summary.loc[
            velocity_summary['all_ids'] == track_id, 'max_velocity'
        ].values[0] < vmax_threshold else 'Detached'
    )

#     # Plot the distribution of max velocities
#     plt.figure(figsize=(8, 6))
#     plt.hist(velocity_summary['max_velocity'], bins=20, color='blue', edgecolor='black')
#     plt.title('Distribution of Max Velocities Per Track')
#     plt.xlabel('Max Velocity (μm/hour)')
#     plt.ylabel('Number of Tracks')

#     # Save the histogram plot
#     output_folder = base_path / 'graphs'
#     output_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
#     plt.savefig(output_folder / 'MaxVelocityDistribution.png')
#     plt.close()

    # Return the modified DataFrame and the velocity summary DataFrame
    return merged_tracks_df, velocity_summary


def filterEMT(df):

    # Filter the DataFrame for 'Attached' and 'Detached'
    attached_df = df[df['EMT'] == 'Attached']
    detached_df = df[df['EMT'] == 'Detached']
    
    return attached_df, detached_df

def filter_duration(df, lim):
    num = lim * 0.9
    # Group by 'all_ids' and calculate the length (count of frames) for each track
    df_summary = df.groupby('all_ids').agg(
        length=('frame', 'count'),
        x_std=('POSITION_X', 'std'),
        y_std=('POSITION_Y', 'std')
    )

    # Filter tracks that meet the minimum frame condition
    valid_tracks = df_summary[df_summary['length'] >= num].index.tolist()

    # Filter the original DataFrame to only include valid tracks
    df_filtered = df[df['all_ids'].isin(valid_tracks)].copy()

    return df_filtered

def add_single_tracks(merged_tracks_df, metric=None, tm_red=None, tm_green=None): 
    """
    Add tracks from individual channels (red or green) to the merged tracks DataFrame.
    """
    # Safely retrieve merged red and green spots, or use empty DataFrames
    if metric is not None:
        try:
            green_spots_in_merged_tracks, red_spots_in_merged_tracks = metric.get_merged_red_green_spot_ids()
        except AttributeError as e:
            print("AttributeError encountered in get_merged_red_green_spot_ids:", e)
            green_spots_in_merged_tracks, red_spots_in_merged_tracks = pd.DataFrame(), pd.DataFrame()
    else:
        green_spots_in_merged_tracks, red_spots_in_merged_tracks = pd.DataFrame(), pd.DataFrame()

    # Dynamically extract columns from tm_red and tm_green
    red_columns = tm_red.spots.columns if tm_red is not None else []
    green_columns = tm_green.spots.columns if tm_green is not None else []

    # Process distinctly red spots
    if tm_red is not None:
        distinctly_red_spots = (
            metric.tm_red.spots.drop(red_spots_in_merged_tracks).reset_index()
            if metric is not None else tm_red.spots
        )
        # Ensure the DataFrame contains the 'ID' column
        if 'ID' not in distinctly_red_spots.columns:
            distinctly_red_spots['ID'] = pd.Series(dtype='int')  # Add missing column
    else:
        distinctly_red_spots = pd.DataFrame(columns=red_columns if red_columns else ['ID'])

    # Debugging step to print the structure of distinctly_red_spots
    print(f"distinctly_red_spots columns: {distinctly_red_spots.columns}")

    # Process distinctly green spots
    if tm_green is not None:
        distinctly_green_spots = (
            metric.tm_green.spots.drop(green_spots_in_merged_tracks).reset_index()
            if metric is not None else tm_green.spots
        )
        # Ensure the DataFrame contains the 'ID' column
        if 'ID' not in distinctly_green_spots.columns:
            distinctly_green_spots['ID'] = pd.Series(dtype='int')  # Add missing column
    else:
        distinctly_green_spots = pd.DataFrame(columns=green_columns if green_columns else ['ID'])

    # Handle static tracks for unmerged cases
    if tm_green is not None:
        unique_greens = distinctly_green_spots.ID.unique()
        green_static = tm_green.tracks.query("SPOT_SOURCE_ID in @unique_greens or SPOT_TARGET_ID in @unique_greens")
    else:
        green_static = pd.DataFrame()

    if tm_red is not None:
        unique_reds = distinctly_red_spots.ID.unique()
        red_static = tm_red.tracks.query("SPOT_SOURCE_ID in @unique_reds or SPOT_TARGET_ID in @unique_reds")
    else:
        red_static = pd.DataFrame()

    # Filter unaccounted tracks
    if not merged_tracks_df.empty:
        merged_track_ids = merged_tracks_df.merged_track_id.unique()
        reds_in_merged = {int(item.split('_')[0][1:]) for item in merged_track_ids if item.startswith('r')}
        greens_in_merged = {int(item.split('_')[1][1:]) for item in merged_track_ids if item.split('_')[1].startswith('g')}
        filtered_red_static = red_static[~red_static['TrackID'].isin(reds_in_merged)]
        filtered_green_static = green_static[~green_static['TrackID'].isin(greens_in_merged)]
    else:
        filtered_red_static = red_static
        filtered_green_static = green_static

    # Process individual tracks for red and green
    if len(filtered_red_static) > 0: 
        reds = pd.concat([
            tm_red.trace_track(i).assign(color="red")  # Add color column
            for i in filtered_red_static.TrackID.unique()
        ], ignore_index=True) if tm_red is not None else pd.DataFrame()
        reds['track_id'] = 'r' + reds['track_id']
    else: 
        reds = pd.DataFrame(columns=red_columns)

    if len(filtered_green_static) > 0:
        greens = pd.concat([
            tm_green.trace_track(i).assign(color="green")  # Add color column
            for i in filtered_green_static.TrackID.unique()
        ], ignore_index=True) if tm_green is not None else pd.DataFrame()
        greens['track_id'] = 'g' + greens['track_id']
    else:
        greens = pd.DataFrame(columns=green_columns)

    # Combine individual red and green tracks
    combined_df = pd.concat([reds, greens], ignore_index=True)

    # Combine merged and individual tracks (if any) into one DataFrame
    if merged_tracks_df.empty:
        merged_and_single = combined_df
        if not combined_df.empty:
            merged_and_single['all_ids'] = merged_and_single['track_id']
        else: 
            merged_and_single['all_ids'] = pd.Series(dtype='str')

    else:
        merged_and_single = pd.concat([merged_tracks_df, combined_df], ignore_index=True)
        merged_and_single['all_ids'] = merged_and_single['merged_track_id'].fillna(merged_and_single['track_id'])
    
    return merged_and_single

def makePies(df, base_path, lim, conv, label='Total'):
    """
    Generates pie charts showing cell cycle ratios at three time points: 
    t = 0, midpoint, and t = last frame, handling missing frames as zero counts.

    Parameters:
    - df: DataFrame containing the tracking data.
    - base_path: Path to save the output plots.
    - label: Label for the plot titles (default is 'Total').

    Returns:
    - pie_chart_df: DataFrame containing normalized cell cycle ratios.
    """
    # Define the frame range explicitly
    first_frame = 0
    last_frame = lim # gets you the index of the last one
    midpoint_frame = (first_frame + last_frame) // 2

    # Helper function to get counts by color, with default zeros for missing frames
    def get_color_counts(frame):
        if frame not in df['frame'].values:
            return pd.Series([0, 0, 0], index=['red', 'green', 'yellow'])  # All zeros for missing frames
        frame_data = df[df['frame'] == frame]
        return frame_data['color'].value_counts().reindex(['red', 'green', 'yellow'], fill_value=0)

    # Get counts at the three time points
    frame_0_counts = get_color_counts(first_frame)
    frame_halfway_counts = get_color_counts(midpoint_frame)
    frame_end_counts = get_color_counts(last_frame)

    # Normalize counts to proportions
    def normalize_counts(counts):
        total = counts.sum()
        return counts / total if total > 0 else counts  # Return zeros for empty frames

    normalized_0 = normalize_counts(frame_0_counts)
    normalized_halfway = normalize_counts(frame_halfway_counts)
    normalized_end = normalize_counts(frame_end_counts)

    # Create a DataFrame for the pie chart data
    pie_chart_df = pd.DataFrame({
        't = 0': normalized_0,
        f't = {midpoint_frame}': normalized_halfway,
        f't = {last_frame}': normalized_end
    }).fillna(0)

    # Log intermediate data
    print("Normalized counts:")
    print(pie_chart_df)

    # Plot the pie charts
    labels = ['G0/G1', 'G1/S', 'S/G2/M']
    colors = ['red', 'green', 'yellow']

    plt.figure(figsize=(15, 5))
    for i, (counts, time_point) in enumerate(zip(
        [normalized_0, normalized_halfway, normalized_end],
        [first_frame, midpoint_frame, last_frame]
    )):
        if counts.sum() == 0:  # Handle case where all proportions are zero
            print(f"Skipping pie chart for t = {time_point} due to zero proportions.")
            continue

        plt.subplot(1, 3, i + 1)
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.title(f'{label} Cell Cycle Ratios at t = {time_point * conv:.1f} hrs')

    # Save the pie chart plot
    output_folder = Path(base_path) / 'graphs'
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "ProportionsPieCharts.png"
    plt.savefig(output_file)
    plt.close()

    print(f"Saved pie charts to {output_file}")
    return pie_chart_df


def plotTotalCells(df, base_path):
    """
    Generates a line plot showing the total number of cells over time.

    Parameters:
    - df: DataFrame containing the tracking data.
    - base_path: Path to save the output plot.
    - label: Label for the plot title (default is 'Total').

    Returns:
    - total_cell_counts_df: DataFrame containing total cell counts per frame.
    """
    # Calculate total cell counts over time
    cell_counts_over_time = df.groupby('frame')['all_ids'].nunique()

    # Convert to DataFrame
    total_cell_counts_df = cell_counts_over_time.reset_index()
    total_cell_counts_df.columns = ['frame', 'total_cells']

    # Plot the total cell counts
    plt.figure(figsize=(10, 6))
    plt.plot(total_cell_counts_df['frame'], total_cell_counts_df['total_cells'], 
             marker='o', color='blue')

    plt.xlabel('Frame')
    plt.ylabel('Number of Cells')
    plt.title(f'{label} Cell Counts Over Time')

    # Save the plot
    output_folder = Path(base_path) / 'graphs'
    output_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_folder / "TotalCellCounts.png")
    plt.show()
    plt.close()

    return total_cell_counts_df


def boxplot(df, base_path, conv):
    """
    Generates a boxplot of cell cycle stage duration and returns a DataFrame with color durations.

    Parameters:
    - df: The DataFrame containing the track data.
    - base_path: The base path where the plot will be saved.

    Returns:
    - A DataFrame with the color durations for each track.
    """

    # Calculate the duration for each color phase per track
    color_durations = (
        df.groupby(['all_ids', 'color'])['frame']
        .apply(lambda x: x.max() - x.min())
        .reset_index(name='duration')
    )

    # Apply the conversion factor to convert frames to hours
    color_durations['duration'] = color_durations['duration'] * conv

    # Define the order of colors for the plot
    color_order = ['red', 'yellow', 'green']
    color_durations['color'] = pd.Categorical(
        color_durations['color'], categories=color_order, ordered=True
    )

    # Create the boxplot
    plt.figure(figsize=(10, 8))
    color_durations.boxplot(by='color', column=['duration'])

    # Set labels and titles
    new_labels = ['G0/G1', 'G1/S', 'S/G2/M']
    plt.xticks([1, 2, 3], new_labels)
    plt.title(f'Cell Cycle Stage Duration for {df["all_ids"].nunique()} Trajectories')
    plt.suptitle('')  # Suppress the default title
    plt.xlabel('Cell Cycle')
    plt.ylabel('Duration (hours)')

    # Save the plot
    output_folder = base_path / 'graphs'
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    output_path = output_folder / "BoxPlots.png"
    plt.savefig(output_path)
    plt.show()
    plt.close()

    # Return the DataFrame with color durations
    return color_durations

def plotEMTCharts(df, base_path, lim, conv):
    # Filter to include only tracks with 'merged_track_id' present in the first frame (t = 0)
    first_frame = df['frame'].min()
    ids_in_first_frame = df[df['frame'] == first_frame]['all_ids'].unique()
    df_counted = df[df['all_ids'].isin(ids_in_first_frame)]

    # Additional filter: keep only 'Attached' tracks present for at least 90 frames
    attached_tracks = (
        df_counted[df_counted['EMT'] == 'Attached']
        .groupby('all_ids')
        .filter(lambda x: len(x) >= 0.9 * lim)
    )['all_ids'].unique()

    detached_tracks = df_counted[df_counted['EMT'] == 'Detached']['all_ids'].unique()

    last_frame = lim
    midpoint_frame = (first_frame + last_frame) // 2

    def get_emt_counts(frame):
        """
        Returns counts of 'Attached' and 'Detached' tracks for a specific frame.
        Ensures 0 for frames without data.
        """
        frame_data = df[df['frame'] == frame]
        attached_count = frame_data[frame_data['all_ids'].isin(attached_tracks) & (frame_data['EMT'] == 'Attached')].shape[0]
        detached_count = frame_data[frame_data['all_ids'].isin(detached_tracks) & (frame_data['EMT'] == 'Detached')].shape[0]
        return [attached_count, detached_count]

    emt_0_counts = get_emt_counts(first_frame)
    emt_halfway_counts = get_emt_counts(midpoint_frame)
    emt_end_counts = get_emt_counts(last_frame)

    # Create a DataFrame for the pie chart data
    emt_chart_df = pd.DataFrame({
        't = 0': emt_0_counts,
        f't = {midpoint_frame}': emt_halfway_counts,
        f't = {last_frame}': emt_end_counts
    }, index=['Attached', 'Detached']).fillna(0)

    # Normalize counts to proportions for pie charts
    normalized_df = emt_chart_df.apply(lambda x: x / x.sum() if x.sum() > 0 else [0, 0], axis=0)

    # Handle any NaN values (unlikely here) to ensure valid plotting
    normalized_df = normalized_df.fillna(0)

    # Save the normalized data as a CSV
    data_path = Path(base_path / 'data')
    data_path.mkdir(parents=True, exist_ok=True)

    # Plot pie charts for t = 0, midpoint, and t = last
    labels = ['Attached', 'Detached']
    colors = ['cyan', 'brown']
    conversion = conv  # Convert frames to hours

    plt.figure(figsize=(15, 5))

    for i, (timepoint, column) in enumerate(zip(
        ['t = 0', f't = {midpoint_frame}', f't = {last_frame}'], 
        normalized_df.columns
    )):
        plt.subplot(1, 3, i + 1)
        if normalized_df[column].sum() > 0:  # Check if the column has valid data
            plt.pie(normalized_df[column], labels=labels, autopct='%1.1f%%', colors=colors)
        else:
            plt.text(0.5, 0.5, "No Data", ha='center', va='center')
        plt.title(f'Proportions at {timepoint}')

    # Save the pie charts
    output_path = Path(base_path) / "graphs"
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "EMTPieCharts.png")
    plt.show()

    # Return the normalized DataFrame for reuse
    return normalized_df

  

def roseplot(df, base_path):
    """
    Generates a rose plot of cell migration trajectories and 
    returns a DataFrame with shifted X and Y positions for all tracks.

    Parameters:
    - df: DataFrame containing the track data.
    - base_path: Base path to save the plots.

    Returns:
    - A DataFrame with shifted X and Y positions for each track.
    """
    # Conversion factor from pixels to micrometers
    um_per_pixel = 1.3556

    # Initialize a list to store shifted trajectory data for each track
    shifted_data = []

    # Create a figure for the rose plot
    plt.figure(figsize=(10, 10))
    
    # only keeps around cells that were there for 90 frames, or 15 hours

    # Loop through each track ID in 'all_ids' and plot its trajectory
    for track_id in df['all_ids'].unique():
        df_filtered = df[df['all_ids'] == track_id].copy()

        # Shift trajectory to start at (0, 0)
        df_filtered['shifted_POSITION_X'] = df_filtered['POSITION_X'] - df_filtered['POSITION_X'].iloc[0]
        df_filtered['shifted_POSITION_Y'] = df_filtered['POSITION_Y'] - df_filtered['POSITION_Y'].iloc[0]

        # Convert positions to micrometers
        df_filtered['shifted_POSITION_X'] *= um_per_pixel
        df_filtered['shifted_POSITION_Y'] *= um_per_pixel

        # Append the shifted data for aggregation later
        shifted_data.append(df_filtered[['all_ids', 'frame', 'shifted_POSITION_X', 'shifted_POSITION_Y']])

        # Plot the trajectory
        plt.plot(df_filtered['shifted_POSITION_X'], df_filtered['shifted_POSITION_Y'], label=f'Track {track_id}')

    # Add labels, title, and grid
    plt.title('Cell Migration Rose Plot')
    plt.xlabel('X-Distance (µm)')
    plt.ylabel('Y-Distance (µm)')
    plt.grid(True)

    # Save the rose plot
    output_folder = base_path / 'graphs'
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    plt.savefig(output_folder / 'Roseplot.png')
    plt.show()
    plt.close()

    if shifted_data:
        roseplot_df = pd.concat(shifted_data, ignore_index=True)
    else:
        print("No data to concatenate for rose plot.")
        roseplot_df = pd.DataFrame()  # Return an empty DataFrame if there’s no data

    # Return the DataFrame with shifted positions
    return roseplot_df




def plotBarcode(df, base_path, conv):

    # Set up the figure
    plt.figure(figsize=(12, 8))

    # Define the color mapping for the stages
    color_map = {'red': 'r', 'yellow': 'y', 'green': 'g'}

    # Calculate the time per frame (16 hours over 97 frames)
    time_per_frame = conv

    # Get unique track IDs from the DataFrame
    track_ids_to_plot = df['all_ids'].unique()

    # Loop through each track ID and plot their color trajectory
    for i, track_id in enumerate(track_ids_to_plot):
        # Filter the DataFrame for the current track_id
        df_filtered = df[df['all_ids'] == track_id]

        # Get the frames and corresponding colors
        frames = df_filtered['frame'].values
        colors = df_filtered['color'].values

        # Convert frames to time in hours
        times = frames * time_per_frame

        # Plot a horizontal line for each track, colored by the cell cycle stage
        for j in range(len(times) - 1):
            plt.plot([times[j], times[j + 1]], [i, i], color=color_map[colors[j]], lw=6)

    # Add labels and format the plot
    plt.yticks(range(len(track_ids_to_plot)), track_ids_to_plot)
    plt.xlabel('Time (hours)')
    plt.ylabel('Merged Track ID')
    plt.title(f'Cell Cycle Stages Over Time for {len(track_ids_to_plot)} Trajectories')

    # Create a legend for the colors
    red_patch = mpatches.Patch(color='r', label='G0/G1')
    yellow_patch = mpatches.Patch(color='y', label='G1/S')
    green_patch = mpatches.Patch(color='g', label='S/G2/M')
    plt.legend(handles=[red_patch, yellow_patch, green_patch], title="Cell Cycle Stages")

    # Save the plot
    output_folder = Path(base_path) / 'graphs'
    output_folder.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    output_path = output_folder / "Colors_Over_Time.png"
    plt.savefig(output_path)
    plt.show()
    plt.close()


def plotIntensities(df, base_path, conv):
    # Paths to image stacks
    red_path = base_path / 'red.tif'
    green_path = base_path / 'green.tif'

    # Load the red and green image stacks
    red_stack = cf.utils.read_stack(red_path)
    green_stack = cf.utils.read_stack(green_path)

    max_frame_index = red_stack.shape[0] - 1  # Last valid frame index

    # Filter out rows with invalid frame indices
    df = df[df['frame'] <= max_frame_index]

    # List to store intensity data
    intensity_data = []

    # Time per frame in hours
    time_per_frame = conv

    # Loop through each unique track ID in the DataFrame
    for track_id in df['all_ids'].unique():
        df_track = df[df['all_ids'] == track_id]

        # Initialize lists for red and green intensities
        red_intensities, green_intensities = [], []

        # Iterate over frames for the current track
        for _, row in df_track.iterrows():
            frame, x, y = row['frame'], int(row['POSITION_X']), int(row['POSITION_Y'])

            # Extract intensity values from a 10x10 neighborhood around the cell
            red_neighborhood = red_stack[frame, max(0, y-10):min(red_stack.shape[1], y+10), max(0, x-10):min(red_stack.shape[2], x+10)]
            green_neighborhood = green_stack[frame, max(0, y-10):min(green_stack.shape[1], y+10), max(0, x-10):min(green_stack.shape[2], x+10)]

            # Compute mean intensities for red and green channels
            red_intensity = np.mean(red_neighborhood)
            green_intensity = np.mean(green_neighborhood)

            # Store the intensities
            red_intensities.append(red_intensity)
            green_intensities.append(green_intensity)

        # Normalize intensities to their respective max values
        max_red = max(red_intensities) if red_intensities else 0
        max_green = max(green_intensities) if green_intensities else 0

        normalized_red = [(i / max_red) if max_red > 0 else 0 for i in red_intensities]
        normalized_green = [(i / max_green) if max_green > 0 else 0 for i in green_intensities]

        # Ensure alignment between frames and intensities
        for idx, (_, row) in enumerate(df_track.iterrows()):
            if idx < len(normalized_red) and idx < len(normalized_green):
                intensity_data.append({
                    'all_ids': track_id,
                    'frame': row['frame'],
                    'time': row['frame'] * time_per_frame,
                    'red_mean': normalized_red[idx],
                    'green_mean': normalized_green[idx]
                })

    # Convert intensity data to DataFrame
    intensity_df = pd.DataFrame(intensity_data)

    # Create output directory for plots
    output_folder = base_path / 'intensity_plots'
    output_folder.mkdir(parents=True, exist_ok=True)

    # Plot individual intensity plots for each track
    for track_id in intensity_df['all_ids'].unique():
        df_track = intensity_df[intensity_df['all_ids'] == track_id]

        plt.figure(figsize=(10, 6))
        plt.plot(df_track['time'], df_track['red_mean'], color='r', lw=2, label='Red Channel')
        plt.plot(df_track['time'], df_track['green_mean'], color='g', lw=2, label='Green Channel')

        plt.title(f'Normalized Pixel Intensity for Track {track_id}')
        plt.xlabel('Time (hours)')
        plt.ylabel('Normalized Intensity (Max RFU = 1.0)')
        plt.legend(loc='upper right')

        # Save the individual plot
        plt.savefig(output_folder / f"Intensity_Plot_{track_id}.png")
        plt.show()
        plt.close()

    return intensity_df


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

import os
from pathlib import Path
from PIL import Image

def check_tiff_stacks(root_path):
    """
    Checks all red.tif and green.tif stacks under Day0, Day1, Day2 directories
    to ensure they have exactly 97 images.
    
    Parameters:
    - root_path: str or Path. Path to the P4 directory.

    """
    root_path = Path(root_path)
    stack_counts = []

    # Ensure we are starting from the root directory
    if not root_path.exists():
        print(f"Root path {root_path} does not exist!")
        return

    # Iterate through Day0, Day1, Day2 directories
    for day in ['Day0', 'Day1', 'Day2']:
        day_path = root_path / day
        if not day_path.exists():
            print(f"Skipping missing day directory: {day_path}")
            continue

        # Iterate through all well directories
        for well_dir in day_path.iterdir():
            if well_dir.is_dir() and well_dir.name == 'B02':
                # Check red.tif and green.tif in the well directory
                for tiff_file in ['red.tif', 'green.tif']:
                    tiff_path = well_dir / tiff_file
                    if not tiff_path.exists():
                        print(f"Missing file: {tiff_path}")
                        continue

                    try:
                        with Image.open(tiff_path) as img:
                            stack_size = 0
                            while True:
                                try:
                                    img.seek(stack_size)
                                    stack_size += 1
                                except EOFError:
                                    break
                     
                            print(f"{tiff_path} contains {stack_size}")
                            stack_counts.append(stack_size)
                    except Exception as e:
                        print(f"Error reading {tiff_path}: {e}")
    return min(stack_counts), max(stack_counts)


# idea, generate just the csv files on the HPC. do the graph stuff on my local computer?
# or do all graphs and graph csv there too, just do agglomeration on local machine?

def safe_get_merged_tracks(metric, max_metric_value=2.0):
    try:
        # Call the original get_merged_tracks method
        merged_tracks_df = metric.get_merged_tracks(max_metric_value)

        # If merged_tracks_df is empty, handle it by returning an empty DataFrame
        if merged_tracks_df.empty:
            print("No tracks found after merging.")
            return pd.DataFrame()
            
    except AttributeError as e:
        print("AttributeError encountered. Likely no tracks to merge:", e)
        return pd.DataFrame()

    return merged_tracks_df



def processWell(base_path, conv):
    red_path = base_path / 'red.tif'
    green_path = base_path / 'green.tif'

    threshold = 0.0001
    print(threshold)

    # Segment red and green channels
    segment_stack(path=red_path, threshold=threshold)
    segment_stack(path=green_path, threshold=threshold)       

    # Trackmate processing
    # USERINPUT TO TRACKMATE BASIC_SETTING.XML FILE
    trackmate_xml_path = Path('/cluster/home/jstras02/alpha/basic_settings.xml')
    red_segmented_path = base_path / 'red_segmented.tiff'
    green_segmented_path = base_path / 'green_segmented.tiff'

    # Process red channel
    try:
        _run_local_trackmate(trackmate_xml_path, red_segmented_path)
    except Exception as e:
        print(f"TrackMate failed for red channel in {base_path}: {e}")
        red_segmented_path = None  # Mark red as failed

    # Process green channel
    try:
        _run_local_trackmate(trackmate_xml_path, green_segmented_path)
    except Exception as e:
        print(f"TrackMate failed for green channel in {base_path}: {e}")
        green_segmented_path = None  # Mark green as failed

    # Load TrackMate results
    tm_red, tm_green = None, None

    if red_segmented_path and (base_path / 'red_segmented.tiff.xml').exists():
        try:
            tm_red = cf.math.TrackmateXML(base_path / 'red_segmented.tiff.xml')
            if tm_red.tracks.empty:
                tm_red = None  # No tracks found
        except Exception as e:
            print(f"Error loading red channel TrackMate XML: {e}")
            tm_red = None

    if green_segmented_path and (base_path / 'green_segmented.tiff.xml').exists():
        try:
            tm_green = cf.math.TrackmateXML(base_path / 'green_segmented.tiff.xml')
            if tm_green.tracks.empty:
                tm_green = None  # No tracks found
        except Exception as e:
            print(f"Error loading green channel TrackMate XML: {e}")
            tm_green = None

    # Ensure at least one channel has data
    if tm_red is None and tm_green is None:
        error_file = base_path / 'error.txt'
        with open(error_file, 'w') as f:
            f.write("Both red and green channels have no tracks. Skipping well.\n")
        print(f"No tracks found in both channels for {base_path}. Error file written.")
        return  # Skip this well

    # Handle merged and single tracks
    if tm_red is not None and not tm_red.tracks.empty and tm_green is not None and not tm_green.tracks.empty:
        # Both channels have data
        metric = cf.math.CartesianSimilarity(tm_red, tm_green)
        merged_tracks_df = safe_get_merged_tracks(metric, max_metric_value=15.0)
        merged_and_single = add_single_tracks(merged_tracks_df, metric, tm_red, tm_green)
    else:
        # Only one channel present
        empty_cols = [
            'ID', 'frame', 'POSITION_X', 'POSITION_Y', 'PERIMETER', 'image_id',
            'AREA', 'ROI', 'roi_polygon', 'ELLIPSE_MAJOR', 'track_id', 'color',
            'source_track', 'merged_track_id'
        ]
        merged_and_single = pd.DataFrame(columns=empty_cols)
        if tm_red is not None and not tm_red.tracks.empty:
            # Provide tm_red directly and handle metric as None
            single_red = add_single_tracks(pd.DataFrame(columns=empty_cols), tm_red=tm_red, tm_green=None, metric=None)
            merged_and_single = pd.concat([merged_and_single, single_red], ignore_index=True)
        if tm_green is not None and not tm_green.tracks.empty:
            # Provide tm_green directly and handle metric as None
            single_green = add_single_tracks(pd.DataFrame(columns=empty_cols), tm_red=None, tm_green=tm_green, metric=None)
            merged_and_single = pd.concat([merged_and_single, single_green], ignore_index=True)
    # Debugging merged_and_single before calling addEMTColumn
    print(f"merged_and_single columns: {merged_and_single.columns}")
    print(f"merged_and_single head:\n{merged_and_single.head()}")

    # Ensure all_ids column exists
    if 'all_ids' not in merged_and_single.columns:
        print("Error: 'all_ids' column is missing from merged_and_single.")
        merged_and_single['all_ids'] = pd.Series(dtype='str')  # Add empty all_ids column


    # Process EMT column and save data
    # if cells move more than 100 um / hour they are considered detached. verified empirically (roughly)
    alltracks_df, velocity_df = addEMTColumn(merged_and_single, base_path, 100, conv)
    
    data_path = base_path / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
    
    tracks_path = data_path / 'all_tracks_df.csv'
    alltracks_df.to_csv(tracks_path, index=False)
    
    velocity_path = data_path / 'velocity_data.csv'
    velocity_df.to_csv(velocity_path, index=False)
    
    print(f"Saved merged tracks to {tracks_path} and velocities to {velocity_path}")
    
    makeVideo(alltracks_df, base_path, red_path, green_path, red_segmented_path, green_segmented_path)
    print(f"Saved movie")
    
    return alltracks_df

    

from pathlib import Path
import pandas as pd

def save_data(base_path, **dataframes):


    # Create the 'data' directory if it doesn't exist
    data_path = Path(base_path) / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    # Save each DataFrame to a CSV
    for name, df in dataframes.items():
        file_path = data_path / f"{name}_data.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved {name} to {file_path}")



def plotter(alltracks_df, base_path, lim, conv):

    if not alltracks_df.empty: 
        attached_df, detached_df = filterEMT(alltracks_df)

        attachedWholeTime_df = filter_duration(attached_df, lim)

        pie_df = makePies(alltracks_df, base_path, lim, conv)
        total_cells_df = plotTotalCells(alltracks_df, base_path)
        EMT_df = plotEMTCharts(alltracks_df, base_path, lim, conv)
        plotBarcode(alltracks_df, base_path, conv)
        save_data(base_path, pie=pie_df, total_cells=total_cells_df, EMT=EMT_df)
        if not attachedWholeTime_df.empty:
            boxplot_df = boxplot(attachedWholeTime_df, base_path, conv)
            roseplot_df = roseplot(attachedWholeTime_df, base_path)
            intensity_df = plotIntensities(attachedWholeTime_df, base_path, conv)
            save_data(base_path, boxplot=boxplot_df, roseplot=roseplot_df, intensity=intensity_df)
    else: 
        return
    


import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def aggregate_data(root_path: str):
    """
    Aggregates data from replicate groups, processes CSV files, and saves results.
    
    Parameters:
    - root_path: str. The root path where the data is located.
    """
    # Ensure root_path is a Path object
    root_path = Path(root_path)
    
    # Define aggregated paths
    aggregated_path = root_path / 'aggregated'
    aggregated_path.mkdir(parents=True, exist_ok=True)
    aggregated_graph_path = aggregated_path / 'graphs'
    aggregated_graph_path.mkdir(parents=True, exist_ok=True)

    # Define replicate groups
    replicate_groups = {
        'Aa': ['B02', 'B03', 'C02', 'C03'],
        'Ab': ['B04', 'B05', 'C04', 'C05'],
        'Ac': ['B06', 'B07', 'C06', 'C07'],
        'Ad': ['B08', 'B09', 'C08', 'C09'],
        'ACtrl': ['B10', 'B11', 'C10', 'C11'],
        'Ba': ['D02', 'D03', 'E02', 'E03'],
        'Bb': ['D04', 'D05', 'E04', 'E05'],
        'Bc': ['D06', 'D07', 'E06', 'E07'],
        'Bd': ['D08', 'D09', 'E08', 'E09'],
        'BCtrl': ['D10', 'D11', 'E10', 'E11'],
        'Ca': ['F02', 'F03', 'G02', 'G03'],
        'Cb': ['F04', 'F05', 'G04', 'G05'],
        'Cc': ['F06', 'F07', 'G06', 'G07'],
        'Cd': ['F08', 'F09', 'G08', 'G09'],
        'CCtrl': ['F10', 'F11', 'G10', 'G11']
    }

    files_to_aggregate = ['pie_data.csv', 'EMT_data.csv', 'boxplot_data.csv', 'roseplot_data.csv', 'velocity_data.csv']


    def load_csv(wells, file_name):
        dfs = []
        for well in wells:
            file_path = root_path / well / 'data' / file_name
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # explicitly skip if DataFrame is empty or contains no columns
                    if df.empty or df.shape[1] == 0:
                        print(f"Warning: {file_path} is empty or has no columns. Skipping this file.")
                        continue
                    dfs.append(df)
                except pd.errors.EmptyDataError:
                    # Handle empty file explicitly
                    print(f"Warning: {file_path} is empty (no data). Skipping this file.")
                except Exception as ex:
                    # Handle other unexpected errors explicitly
                    print(f"Warning: Error reading {file_path}: {ex}. Skipping this file.")
        return dfs

    def average_dfs(df_list):
        return sum(df_list) / len(df_list) if df_list else pd.DataFrame()

    def extract_boxplot_stats(wells):
        dfs = load_csv(wells, 'boxplot_data.csv')
        combined_df = pd.concat(dfs, ignore_index=True)
        stats = combined_df.groupby('color')['duration'].describe()[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

        boxplot_data = [
            {
                'label': color,
                'mean': row['mean'],
                'med': row['50%'],
                'q1': row['25%'],
                'q3': row['75%'],
                'whislo': row['min'],
                'whishi': row['max'],
                'fliers': []
            }
            for color, row in stats.iterrows()
        ]
        return pd.DataFrame(boxplot_data)

    # Main aggregation logic
    for group, wells in replicate_groups.items():
        for file_name in files_to_aggregate:
            try:
                dfs = load_csv(wells, file_name)

                # Filter dfs based on the validity check
                valid_dfs = [df for df in dfs if is_valid_df(df, file_name)]

                output_file = aggregated_path / f'{group}_{file_name}'
                output_graph_file = aggregated_graph_path / f'{group}_{file_name}'

                if file_name in ['pie_data.csv', 'EMT_data.csv']:
                    avg_df = average_dfs(valid_dfs)
                    if len(avg_df) > 0:
                        avg_df.to_csv(output_file, index=False)
                        print(f"Saved {output_file}.")
                    else:
                        print(f"Error processing {file_name} for group {group}: Empty or invalid data after filtering")
                elif file_name == 'boxplot_data.csv':
                    avg_boxplot_stats = extract_boxplot_stats(wells)
                    avg_boxplot_stats.to_csv(output_file, index=False)
                    print(f"Saved {output_file}.")
                """
                there is a potential weak spot here:

                """
                elif file_name in ['roseplot_data.csv', 'velocity_data.csv']:
                    # Filter out empty or NaN-filled DataFrames explicitly before concatenating
                    valid_dfs = [df for df in dfs if not df.empty and not df.dropna(how='all').empty]

                    if valid_dfs:
                        concatenated_df = pd.concat(valid_dfs, ignore_index=True)
                        concatenated_df.to_csv(output_file, index=False)
                        print(f"Saved {output_file}.")
                    else:
                        # Handle explicitly the case of no valid dataframes
                        print(f"No valid data found for {file_name}. Skipping file.")

            except Exception as e:
                print(f"Exception processing {file_name} for group {group}: {e}")


def single_well(base_path, lim, conv): 
    base_path = Path(base_path)
    well = base_path.name
    print(f'Starting on well: {well}')
    
    # Call processWell and check its return value
    alltracks_df = processWell(base_path, conv)
    if alltracks_df is None:
        print(f'No data returned for well: {well}. Skipping plotting.')
        return  # Skip plotting and end this function gracefully
    
    # Proceed with plotting if processWell succeeded
    plotter(alltracks_df, base_path, lim, conv)
    print('Done')

def process_all_wells(root_path, lim, conv):
    """
    Loops through each well in the root path and calls the main() function with the well's path.
    Skips processing if the 'velocity.csv' file already exists in the well's 'data' folder.
    """
    # Iterate through each subdirectory (well) in the root directory
    for well_dir in root_path.iterdir():
        if well_dir.is_dir() and well_dir.name != "aggregated":  # Ensure it's a directory (skip files)
            check_file = well_dir / 'data' / 'dummy.csv'  # Path to the target file
            # Check if the file exists
            if check_file.exists():
                continue  # Skip this well if the file exists
            else: 
                print(f"Processing well {well_dir}")
                single_well(well_dir, lim, conv)  # Call your processing function

def main(root_path):
    # Loop through each subdirectory in the root_path (e.g., Day0, Day1, Day2)
    duration = 16 
    minimum, maximum = check_tiff_stacks(root_path)
    # frame limit
    lim = (minimum - 1)
    #conversion factor for minutes between frames
    conv = duration / maximum
    print(f"Common frame denominator: {lim}")
    for subroot in root_path.iterdir():
        if subroot.is_dir() and subroot.name.startswith("Day"):
            print(f"Processing subdirectory: {subroot}")
            process_all_wells(subroot, lim, conv)  # Process all wells in the current day subdirectory
            aggregate_data(subroot)

if __name__ == "__main__":
    # Check if root_path was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Error: Please provide the root path (e.g., P4) as a command-line argument.")
        sys.exit(1)
    
    # Get root_path from the command-line argument and convert it to a Path object
    root_path = Path(sys.argv[1])
    main(root_path)
