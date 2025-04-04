#!/usr/bin/env /cluster/home/jstras02/cf/python
import sys
sys.path.append('/cluster/home/jstras02/.conda/envs/rhel8/2024.06-py311/extra/lib/python3.10/site-packages')

# Primarily for re-running the aggregation step to drop problematic wells. 
# User will need the .txt file of wells to drop in the is_valid() function 

from pathlib import Path
import json
import pandas as pd

def makePies(df, base_path, lim, conv, label='Total'):
    """
    Generates pie charts showing cell cycle ratios at three time points: 
    t = 0, midpoint, and t = last frame, handling missing frames as zero counts.

    Parameters:
    - df: DataFrame containing the tracking data.
    - base_path: Path to save the anoutput plots.
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


def save_data(base_path, **dataframes):


    # Create the 'data' directory if it doesn't exist
    data_path = Path(base_path) / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    # Save each DataFrame to a CSV
    for name, df in dataframes.items():
        file_path = data_path / f"{name}_data.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved {name} to {file_path}")

def aggregate_data(root_path: str):
    """
    Aggregates data from replicate groups, processes CSV files, and saves results.
    
    Parameters:
    - root_path: str. The root path where the data is located.
    """
    # Ensure root_path is a Path object
    import json
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

    def is_valid(wells, root_path):
        from pathlib import Path
        import json

        plate = root_path.parts[-2]
        day = root_path.parts[-1]
        # user input, change to your local file
        dropped_path = Path('/cluster/home/jstras02/levinlab_link/fucci/code/dropped_wells.txt')
        dropped_wells = []

        if dropped_path.exists():
            with open(dropped_path, 'r') as f:
                dropped_wells_df = json.load(f)
            dropped_wells = dropped_wells_df.get(plate, {}).get(day, [])
            print(f"Filtering out dropped wells for {plate} {day}: {dropped_wells}")
        else:
            print(f"No dropped wells file found at {dropped_path} — loading all wells.")

        valid_wells = [well for well in wells if well not in dropped_wells]
        return valid_wells


    def load_csv(wells, file_name, root_path):
        valid_wells = is_valid(wells, root_path)

        dfs = []
        for well in valid_wells:
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
        dfs = load_csv(wells, 'boxplot_data.csv', root_path)
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
                dfs = load_csv(wells, file_name, root_path)
                output_file = aggregated_path / f'{group}_{file_name}'
                output_graph_file = aggregated_graph_path / f'{group}_{file_name}'

                if file_name in ['pie_data.csv', 'EMT_data.csv']:
                    avg_df = average_dfs(dfs)
                    if len(avg_df) > 0:
                        avg_df.to_csv(output_file, index=False)
                        print(f"Saved {output_file}.")
                    else:
                        print(f"Error processing {file_name} for group {group}: Empty file")

                elif file_name == 'boxplot_data.csv':
                    avg_boxplot_stats = extract_boxplot_stats(wells)
                    avg_boxplot_stats.to_csv(output_file, index=False)
                    print(f"Saved {output_file}.")

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
                print(f"Error processing {file_name} for group {group}: {e}")



def run_aggregate_on_selected_plates(root_dir, plates_to_include):
    root_dir = Path(root_dir)

    for plate_name in plates_to_include:
        plate_dir = root_dir / plate_name
        if not plate_dir.exists():
            print(f"Plate {plate_name} not found at {plate_dir}, skipping.")
            continue

        for day_dir in sorted(plate_dir.glob("Day*")):
            if not day_dir.is_dir():
                continue

            print(f"Running aggregation on: {day_dir}")
            try:
                aggregate_data(day_dir)
            except Exception as e:
                print(f"Failed on {day_dir}: {e}")

plates = ['P1', 'P2', 'P3', 'P4', 'P5']
run_aggregate_on_selected_plates('/cluster/home/jstras02/levinlab_link/data/FUCCI', plates)



#re-running individual functions on all
# for well_dir in root_path.iterdir():
#     if well_dir.is_dir():  # Ensure it's a directory (skip files)
#         data_path = well_dir / 'data' / 'all_tracks_df.csv'  # Path to the target file
#         # Debugging: Print the full path before checking if the file exists
#         print(f"Checking path: {data_path}")
#         total_cells_path = well_dir / 'data' / 'total_cells_data.csv'
#         # Check if the file exists
#         if data_path.exists() and total_cells_path.exists():
#             print(f"Found file: {data_path}")
#             alltracks_df = pd.read_csv(data_path)
#             EMT_df = plotEMTCharts(alltracks_df, well_dir, lim, conv)
#             pie_df = makePies(alltracks_df, well_dir, lim, conv)
#             save_data(well_dir, EMT=EMT_df, pie=pie_df)
        
#         else:
#             print(f"skipping: {well_dir}")
#             continue



