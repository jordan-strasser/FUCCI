# USER INPUT TO ENVIRONMENT CONTAINING CONFLUENTFUCCI'S PYTHON
#!/usr/bin/env /cluster/home/jstras02/cf/python
import sys
# USER INPUT PATH TO CORRECT PYTHON ENVIRONMENT
sys.path.append('/cluster/home/jstras02/.conda/envs/rhel8/2024.06-py311/extra/lib/python3.10/site-packages')
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_pie_chart(base_path, drugs):
    """
    Load, process, and generate pie charts from PIE data across multiple days.
    
    Parameters:
    - base_path (str or Path): The base path where the data is located.

    Saves:
    - A pie chart graph saved to 'collective_graphs/pie_chart.png' within base_path.
    """
    # Convert base_path to Path object if it's a string
    base_path = Path(base_path)

    # Days and filenames to process
    days = ['Day0', 'Day1', 'Day2']
    concs = ['a', 'b', 'c', 'd', 'Ctrl']
    filename = '_pie_data.csv'

    # Prepare to store DataFrames for each day

    for drug in drugs: 
        names = [drug + conc for conc in concs]
        filenames = [name + filename for name in names]
        dataframes_per_day = []
    # Process data for each day
        for day in days:
            # Define the path for the current day's data
            day_path = base_path / day / 'aggregated'

            # Load and concatenate all files for the current day
            dataframes = [
                pd.read_csv(day_path / filename) 
                for filename in filenames 
                if (day_path / filename).exists()
            ]
                
            combined_df = pd.concat(dataframes, axis=0).reset_index(drop=True)

            # Store the combined DataFrame
            dataframes_per_day.append(combined_df)

        # Concatenate all days into a final DataFrame
        final_df = pd.concat(dataframes_per_day, axis=1).reset_index(drop=True)

        # Assign new column names representing time points
        new_columns = [f't = {i}' for i in range(0, 65, 8)]
        final_df.columns = new_columns

        # Prepare data for plotting
        y_labels = concs  # Group labels
        time_points = new_columns  # Use new columns as time points
        colors = ['red', 'green', 'yellow']  # Colors for pie slices
        labels = ['G0/G1', 'S/G2/M', 'G1/S']  # Corresponding biological stages

        # Create a 5x9 grid of subplots (5 groups, 9 time points)
        fig, axes = plt.subplots(5, 9, figsize=(20, 10))

        # Generate pie charts for each group and time point
        for i, ax_row in enumerate(axes):  # Loop over the 5 groups (rows)
            for j, ax in enumerate(ax_row):  # Loop over the 9 time points (columns)
                # Extract values for red, green, and yellow for the current subplot
                red = final_df.iloc[i * 3, j]
                green = final_df.iloc[i * 3 + 1, j]
                yellow = final_df.iloc[i * 3 + 2, j]

                # Create a pie chart in the current subplot
                ax.pie([red, green, yellow], colors=colors, startangle=90)
                ax.axis('equal')  # Ensure the pie chart is a circle

        # Adjust layout for better spacing
        plt.tight_layout()

        # Add a y-axis label for concentration
        fig.text(0.0, 0.5, 'Concentration', va='center', rotation='vertical', fontsize=14)

        # Add row labels (A, B, C, D, Ctrl) to the left of each row
        for i, label in enumerate(y_labels):
            fig.text(0.03, 0.9 - i * 0.18, label, va='center', ha='right', fontsize=12)

        # Add a common x-axis label for time
        fig.text(0.5, 0.00, 'Time (hours)', ha='center', va='center', fontsize=14)

        # Add individual time point labels at the bottom of each column
        for j, time in enumerate(time_points):
            fig.text(0.10 + j * 0.1, 0.02, time, ha='center', va='center', fontsize=12)

        # Add a legend for the pie chart labels
        fig.legend(labels=labels, loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=12)

        # Adjust layout to ensure proper spacing
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.5)

        # Define the path to save the graph
        output_dir = base_path / 'output' / 'collective_graphs'
        output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if not exists
        output_file = output_dir / f'{drug}_pie_chart.png'

        # Save the figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph saved at: {output_file}")

        # Display the plot
        plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_EMT_chart(base_path, drugs):
    """
    Load, process, and generate pie charts from EMT data across multiple days.
    
    Parameters:
    - base_path (str or Path): The base path where the data is located.

    Saves:
    - A pie chart graph saved to 'collective_graphs/EMT_chart.png' within base_path.
    """
    # Convert base_path to Path object if it's a string
    base_path = Path(base_path)

    # Days and filenames to process
    days = ['Day0', 'Day1', 'Day2']
    concs = ['a', 'b', 'c', 'd', 'Ctrl']
    filename = '_EMT_data.csv'

    for drug in drugs: 
        # Define group names and filenames
        names = [drug + conc for conc in concs]
        filenames = [name + filename for name in names]
        dataframes_per_day = []

        # Process data for each day
        for day in days:
            day_path = base_path / day / 'aggregated'
            # Load and concatenate all files for the current day
            dataframes = [
                pd.read_csv(day_path / filename) 
                for filename in filenames 
                if (day_path / filename).exists()
            ]
            combined_df = pd.concat(dataframes, axis=0).reset_index(drop=True)
            dataframes_per_day.append(combined_df)

        # Concatenate all days into a final DataFrame
        EMT_df = pd.concat(dataframes_per_day, axis=1).reset_index(drop=True)

        # Select only the desired columns (1st, 4th, 7th)
        selected_columns = [0, 3, 6]  # Indices for 1st, 4th, and 7th columns
        EMT_df = EMT_df.iloc[:, selected_columns]

        # Assign new column names representing time points
        new_columns = [f't = {i}' for i in range(0, 65, 24)]  # [0, 24, 48]
        EMT_df.columns = new_columns

        # Prepare data for plotting
        y_labels = ['A', 'B', 'C', 'D', 'Ctrl']  # Group labels
        time_points = new_columns  # Use new columns as time points
        colors = ['brown', 'cyan']  # Colors for attached and detached cells
        labels = ['attached', 'detached']  # Labels for the legend

        # Create a 5x3 grid of subplots (5 groups, 3 time points)
        fig, axes = plt.subplots(5, 3, figsize=(15, 10))

        # Generate pie charts for each group and time point
        for i, ax_row in enumerate(axes):  # Loop over the 5 groups (rows)
            for j, ax in enumerate(ax_row):  # Loop over the 3 selected columns (columns)
                # Extract values for attached and detached cells for the current subplot
                attached = EMT_df.iloc[i * 2, j]
                detached = EMT_df.iloc[i * 2 + 1, j]

                # Create a pie chart in the current subplot
                ax.pie([attached, detached], colors=colors, startangle=90)
                ax.axis('equal')  # Ensure the pie chart is a circle

        # Adjust layout for better spacing
        plt.tight_layout()

        # Add a y-axis label for concentration
        fig.text(0.0, 0.5, 'Concentration', va='center', rotation='vertical', fontsize=14)

        # Add row labels (A, B, C, D, Ctrl) to the left of each row
        for i, label in enumerate(y_labels):
            fig.text(0.03, 0.9 - i * 0.18, label, va='center', ha='right', fontsize=12)

        # Add a common x-axis label for time
        fig.text(0.5, 0.00, 'Time (hours)', ha='center', va='center', fontsize=14)

        # Add individual time point labels at the bottom of each column
        n_columns = 3  # Number of columns in the grid
        x_positions = [0.19, 0.5, 0.81]  # Adjusted x positions for 3 columns

        for j, (x, time) in enumerate(zip(x_positions, time_points)):
            fig.text(x, 0.02, time, ha='center', va='center', fontsize=12)

        # Add a legend for attached and detached labels
        fig.legend(labels=labels, loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=12)

        # Adjust layout to ensure proper spacing
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08, wspace=0.05, hspace=0.4)

        # Define the path to save the graph
        output_dir = base_path / 'output'/ 'collective_graphs'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{drug}_EMT_chart.png'

        # Save the figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph saved at: {output_file}")

        # Display the plot
        plt.show()





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




import os
def ratiometricjourney(root_path, drugs):

    # Define replicate groups
    replicate_groups = {
        'A_C1': ['B02', 'B03', 'C02', 'C03'],
        'A_C2': ['B04', 'B05', 'C04', 'C05'],
        'A_C3': ['B06', 'B07', 'C06', 'C07'],
        'A_C4': ['B08', 'B09', 'C08', 'C09'],
        'A_Ctrl': ['B10', 'B11', 'C10', 'C11'],
        'B_C1': ['D02', 'D03', 'E02', 'E03'],
        'B_C2': ['D04', 'D05', 'E04', 'E05'],
        'B_C3': ['D06', 'D07', 'E06', 'E07'],
        'B_C4': ['D08', 'D09', 'E08', 'E09'],
        'B_Ctrl': ['D10', 'D11', 'E10', 'E11'],
        'C_C1': ['F02', 'F03', 'G02', 'G03'],
        'C_C2': ['F04', 'F05', 'G04', 'G05'],
        'C_C3': ['F06', 'F07', 'G06', 'G07'],
        'C_C4': ['F08', 'F09', 'G08', 'G09'],
        'C_Ctrl': ['F10', 'F11', 'G10', 'G11']
    }

    # Define base path and days
    output_dir = os.path.join(root_path, 'output')
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    days = ['Day0', 'Day1', 'Day2']
    hours_per_frame = 16 / 97  # Each frame represents approximately 0.165 hours

    # Function to calculate ratiometric counts by frame for a single well
    def get_ratiometric_counts_by_frame(df, day_offset_hours):
        unique_frames = df['frame'].unique()
        unique_frames.sort()

        def get_color_counts(frame):
            frame_data = df[df['frame'] == frame]
            return frame_data['color'].value_counts().reindex(['red', 'green', 'yellow'], fill_value=0)

        def normalize_counts(counts):
            total = counts.sum()
            return counts / total if total > 0 else counts

        results = []
        for frame in unique_frames:
            color_counts = get_color_counts(frame)
            normalized_counts = normalize_counts(color_counts)

            results.append({
                'time_hours': day_offset_hours + frame * hours_per_frame,  # Adjust time with day offset
                'red_ratio': normalized_counts['red'],
                'green_ratio': normalized_counts['green']
            })

        return pd.DataFrame(results)

    # Iterate over each drug
    for drug in drugs:
        all_concentration_data = []  # Collect data for all concentrations of a drug

        # Collect data for each concentration group across all days
        for concentration in ['C1', 'C2', 'C3', 'C4', 'Ctrl']:
            group_key = f"{drug}_{concentration}"
            wells = replicate_groups[group_key]

            all_ratios = []
            for day_index, day in enumerate(days):
                day_path = os.path.join(root_path, day)
                day_offset_hours = day_index * 16  # Offset for each day in hours

                for well in wells:
                    path = f'{day_path}/{well}/data/all_tracks_df.csv'
                    totalcells_path = f'{day_path}/{well}/data/total_cells_data.csv'

                    # Check if the file exists, skip if not
                    if not os.path.exists(path) or not os.path.exists(totalcells_path):
                        print(f"File not found for well {well} on {day}, skipping...")
                        continue

                    df = pd.read_csv(path)
                    ratiometric_df = get_ratiometric_counts_by_frame(df, day_offset_hours)
                    ratiometric_df['concentration'] = concentration  # Add concentration information
                    all_ratios.append(ratiometric_df)

            # Aggregate data for the concentration if there is data
            if all_ratios:
                merged_ratios = pd.concat(all_ratios).groupby('time_hours').agg({'red_ratio': ['mean', 'std'], 'green_ratio': ['mean', 'std']}).reset_index()
                merged_ratios.columns = ['time_hours', 'red_mean', 'red_std', 'green_mean', 'green_std']
                merged_ratios['concentration'] = concentration
                all_concentration_data.append(merged_ratios)

        # Combine all concentration data into a single DataFrame for the drug, if any data exists
        if all_concentration_data:
            drug_data = pd.concat(all_concentration_data, ignore_index=True)
            csv_path = os.path.join(output_dir, f'{drug}_data.csv')
            drug_data.to_csv(csv_path, index=False)

            # Plot Red and Green Proportions with Shaded Standard Deviation for each concentration in the drug
            plt.figure(figsize=(12, 10))

            # Red proportion plot for the drug over 48 hours
            plt.subplot(2, 1, 1)
            for concentration in ['C1', 'C2', 'C3', 'C4', 'Ctrl']:
                data = drug_data[drug_data['concentration'] == concentration]
                plt.plot(data['time_hours'], data['red_mean'], label=f'{concentration} Red Proportion')
                plt.fill_between(data['time_hours'], 
                                 data['red_mean'] - data['red_std'], 
                                 data['red_mean'] + data['red_std'], 
                                 alpha=0.3)
            plt.xlabel('Time (Hours)')
            plt.ylabel('Red Proportion')
            plt.title(f'Red Proportion Over 48 Hours for Drug {drug}')
            plt.legend()

            # Green proportion plot for the drug over 48 hours
            plt.subplot(2, 1, 2)
            for concentration in ['C1', 'C2', 'C3', 'C4', 'Ctrl']:
                data = drug_data[drug_data['concentration'] == concentration]
                plt.plot(data['time_hours'], data['green_mean'], label=f'{concentration} Green Proportion')
                plt.fill_between(data['time_hours'], 
                                 data['green_mean'] - data['green_std'], 
                                 data['green_mean'] + data['green_std'], 
                                 alpha=0.3)
            plt.xlabel('Time (Hours)')
            plt.ylabel('Green Proportion')
            plt.title(f'Green Proportion Over 48 Hours for Drug {drug}')
            plt.legend()

            plt.tight_layout()

            # Save the plot as PNG
            png_path = os.path.join(output_dir, f'{drug}_proportions.png')
            plt.savefig(png_path)
            plt.show()
        else:
            print(f"No data available for drug {drug}, skipping plot and CSV save.")

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

def roseplot(base_path):
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

            um_per_pixel = 1.3556

            # Initialize a list to store shifted trajectory data for each track
            shifted_data = []

            # Create a figure for the rose plot
            plt.figure(figsize=(10, 10))

            # only keeps around cells that were there for 90 frames, or 15 hours

            # Loop through each track ID in 'all_ids' and plot its trajectory
            for track_id in df['all_ids'].unique():
                df_filtered = df[df['all_ids'] == track_id].copy()
                plt.plot(df_filtered['shifted_POSITION_X'], df_filtered['shifted_POSITION_Y'], label=f'Track {track_id}')

            # Add labels, title, and grid
            plt.title('Cell Migration Rose Plot')
            plt.xlabel('X-Distance (µm)')
            plt.ylabel('Y-Distance (µm)')
            plt.grid(True)

            # Save the rose plot
            output_folder = base_path / 'output' / day
            output_folder.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            name_parts = roseplot_path.name.split('_')
            graph_filename = "_".join(name_parts[:2]) + '.png'
            plt.savefig(output_folder / graph_filename)
            plt.show()
            plt.close()
        else: 
            print(f'{t} does not exist')

def main():
    # USER INPUT
    base_path = Path('PATH_TO_DIR_CONTAINING_WELLS')
    drugs = getdrugs(base_path)
    print(drugs)
    generate_pie_chart(base_path, drugs)
    generate_EMT_chart(base_path, drugs)
    process_velocity_data(base_path, drugs, column_name='max')
    process_velocity_data(base_path, drugs, column_name='average')
    ratiometricjourney(base_path, drugs)
    roseplot(base_path)
    
if __name__ == "__main__":
    main()  

