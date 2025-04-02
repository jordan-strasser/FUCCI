#!/usr/bin/env /cluster/home/jstras02/cf/python
import sys
sys.path.append('/cluster/home/jstras02/.conda/envs/rhel8/2024.06-py311/extra/lib/python3.10/site-packages')
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch 
import numpy as np

def generate_pie_chart(base_path, drugs):
    base_path = Path(base_path)
    days = ['Day0', 'Day1', 'Day2']
    concs = ['a', 'b', 'c', 'd', 'Ctrl']
    filename = '_pie_data.csv'

    for drug in drugs:
        dataframes_per_day = []

        for day in days:
            day_path = base_path / day / 'aggregated'
            daily_dfs = []

            for conc in concs:
                fname = f"{drug}{conc}{filename}"
                file_path = day_path / fname

                if file_path.exists():
                    df = pd.read_csv(file_path)
                    # Ensure shape (3 rows x 3 cols)
                    if df.shape != (3, 3):
                        raise ValueError(f"File {fname} has unexpected shape: {df.shape}")
                else:
                    # Explicitly match the file structure: 3 rows × 3 columns placeholder
                    df = pd.DataFrame(np.nan, columns=['t = 0', 't = 48', 't = 96'], index=range(3))

                daily_dfs.append(df)

            # Concatenate vertically (stack concentrations): 5 concentrations × 3 rows each = 15 rows, 3 cols
            combined_day_df = pd.concat(daily_dfs, axis=0).reset_index(drop=True)
            dataframes_per_day.append(combined_day_df)

        # Concatenate horizontally (days): 15 rows, 3 days × 3 cols = 9 cols total
        final_df = pd.concat(dataframes_per_day, axis=1, ignore_index=True)

        # Explicitly assign correct timepoint labels across 3 days
        final_df.columns = [
            '0', '8', '16',
            '24', '32', '40',
            '48', '56', '64'
        ]

        # Prepare plotting variables
        y_labels = concs
        colors = ['red', 'green', 'yellow']
        labels = ['G0/G1', 'S/G2/M', 'G1/S']

        fig, axes = plt.subplots(5, 9, figsize=(20, 10))

        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                red = final_df.iloc[i * 3, j]
                green = final_df.iloc[i * 3 + 1, j]
                yellow = final_df.iloc[i * 3 + 2, j]

                # Grey placeholder if any value missing or invalid
                if pd.isna(red) or pd.isna(green) or pd.isna(yellow) or (red + green + yellow == 0):
                    ax.pie([1], colors=['lightgrey'], labels=['No Data'], textprops={'fontsize': 10})
                else:
                    ax.pie([red, green, yellow], colors=colors, startangle=90)

                ax.axis('equal')

        plt.tight_layout()

        fig.text(0.0, 0.5, 'Concentration', va='center', rotation='vertical', fontsize=14)
        for i, label in enumerate(y_labels):
            fig.text(0.03, 0.9 - i * 0.18, label, va='center', ha='right', fontsize=12)

        fig.text(0.5, 0.00, 'Time (hours)', ha='center', fontsize=14)
        for j, col_label in enumerate(final_df.columns):
            fig.text(0.1 + j * 0.1, 0.02, col_label, ha='center', fontsize=12)
        # Create custom legend elements
        legend_elements = [
            Patch(facecolor='red', label='G0/G1'),
            Patch(facecolor='green', label='S/G2/M'),
            Patch(facecolor='yellow', label='G1/S'),
            Patch(facecolor='lightgrey', label='No Data')
        ]

        # Replace your old legend line with this explicit legend:
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=12)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.5)

        output_dir = base_path / 'output' / 'collective_graphs'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{drug}_pie_chart.png'

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph saved at: {output_file}")
        plt.show()

def generate_EMT_chart(base_path, drugs):
    base_path = Path(base_path)
    days = ['Day0', 'Day1', 'Day2']
    concs = ['a', 'b', 'c', 'd', 'Ctrl']
    filename = '_EMT_data.csv'

    for drug in drugs:
        dataframes_per_day = []

        for day in days:
            day_path = base_path / day / 'aggregated'
            daily_dfs = []

            for conc in concs:
                fname = f"{drug}{conc}{filename}"
                file_path = day_path / fname

                if file_path.exists():
                    df = pd.read_csv(file_path)
                    if df.shape != (2, 3):
                        raise ValueError(f"File {fname} has unexpected shape: {df.shape}")
                else:
                    # Correct 2 rows × 3 columns placeholder
                    df = pd.DataFrame(np.nan, columns=['t = 0', 't = 48', 't = 96'], index=range(2))

                daily_dfs.append(df)

            # Stack concentrations vertically: 5 concentrations × 2 rows = 10 rows, 3 columns per day
            combined_day_df = pd.concat(daily_dfs, axis=0).reset_index(drop=True)
            dataframes_per_day.append(combined_day_df)

        # Concatenate horizontally (3 days): final_df is 10 rows × 9 columns
        final_df = pd.concat(dataframes_per_day, axis=1, ignore_index=True)
        selected_columns = [0, 3, 6]  # Indices for 1st, 4th, and 7th columns
        final_df = final_df.iloc[:, selected_columns]

        # Explicitly name columns
        final_df.columns = ['0','24','48']

        # Plotting variables
        y_labels = ['a', 'b', 'c', 'd', 'Ctrl']
        colors = ['brown', 'cyan']
        labels = ['attached', 'detached']

        fig, axes = plt.subplots(5, 3, figsize=(20, 10))

        for i, ax_row in enumerate(axes):
            for j, ax in enumerate(ax_row):
                attached = final_df.iloc[i * 2, j]
                detached = final_df.iloc[i * 2 + 1, j]

                # Check for NaNs or zero total explicitly
                if pd.isna(attached) or pd.isna(detached) or (attached + detached == 0):
                    ax.pie([1], colors=['lightgrey'], labels=['No Data'], textprops={'fontsize': 10})
                else:
                    ax.pie([attached, detached], colors=colors, startangle=90)

                ax.axis('equal')

        plt.tight_layout()

        fig.text(0.0, 0.5, 'Concentration', va='center', rotation='vertical', fontsize=14)
        for i, label in enumerate(y_labels):
            fig.text(0.03, 0.9 - i * 0.18, label, va='center', ha='right', fontsize=12)

        fig.text(0.5, 0.00, 'Time (hours)', ha='center', fontsize=14)
        for j, col_label in enumerate(final_df.columns):
            fig.text(0.10 + j * 0.1, 0.02, col_label, ha='center', fontsize=12)

        # Explicit custom legend with correct colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='brown', label='attached'),
            Patch(facecolor='cyan', label='detached'),
            Patch(facecolor='lightgrey', label='No Data')
        ]

        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize=12)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.5)

        output_dir = base_path / 'output' / 'collective_graphs'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{drug}_EMT_chart.png'

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph saved at: {output_file}")
        plt.show()


def process_velocity_data(base_path, drugs, column_name):
    base_path = Path(base_path)

    days = ['Day0', 'Day1', 'Day2']
    concs = ['a', 'b', 'c', 'd', 'Ctrl']
    filename = '_velocity_data.csv'

    results_path = base_path / 'output' / 'velocity_tables'
    results_path.mkdir(parents=True, exist_ok=True)

    for drug in drugs:
        names = [drug + conc for conc in concs]

        for day in days:
            day_path = base_path / day / 'aggregated'

            columns_data = []
            for conc in concs:
                fname = f"{drug}{conc}{filename}"
                file_path = day_path / fname

                if file_path.exists():
                    df_temp = pd.read_csv(file_path)
                    # Check if the column exists
                    if f'{column_name}_velocity' in df_temp.columns:
                        velocity_series = df_temp[f'{column_name}_velocity']
                    else:
                        # Column missing, fill with 0
                        velocity_series = pd.Series([0])
                else:
                    # File missing, explicitly insert a single '0' as placeholder
                    velocity_series = pd.Series([0])

                columns_data.append(velocity_series.reset_index(drop=True))

            # Make sure all series have same length (fill shorter ones with zeros)
            max_length = max(len(s) for s in columns_data)
            columns_data_aligned = [
                s.reindex(range(max_length), fill_value=0)
                for s in columns_data
            ]

            # Concatenate horizontally
            velocity_df = pd.concat(columns_data_aligned, axis=1)
            velocity_df.columns = names

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
    base_path = Path(base_path)
    days = ['Day0', 'Day1', 'Day2']
    drugs = ['A', 'B', 'C']
    concs = ['a', 'b', 'c', 'd', 'Ctrl']
    combos = [d + c for c in concs for d in drugs]
    filepaths = [c + '_roseplot_data.csv' for c in combos]

    for day in days:
        for file in filepaths:
            roseplot_path = base_path / day / 'aggregated' / file

            if not roseplot_path.exists():
                # Explicitly skip missing files
                print(f"File {roseplot_path} does not exist. Skipping...")
                continue

            # Load and plot existing files
            df = pd.read_csv(roseplot_path)

            fig, ax = plt.subplots(figsize=(10, 10))

            for track_id in df['all_ids'].unique():
                df_filtered = df[df['all_ids'] == track_id]
                ax.plot(df_filtered['shifted_POSITION_X'], df_filtered['shifted_POSITION_Y'], label=f'Track {track_id}')

            # Explicitly set consistent axis limits
            ax.set_xlim(-300, 300)
            ax.set_ylim(-300, 300)
            tick_interval = 100
            ax.set_xticks(np.arange(-300, 301, tick_interval))
            ax.set_yticks(np.arange(-300, 301, tick_interval))

            font_size = 36
            ax.set_title('Cell Migration Rose Plot', fontsize=font_size, pad=30)
            ax.set_xlabel('X-Distance (µm)', fontsize=font_size, labelpad=20)
            ax.set_ylabel('Y-Distance (µm)', fontsize=font_size, labelpad=20)

            ax.tick_params(axis='both', labelsize=font_size, pad=10)
            ax.grid(True)

            plt.tight_layout()

            output_folder = base_path / 'output' / 'roseplots' / day
            output_folder.mkdir(parents=True, exist_ok=True)
            graph_filename = file.replace('_roseplot_data.csv', '_roseplot.png')
            fig.savefig(output_folder / graph_filename, bbox_inches='tight', pad_inches=0.5)
            print(f"Rose plot saved at: {output_folder / graph_filename}")

            plt.close(fig)


def main():
    if len(sys.argv) != 2:
        print("Usage: python graphs.py <plate_path>")
        sys.exit(1)
    base_path = Path(sys.argv[1])
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

