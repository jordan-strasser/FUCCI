#!/usr/bin/env /cluster/home/jstras02/cf/python
import sys
sys.path.append('/cluster/home/jstras02/.conda/envs/rhel8/2024.06-py311/extra/lib/python3.10/site-packages')
import pandas as pd
import os
import glob
# Define base path
root_path = "/cluster/home/jstras02/levinlab_link/data/FUCCI/"
plates = ['NS', 'P1', 'P2', 'P3', 'P4', 'P5']
for p in plates: 
    base_path = os.path.join(root_path, p)
    # List of days to process
    days = ["Day0", "Day1", "Day2"]
    # Process each day
    for day in days:
        day_path = os.path.join(base_path, day, "aggregated")
        if not os.path.exists(day_path):
            print(f"Skipping: {day_path} (Directory not found)")
            continue
        # Find all CSV files ending with "_velocity_data.csv"
        velocity_files = glob.glob(os.path.join(day_path, "*_velocity_data.csv"))
        for file_path in velocity_files:
            print(f"Processing: {file_path}")
            # Load the CSV file
            df = pd.read_csv(file_path)
            # Filter out rows where max_velocity >= 100
            df_filtered = df[df["max_velocity"] < 100]
            # Save over the original file
            df_filtered.to_csv(file_path, index=False)
            print(f"Updated: {file_path}")
    print(f"Processing complete for {p}.")
print("Processing complete for all plates.")