import os
import json

def drop_problematic_wells(output_path='dropped_wells.txt'):
    dropped_wells_df = {}
    history_stack = []  # Keep track of (plate, day, well) for undo

    print("Welcome to the Dropped Wells Interface.\n")
    print("→ Type 'done' to finish a section.")
    print("→ Type 'undo' to remove the last entered well.\n")

    while True:
        plate = input("Plate? (or type 'done' to finish): ").strip()
        if plate.lower() == "done":
            break
        
        if plate not in dropped_wells_df:
            dropped_wells_df[plate] = {}

        while True:
            day = input(f"Day for {plate}? (e.g., Day0, or 'done' to finish this plate): ").strip()
            if day.lower() == "done":
                break
            
            if day not in dropped_wells_df[plate]:
                dropped_wells_df[plate][day] = []

            print(f"Enter wells for {plate} {day} (type 'done' when finished with this day; 'undo' to remove last well):")
            while True:
                well = input("  Well: ").strip()
                
                if well.lower() == "done":
                    break
                elif well.lower() == "undo":
                    if history_stack:
                        last_plate, last_day, last_well = history_stack.pop()
                        if last_well in dropped_wells_df[last_plate][last_day]:
                            dropped_wells_df[last_plate][last_day].remove(last_well)
                            print(f"  ↩️ Removed last well: {last_well} from {last_plate} {last_day}")
                        else:
                            print("  ⚠️ Last well not found. Nothing to undo.")
                    else:
                        print("  ⚠️ Nothing to undo.")
                elif well:
                    well = well.upper()
                    dropped_wells_df[plate][day].append(well)
                    history_stack.append((plate, day, well))

    # Save to .txt file as JSON
    with open(output_path, 'w') as f:
        f.write(json.dumps(dropped_wells_df, indent=4))
    
    print(f"\n✅ Dropped wells saved to {output_path}")
    return dropped_wells_df


drop_problematic_wells()

