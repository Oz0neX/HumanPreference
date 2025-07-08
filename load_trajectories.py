import numpy as np
import os
import glob
from imitation.data import serialize

def load_trajectories_from_directory(directory_path):
    """Load trajectories from a directory using imitation library's serialize.load()"""
    trajectories = []
    
    if not os.path.exists(directory_path):
        return trajectories
    
    # Find all trajectory files in the directory
    trajectory_files = glob.glob(os.path.join(directory_path, "trajectory_*"))
    
    for file_path in trajectory_files:
        try:
            # Load trajectory using imitation library
            loaded_trajectories = serialize.load(file_path)
            trajectories.extend(loaded_trajectories)
            print(f"Loaded trajectory from {file_path} with {len(loaded_trajectories)} trajectories")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return trajectories

def extract_data():
    """Extract trajectory data from the trajectory_data directory structure"""
    maps_data = {}
    
    if not os.path.exists("trajectory_data"):
        print("trajectory_data directory not found!")
        return maps_data
    
    # Load from map_* folders
    map_paths = glob.glob(os.path.join("trajectory_data", "map_*"))
    for map_path in map_paths:
        map_name = os.path.basename(map_path)
        maps_data[map_name] = {'optimal': [], 'teaching': []}
        
        # Load optimal trajectories
        optimal_dir = os.path.join(map_path, "optimal")
        optimal_trajectories = load_trajectories_from_directory(optimal_dir)
        maps_data[map_name]['optimal'] = optimal_trajectories
        
        # Load teaching trajectories
        teaching_dir = os.path.join(map_path, "teaching")
        teaching_trajectories = load_trajectories_from_directory(teaching_dir)
        maps_data[map_name]['teaching'] = teaching_trajectories
        
        print(f"Map {map_name}: {len(optimal_trajectories)} optimal, {len(teaching_trajectories)} teaching trajectories")
    
    return maps_data

def main():
    """Main function to extract and display trajectory data"""
    print("Extracting trajectory data...")
    maps_data = extract_data()
    
    if not maps_data:
        print("No trajectory data found!")
        return None
    
    print(f"Found data for {len(maps_data)} maps")
    
    # Display summary
    for map_name, trajectories in maps_data.items():
        print(f"\n=== {map_name} ===")
        print(f"Optimal trajectories: {len(trajectories['optimal'])}")
        print(f"Teaching trajectories: {len(trajectories['teaching'])}")
    
    print("\nTrajectory extraction completed!")
    return maps_data

if __name__ == "__main__":
    main()
