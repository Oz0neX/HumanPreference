#!/usr/bin/env python
"""
Map Configuration for IRL Experiments
Provides predefined map configurations that work with MetaDrive's built-in map system.
"""

# Available map configurations for experiments
MAP_CONFIGS = {
    "simple": {
        "map_string": "SSS",
        "description": "Simple straight road - good for basic testing",
        "difficulty": "easy"
    },
    "curves": {
        "map_string": "SCSCS", 
        "description": "Straight-Curve-Straight-Curve-Straight - moderate difficulty",
        "difficulty": "medium"
    },
    "complex": {
        "map_string": "SCXRXCS",
        "description": "Complex route with intersections and ramps",
        "difficulty": "hard"
    },
    "highway": {
        "map_string": "SSSSSS",
        "description": "Long straight highway - good for speed control testing",
        "difficulty": "easy"
    },
    "city": {
        "map_string": "CXCXCX",
        "description": "City-like environment with many turns and intersections",
        "difficulty": "hard"
    }
}

DEFAULT_MAP = "curves"
DEFAULT_SEED = 12345


def get_map_config(map_name: str = None):
    """
    Get map configuration by name.
    
    Args:
        map_name: Name of the map configuration. If None, returns default.
        
    Returns:
        dict: Map configuration with 'map_string' and metadata
    """
    if map_name is None:
        map_name = DEFAULT_MAP
        
    if map_name not in MAP_CONFIGS:
        print(f"Warning: Map '{map_name}' not found. Using default '{DEFAULT_MAP}'")
        map_name = DEFAULT_MAP
        
    config = MAP_CONFIGS[map_name].copy()
    config["name"] = map_name
    config["seed"] = DEFAULT_SEED
    
    return config


def list_available_maps():
    """Print all available map configurations."""
    print("Available Map Configurations:")
    print("=" * 50)
    for name, config in MAP_CONFIGS.items():
        print(f"{name:10} | {config['map_string']:10} | {config['difficulty']:6} | {config['description']}")
    print("=" * 50)


def create_environment_config(map_name: str = None, manual_control: bool = True, **kwargs):
    """
    Create a complete environment configuration for MetaDrive.
    
    Args:
        map_name: Name of the map configuration
        manual_control: Whether to enable manual control
        **kwargs: Additional configuration options
        
    Returns:
        dict: Complete environment configuration
    """
    map_config = get_map_config(map_name)
    
    # Default environment configuration
    env_config = {
        "map": map_config["map_string"],
        "traffic_density": 0.1,
        "num_scenarios": 1,
        "start_seed": map_config["seed"],
        "manual_control": manual_control,
        "use_render": True,
        "vehicle_config": {
            "show_lidar": True,
            "show_navi_mark": True,
            "show_line_to_navi_mark": True
        }
    }
    
    # Update with any additional kwargs
    env_config.update(kwargs)
    
    return env_config, map_config


if __name__ == "__main__":
    # Demo the map configuration system
    print("Map Configuration System Demo")
    print("=" * 40)
    
    # List all available maps
    list_available_maps()
    
    # Test getting different map configs
    print("\nTesting map configurations:")
    for map_name in ["simple", "curves", "complex", "nonexistent"]:
        config = get_map_config(map_name)
        print(f"{map_name:12} -> {config['map_string']} ({config['description']})")
    
    # Test creating environment config
    print("\nExample environment configuration:")
    env_config, map_info = create_environment_config("curves", manual_control=False)
    print(f"Map: {map_info['name']} ({map_info['map_string']})")
    print(f"Manual control: {env_config['manual_control']}")
    print(f"Traffic density: {env_config['traffic_density']}")
