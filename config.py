import yaml

# Define default configuration values
DEFAULT_CONFIG = {
    "experiment_name": "default_experiment",
    "logging": {
        "log_dir": None,
        "tensorboard": True,
    },
    "training": {
        "chkpt_interval": 10,
        "num_episodes": 200,
        "num_steps": 1000,
        "save_model_interval": 50,
        "target_update_frequency": 1000,
        "batch_size": 64,
        "epsilon_decay": 0.995,
        "epsilon_end": 0.01,
        "epsilon_start": 1.0,
    },
    "environment": {
        "camera": 0,
        "render": True,
        "parameters": [
            "CAP_PROP_BRIGHTNESS",
            "CAP_PROP_CONTRAST",
            "CAP_PROP_SATURATION",
            "CAP_PROP_EXPOSURE",
        ],
    },
    "agent": {
        "gamma": 0.99,
        "learning_rate": 0.001,
        "type": "QAgent",
        "checkpoint": None,
        "memory_size": 10000,
    },
}

def merge_with_defaults(config, defaults):
    """
    Recursively merges a user config with default values.
    If a key is missing in `config`, it is taken from `defaults`.
    """
    merged = defaults.copy()
    for key, value in defaults.items():
        if key in config:
            if isinstance(value, dict) and isinstance(config[key], dict):
                # Recursively merge nested dictionaries
                merged[key] = merge_with_defaults(config[key], value)
            else:
                # Take the user's value
                merged[key] = config[key]
    return merged

# Load the YAML config file and merge it with defaults
def load_config(config_file, default_config = None):
    if default_config is None:
        default_config = DEFAULT_CONFIG

    try:
        with open(config_file, "r") as file:
            user_config = yaml.safe_load(file)
        return merge_with_defaults(user_config, default_config)
    except FileNotFoundError:
        print(f"Config file '{config_file}' not found. Using default configuration.")
        return default_config
    except yaml.YAMLError as e:
        print(f"Error parsing the config file: {e}")
        return default_config

def save_config(config, config_file):
    with open(config_file, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
