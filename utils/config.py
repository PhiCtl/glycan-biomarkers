import random

import numpy as np
import json
import os
import torch

import seaborn as sns
import matplotlib.pyplot as plt

def set_seed(config):
    """Set seed for reproducibility."""
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

def load_config(config_path='../config.json'):
    """
    Loads a configuration file in json format.
    Args:
        config_path (str): The path to the configuration file. Defaults to '../config.json'.
    Returns:
        dict or None: A dictionary containing the configuration data if the file exists,
                      otherwise None.
    Note:
        This function assumes the configuration file is in json format and requires
        the `json` module to parse it. Ensure the `json` module is installed and
        imported before using this function.
    """

    config = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
    return config

def apply_plotting_settings(config):
    """Apply seaborn and matplotlib plotting settings."""
    try:
        plotting = config.get("plotting", {})
        
        sns.set_style(plotting.get("seaborn_style", "darkgrid"))
        sns.set_context(plotting.get("context", "notebook"))
        sns.set_palette(plotting.get("palette", "deep"))
        sns.set_theme(font_scale=plotting.get("font_scale", 2))
        
        plt.rcParams["figure.figsize"] = plotting.get("figsize", [10, 6])

        print("Plotting settings applied!")
    except Exception as e:
        print(f"Error applying plotting settings: {e}")


