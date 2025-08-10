import yaml
import importlib

####### CHANGE THIS PART TO RUN DIFFERENT SCRIPTS/DATASETS #######
SCRIPT_TO_RUN = "CNN"
DATASET = "real_timefreq"
###################################################################


if __name__ == "__main__":
    # Error handling for script and dataset selection
    try:
        script_module = importlib.import_module(f"scripts.{SCRIPT_TO_RUN}")
    except ModuleNotFoundError:
        print(f"Error: Script '{SCRIPT_TO_RUN}' not found in scripts/")
        exit(1)

    if DATASET not in config["datasets"]:
        print(f"Error: Dataset '{DATASET}' not found in config.yaml")
        exit(1)
        
    print(f"Running script '{SCRIPT_TO_RUN}' on dataset '{DATASET}'")
    
    # Load config file
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Dynamically load and run the desired script!
    script_module = importlib.import_module(f"scripts.{SCRIPT_TO_RUN}")
    script_module.run(config["datasets"][DATASET])

