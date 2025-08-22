import yaml
import importlib

####### CHANGE THIS PART TO RUN DIFFERENT SCRIPTS/DATASETS #######
SCRIPT_TO_RUN = "CNN"   # no .py extension

# See config.yaml for available datasets
# Either specify a single dataset (e.g., ["sim_lflr_tf"]), or
# 3 datasets (e.g., ["cuedts_tf", "cuedts_sw", "cuedts_cap"]. Only for CNN_multichannel)
DATASET_NAME = ["axcpt_tf"]  # must always be a list
###################################################################


if __name__ == "__main__":
    # Load config file
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Error handling for script and dataset selection
    try:
        script_module = importlib.import_module(f"scripts.{SCRIPT_TO_RUN}")
    except ModuleNotFoundError:
        print(f"Error: Script '{SCRIPT_TO_RUN}' not found in scripts/")
        exit(1)

    assert isinstance(DATASET_NAME, list), \
        "DATASET_NAME must be a list, even if it contains a single dataset"
    assert len(DATASET_NAME) in [1, 3], \
        "DATASET_NAME must contain either a single dataset or exactly 3 datasets"
    for ds in DATASET_NAME:
        if ds not in config["datasets"]:
            print(f"Error: Dataset '{ds}' not found in config.yaml")
            exit(1)
        
    print(f"Running script '{SCRIPT_TO_RUN}' on dataset(s) '{DATASET_NAME}'")

    # Dynamically load and run the desired script
    script_module = importlib.import_module(f"scripts.{SCRIPT_TO_RUN}")
    script_module.run(config, DATASET_NAME)

