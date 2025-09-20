import yaml
import importlib


def main(script, dataset_name, description="None"):
    '''Main function to run specified script on given dataset(s) with 
    an optional description.
    Parameters:
        script (str): Name of the script to run (without .py extension).
        dataset_name (list): List of dataset name(s) to use. See config.yaml.
            E.g., ["cuedts_tf"] for scripts that use one dataset, or 
            ["cuedts_tf", "cuedts_sw", "cuedts_cap"] only for CNN_multichannel
        description (str): Description of changes from last version for logging
    '''
    # Load config file
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Error handling for script and dataset selection
    try:
        script_module = importlib.import_module(f"scripts.{script}")
    except Exception as e:
        import traceback
        print("Import script failed with exception:")
        traceback.print_exc()
        exit(1)

    assert isinstance(dataset_name, list), \
        "DATASET_NAME must be a list, even if it contains a single dataset"
    assert len(dataset_name) in [1, 3], \
        "DATASET_NAME must contain either a single dataset or exactly 3 datasets"
    for ds in dataset_name:
        if ds not in config["datasets"]:
            print(f"Error: Dataset '{ds}' not found in config.yaml")
            exit(1)
    
    print(f"Description of changes: {description}")
    print(f"Running script '{script}' on dataset(s) '{dataset_name}'")

    # Dynamically load and run the desired script
    script_module = importlib.import_module(f"scripts.{script}")
    script_module.run(config, dataset_name)


if __name__ == "__main__":
    script = "GCN"  # Change this to run different scripts
    description = "Running script on multiple datasets at once."

    if script == "CNN_multichannel": # Only this script uses multiple datasets at once
        main(script, ["axcpt_sw", "axcpt_tf", "axcpt_cap"], description)
    else: # Run other scripts on individual datasets; loop for job submission convenience
        for dataset in [["axcpt_sw"], ["axcpt_tf"], ["axcpt_cap"]]:
            main(script, dataset, description)