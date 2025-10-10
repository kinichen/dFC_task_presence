import yaml
import importlib


def main(script, dataset_name, date_str, description="None"):
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
    
    print(f"Description: {description}, Date/Version: {date_str}")
    print(f"Running script '{script}' on dataset(s) '{dataset_name}'")

    # Dynamically load and run the desired script
    script_module = importlib.import_module(f"scripts.{script}")
    script_module.run(config, dataset_name, date_str)


if __name__ == "__main__":
    ######################### CHANGE THIS FOR RUNS ############################
    script = "CNN"  # CNN or GCN (if want multichannel, set script = "CNN"; dataset_name = list of 3 datasets)
    dataset_name = ["axcpt_tf"]  # Multichannel e.g. ["stern_sw", "stern_tf", "stern_cap"]; Single e.g. ["axcpt_cap"]
    run_whole_paradigm = False   # Set to False to submit job for only one dataset (non-CNN_multichannel)
    description = "Resnet18, AdamW, No freezing, Dropout=0.3, 3 outer CV folds, batchnorm before classifier"
    
    # *** Version of the performance output. Must change to avoid overwriting previous results ***
    date_str = "20251010"
    ############################################################################

    if len(dataset_name) == 3:
        assert script == "CNN", "If using 3 datasets, script must be 'CNN' for multichannel"
        run_whole_paradigm = False # if multichannel, methods for each paradigm are collapsed into one run, so no need to loop
        
    if not run_whole_paradigm:   # Only run on one dataset
        main(script, dataset_name, date_str, description)
        
    else: # For GCN and single-channel CNN, loop through all method datasets of one paradigm for job submission convenience
        for dataset in [["stroop_sw"], ["stroop_tf"], ["stroop_cap"]]:
            main(script, dataset, date_str, description)