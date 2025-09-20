import json
import subprocess
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import requests

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    original_cwd = get_original_cwd()

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            with wandb.init(job_type="download_data") as run:
                artifact = wandb.Artifact(
                    name=config["etl"]["sample"],  # This is "sample.csv" from your config
                    type="raw_data",
                    description="Raw data from local file"
                )
                
                # Get the full path to the local file
                local_file_path = os.path.join(original_cwd, "data", config["etl"]["sample"])
                
                # Add the local file to the artifact and log it
                artifact.add_file(local_file_path)
                run.log_artifact(artifact)
                artifact.wait()

        if "basic_cleaning" in active_steps:
            subprocess.run(
                [
                    "python", "run.py",
                    "--input_artifact", "sample.csv:latest",
                    "--output_artifact", config["etl"]["output_artifact"],
                    "--output_type", config["etl"]["output_type"],
                    "--output_description", config["etl"]["output_description"],
                    "--min_price", str(config["etl"]["min_price"]),
                    "--max_price", str(config["etl"]["max_price"]),
                ],
                check=True,
                cwd=os.path.join(original_cwd, "src/basic_cleaning")
            )
            pass

        if "data_check" in active_steps:
            subprocess.run(
                [
                    "pytest", ".", "-vv",
                    "--csv", "clean_sample:latest",
                    "--ref", "clean_sample:reference",
                    "--kl_threshold", str(config["data_check"]["kl_threshold"]),
                    "--min_price", str(config["etl"]["min_price"]),
                    "--max_price", str(config["etl"]["max_price"]),
                ],
                check=True,
                cwd=os.path.join(original_cwd, "src/data_check")
            )
            pass

        if "data_split" in active_steps:
            _ = mlflow.run(
            f"{config['main']['components_repository']}/train_val_test_split",
            'main',
            parameters={
                "input": "clean_sample:latest",
                "test_size": config["modeling"]["test_size"],
                "random_seed": config["modeling"]["random_seed"],
                "stratify_by": config["modeling"]["stratify_by"]
            }
        )
            pass

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step
            subprocess.run(
            [
                "python", "run.py",
                "--trainval_artifact", "trainval_data:latest",
                "--val_size", str(config["modeling"]["val_size"]),
                "--random_seed", str(config["modeling"]["random_seed"]),
                "--stratify_by", config["modeling"]["stratify_by"],
                "--rf_config", rf_config,  
                "--max_tfidf_features", str(config["modeling"]["max_tfidf_features"]),
                "--output_artifact", config["modeling"]["output_artifact"],
            ],
            check=True,
            cwd=os.path.join(original_cwd, "src/train_random_forest")
        )

            ##################
            # Implement here #
            ##################

            pass

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################
            _ = mlflow.run(
            f"{config['main']['components_repository']}/test_regression_model",
            "main",
            parameters={
                "mlflow_model": "random_forest_export:prod",
                "test_dataset": "test_data:latest"
            },
        )

            pass


if __name__ == "__main__":
    go()
