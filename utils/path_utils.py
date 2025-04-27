# Copyright (c) 2022 @ FBK - Fondazione Bruno Kessler
# Author: Roberto Doriguzzi-Corin
# Project: LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import datetime

def create_output_subfolder(base_dir="./output/"):
    """
    Create a unique subfolder under the given base directory using the format:
    output_XXX_YYMMDD, where XXX is an incrementing number and YYMMDD is today's date.

    This is useful for organizing output from repeated experiments without overwriting previous results.

    Args:
        base_dir (str): The root directory where output folders should be created.

    Returns:
        str: The full path to the newly created subfolder.
    """
    # Get today's date as YYMMDD
    date_str = datetime.datetime.now().strftime("%y%m%d")

    # Create the base output directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # List existing subfolders that match the output_XXX_YYMMDD pattern
    existing = [
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
        and re.match(rf"output_{date_str}_\d{{3}}", name)
    ]

    # Determine the next available run number
    run_ids = [int(name.split("_")[2]) for name in existing]
    next_id = max(run_ids) + 1 if run_ids else 0

    # Build the new subfolder name and path
    new_dir_name = f"output_{date_str}_{next_id:03d}"
    full_path = os.path.join(base_dir, new_dir_name)

    # Create the subfolder
    os.makedirs(full_path, exist_ok=True)

    return full_path

def get_output_path(output_folder: str, filename: str) -> str:
    """
    Join a filename with the output folder path safely.

    Args:
        output_folder (str): Path to the experiment's output directory
        filename (str): Name of the file (e.g., 'model.h5', 'predictions.csv')

    Returns:
        str: Full path to the target file inside the output folder
    """
    return os.path.join(output_folder, filename)

def get_model_basename(time_window: int, max_flow_len: int, model_name: str) -> str:
    """
    Construct a base filename for a model using its parameters.

    The format is: {time_window}t-{max_flow_len}n-{model_name}

    This is helpful for keeping output files organized and consistently named
    based on their experimental settings.

    Args:
        time_window (int): The time window used during dataset creation
        max_flow_len (int): The maximum number of packets per flow
        model_name (str): The name of the model or dataset (e.g., 'SYN2020-LUCID')

    Returns:
        str: A formatted base filename (e.g., '10t-20n-SYN2020-LUCID')
    """
    return f"{time_window}t-{max_flow_len}n-{model_name}"


def get_model_path(output_folder: str, model_basename: str) -> str:
    """
    Build the full path to a model output file based on its basename and output folder.

    This does not add a file extension (e.g., '.h5') — it returns the base path to be extended as needed.

    Args:
        output_folder (str): The path to the directory where model files are saved
        model_basename (str): The base filename of the model (e.g., '10t-20n-SYN2020-LUCID')

    Returns:
        str: The full path to the model file (e.g., './output/output_001_250410/10t-20n-SYN2020-LUCID')
    """
    return os.path.join(output_folder, model_basename)
