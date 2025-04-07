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
        and re.match(rf"output_\d{{3}}_{date_str}", name)
    ]

    # Determine the next available run number
    run_ids = [int(name.split("_")[1]) for name in existing]
    next_id = max(run_ids) + 1 if run_ids else 0

    # Build the new subfolder name and path
    new_dir_name = f"output_{next_id:03d}_{date_str}"
    full_path = os.path.join(base_dir, new_dir_name)

    # Create the subfolder
    os.makedirs(full_path, exist_ok=True)

    return full_path
