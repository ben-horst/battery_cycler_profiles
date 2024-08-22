import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import click
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidateItem:
    value: Any
    type: str
    criteria: Optional[str] = None
    units: Optional[str] = None


validator_result_dict: Dict[str, Dict] = {}

#test run with:
#sudo python3 validators/validator_ben_test.py --csv_file sample_log.csv --validator_result_json test_json.json

# ------------------------------------------------
# Validation function (modify this function)
# ------------------------------------------------
def validate_csv(
    csv_file: Optional[str] = None, validator_result_json: Optional[str] = None
) -> bool:
    """
    This function reads the csv files and validates them.
    input:
        csv_file: csv file to validate
    output:
        test_passed: True if all csv files have at least one row, False otherwise
    """
    if csv_file is None:
        raise ValueError("No CSV file provided")

    test_passed = True

    # TODO: Add more validation tests here
    # example: check if the csv file has at least one row
    row_count = 0
    max_voltage = 0
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        row_count = sum(1 for row in reader)
        if row_count == 0:
            logger.error(f"{csv_file} has 0 rows")
            test_passed = False
        data = pd.read_csv(csv_file)
        max_voltage = data['pack_voltage'].max()

    # TODO: Add the validation result to the dictionary, so it can be written to a JSON file
    # Please use the ValidateItem dataclass to store the validation results
    validator_result_dict["csv_row_count"] = asdict(
        ValidateItem(value=row_count, type="int", criteria="x>0", units="rows")
    )
    logger.debug(f"csv_row_count: {row_count}")

    validator_result_dict["max_voltage"] = asdict(
        ValidateItem(value=max_voltage, type="float", criteria="x<50", units="V")
    )
    logger.debug(f"max_voltage: {max_voltage}")

    # (Do not modify) Save the validation result to a JSON file
    save_validation_result(test_passed, validator_result_json)

    return test_passed


# ------------------------------------------------
# Command line interface (Do not modify)
# ------------------------------------------------
@click.command()
@click.option("--csv_file", type=click.Path(), help="csv file path", default=None)
@click.option(
    "--validator_result_json",
    type=click.Path(),
    help="validation result json file path",
    default=None,
)
@click.pass_context
def _cli(ctx, *args, **kwargs):
    main(**ctx.params)


def save_validation_result(
    validation_result: bool, validator_result_json: Optional[str] = None
):
    if validator_result_json is None:
        raise ValueError("No validator result json provided")

    # Add the csv file to the dictionary
    validator_result_dict["validator_file"] = asdict(
        ValidateItem(value=os.path.basename(validator_result_json), type="str")
    )
    # Add the validation result to the dictionary
    validator_result_dict["validation_result"] = asdict(
        ValidateItem(
            value=validation_result, type="bool", criteria="x==True", units="status"
        )
    )
    # Write the dictionary to a JSON file
    with open(validator_result_json, "w") as f:
        json.dump(validator_result_dict, f)


def main(csv_file: Optional[str] = None, validator_result_json: Optional[str] = None):
    # Configure logging.
    logging.basicConfig(level=logging.DEBUG)

    if csv_file is None or validator_result_json is None:
        raise ValueError("csv_file and validator_result_json must be provided")

    # Log the script filename
    script_name = os.path.basename(__file__)
    logger.debug(f"Running script: {script_name}")

    # Validate the csv file
    if validate_csv(csv_file=csv_file, validator_result_json=validator_result_json):
        logger.debug("validation_result: passed")
    else:
        logger.error("validation_result: failed")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # Command line arguments provided
        _cli()  # Use click interface
