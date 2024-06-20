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
#sudo python3 validators/validator_ZB47.py --csv_file sample_logs/sample_log_5-30-24.csv --validator_result_json sample_logs/test_json.json

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
    min_voltage = 0
    max_brick_delta = 0
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        row_count = sum(1 for row in reader)
        if row_count == 0:
            logger.error(f"{csv_file} has 0 rows")
            test_passed = False
        data = pd.read_csv(csv_file)

        #find all rows where the identifier is /battery_monitor.zip_battery_telemetry
        battery_monitor_data = data[data['identifier'] == '/battery_monitor.zip_battery_telemetry']

        #find max and min pack voltage at anytime in data
        max_voltage = battery_monitor_data['pack_voltage'].max()
        min_voltage = battery_monitor_data['pack_voltage'].min()
        max_voltage_criteria = 59
        min_voltage_criteria = 35
        max_voltage_okay = max_voltage < max_voltage_criteria
        min_voltage_okay = min_voltage > min_voltage_criteria

        #make column of highest and lowest cell brick voltages to find bricke deltas
        highest_brick_voltage = battery_monitor_data.filter(like='brick_voltage[').max(axis=1)
        lowest_brick_voltage = battery_monitor_data.filter(like='brick_voltage[').min(axis=1)
        brick_delta = highest_brick_voltage - lowest_brick_voltage
        max_brick_delta = brick_delta.max()
        brick_delta_criteria = 0.05
        max_brick_delta_okay = max_brick_delta < brick_delta_criteria
        max_brick_voltage = highest_brick_voltage.max()
        min_brick_voltage = lowest_brick_voltage.min()
        max_brick_voltage_criteria = 4.2
        min_brick_voltage_criteria = 2.5
        max_brick_voltage_okay = max_brick_voltage < max_brick_voltage_criteria
        min_brick_voltage_okay = min_brick_voltage > min_brick_voltage_criteria

        #check that every entry in columns starting with "brick_voltage[" is a numeric value
        brick_voltages_okay = []
        for column in battery_monitor_data.filter(like='brick_voltage['):
            if pd.to_numeric(battery_monitor_data[column], errors='coerce').notna().all():
                brick_voltages_okay.append(True)
            else:    
                brick_voltages_okay.append(False)

        missing_brick_voltages = [i+1 for i, x in enumerate(brick_voltages_okay) if not x]
        missing_brick_voltages_str = ','.join(str(x) for x in missing_brick_voltages)
        if missing_brick_voltages_str == '':
            missing_brick_voltages_str = 'none'
        all_brick_voltages_okay = all(brick_voltages_okay)

        #check that every entry in columns starting with temperature is a numeric value in a reasonable range
        temperatures_okay = []
        for column in battery_monitor_data.filter(like='temperature['):
            if pd.to_numeric(battery_monitor_data[column], errors='coerce').between(10, 50).all():
                temperatures_okay.append(True)
            else:    
                temperatures_okay.append(False)
        missing_temperatures = [i+1 for i, x in enumerate(temperatures_okay) if not x]
        missing_temperatures_str = ','.join(str(x) for x in missing_temperatures)
        if missing_temperatures_str == '':
            missing_temperatures_str = 'none'
        all_temperatures_okay = all(temperatures_okay)

        #compare pack current measurements to current shunt measurements
        current_error = []
        for index, row in data.iterrows():
            if row['identifier'] == '/battery_monitor.zip_battery_telemetry':
                pack_current = row['pack_current']
                i = 0
                while index+i < len(data):
                    if data.iloc[index+i]['identifier'] == '/shunt.status':
                        shunt_current = data.iloc[index+i]['current']
                        current_error.append(shunt_current - pack_current)
                        break
                    else:
                        i += 1
                
        
        max_current_error = max(current_error, key=abs)
        current_error_criteria = 5.0
        current_error_okay = abs(max_current_error) < current_error_criteria

        #evaluate overall test pass
        test_passed = all([max_voltage_okay,
                           min_voltage_okay,
                           max_brick_delta_okay,
                           all_brick_voltages_okay,
                           all_temperatures_okay, 
                           current_error_okay,
                           max_brick_voltage_okay,
                           min_brick_voltage_okay])
        

    # TODO: Add the validation result to the dictionary, so it can be written to a JSON file
    # Please use the ValidateItem dataclass to store the validation results
    validator_result_dict["csv_row_count"] = asdict(ValidateItem(value=row_count, type="int", criteria="x>0", units="rows"))
    logger.debug(f"csv_row_count: {row_count}")

    validator_result_dict["max_voltage"] = asdict(ValidateItem(value=max_voltage, type="float", criteria=f"x<{max_voltage_criteria}", units="V"))
    logger.debug(f"max_voltage: {max_voltage}")

    validator_result_dict["min_voltage"] = asdict(ValidateItem(value=min_voltage, type="float", criteria=f"x>{min_voltage_criteria}", units="V"))
    logger.debug(f"min_voltage: {min_voltage}")

    validator_result_dict["max_brick_delta"] = asdict(ValidateItem(value=max_brick_delta, type="float", criteria=f"x<{brick_delta_criteria}", units="V"))
    logger.debug(f"max_brick_delta: {max_brick_delta}")

    validator_result_dict["missing_brick_voltages"] = asdict(ValidateItem(value=missing_brick_voltages_str, type="str", criteria="x == 'none'", units="status"))
    logger.debug(f"missing_brick_voltages: {missing_brick_voltages_str}")

    validator_result_dict["missing_temperatures"] = asdict(ValidateItem(value=missing_temperatures_str, type="str", criteria="x == 'none'", units="status"))
    logger.debug(f"missing_temperatures: {missing_temperatures_str}")

    validator_result_dict["max_current_error"] = asdict(ValidateItem(value=max_current_error, type="float", criteria=f"-{current_error_criteria}<x<{current_error_criteria}", units="A"))
    logger.debug(f"max_current_error: {max_current_error}")

    validator_result_dict["max_brick_voltage"] = asdict(ValidateItem(value=max_brick_voltage, type="float", criteria=f"x<{max_brick_voltage_criteria}", units="V"))
    logger.debug(f"max_brick_voltage: {max_brick_voltage}")

    validator_result_dict["min_brick_voltage"] = asdict(ValidateItem(value=min_brick_voltage, type="float", criteria=f"x>{min_brick_voltage_criteria}", units="V"))
    logger.debug(f"min_brick_voltage: {min_brick_voltage}")

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
