import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import click
import pandas as pd
import math

logger = logging.getLogger(__name__)


@dataclass
class ValidateItem:
    value: Any
    type: str
    criteria: Optional[str] = None
    units: Optional[str] = None


validator_result_dict: Dict[str, Dict] = {}



#test run with:
#sudo python3 validators/validator_ZB04.py --csv_file sample_logs/ZB04_sample_log.csv --validator_result_json sample_logs/test_json.json

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

    CSV_CHUNK_SIZE_ROWS = 1000 # Define the chunk size (number of rows per chunk)
    chunks_passed = []  #list of passing results for each chunk of data analyzed
    chunk_count = 0
    #---validation tests---
    row_count = 0
    
    #pack min & max voltages - must be 35 V - 59 V
    chunks_max_voltage = []     #list of max voltage at each chunk
    MAX_VOLTAGE_CRITERIA = 59
    chunks_min_voltage = []     #list of min voltge at each chunk
    MIN_VOLTAGE_CRITERIA = 35

    #brick delta - difference between bricks can not exceed 50 mV
    chunks_max_brick_delta = []     #list of max brick delta at each chunk
    BRICK_DELTA_CRITERIA = 0.10

    #cell min & max voltages - must be 2.5 V - 4.21 V
    chunks_max_brick_voltage = []     #list of max cell voltage at each chunk
    MAX_BRICK_VOLTAGE_CRITERIA = 4.21
    chunks_min_brick_voltage = []     #list of min cell voltage at each chunk
    MIN_BRICK_VOLTAGE_CRITERIA = 2.5

    #brick voltages - all brick voltages must be numeric values
    overall_missing_brick_voltages = set() #set of all missing brick voltages

    #temperatures - all temperatures must be numeric values between -20 and 100 degrees C
    overall_missing_temperatures = set() #set of missing all missing temperatures
    MAX_CELL_TEMP_CRITERIA = 100
    MIN_CELL_TEMP_CRITERIA = -20

    #current error - difference between pack current and shunt current cannot exceed
    CURRENT_SAMPLING_INTERVAL = 1.0      #interval to average current measurements over to compare pack and shunt
    CURRENT_ERROR_CRITERIA = 5.0
    current_sampling_data = pd.DataFrame(columns=['time','pack_current', 'shunt_current'])

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        row_count = sum(1 for row in reader)
        if row_count == 0:
            logger.error(f"{csv_file} has 0 rows")
            test_passed = False
        logger.debug(f"row_count: {row_count}")
        

        #read csv file into pandas dataframe, chunk by chunk
        data_iterator = pd.read_csv(csv_file, chunksize=CSV_CHUNK_SIZE_ROWS)

        # Process each chunk
        # for data in data_iterator:
        for iteration, data in enumerate(data_iterator, start=0):
            logger.debug(f"Processing chunk {iteration}")

            #find all rows where the identifier is /battery_monitor.zip_battery_telemetry
            battery_monitor_data = data[data['identifier'] == '/battery_monitor.zip_battery_telemetry']

            #find max and min pack voltage at anytime in this chunk of data
            max_voltage = battery_monitor_data['pack_voltage'].max()
            min_voltage = battery_monitor_data['pack_voltage'].min()

            #check against criteria
            max_voltage_okay = max_voltage < MAX_VOLTAGE_CRITERIA
            min_voltage_okay = min_voltage > MIN_VOLTAGE_CRITERIA

            #update list of maxima/minima
            chunks_max_voltage.append(max_voltage)
            chunks_min_voltage.append(min_voltage)

            #make column of highest and lowest cell brick voltages to find brick deltas
            highest_brick_voltage = battery_monitor_data.filter(like='brick_voltage[').max(axis=1)
            lowest_brick_voltage = battery_monitor_data.filter(like='brick_voltage[').min(axis=1)
            brick_delta = highest_brick_voltage - lowest_brick_voltage
            max_brick_delta = brick_delta.max()
            brick_delta_okay = max_brick_delta < BRICK_DELTA_CRITERIA
            #update list of drick deltas
            chunks_max_brick_delta.append(max_brick_delta)

            #check max and min cell voltages
            max_brick_voltage = highest_brick_voltage.max()
            min_brick_voltage = lowest_brick_voltage.min()
            max_brick_voltage_okay = max_brick_voltage < MAX_BRICK_VOLTAGE_CRITERIA
            min_brick_voltage_okay = min_brick_voltage > MIN_BRICK_VOLTAGE_CRITERIA
            chunks_max_brick_voltage.append(max_brick_voltage)
            chunks_min_brick_voltage.append(min_brick_voltage)

            #check that every entry in columns starting with "brick_voltage[" is a numeric value
            brick_voltages_okay = []
            for column in battery_monitor_data.filter(like='brick_voltage['):
                if pd.to_numeric(battery_monitor_data[column], errors='coerce').notna().all():
                    brick_voltages_okay.append(True)
                else:    
                    brick_voltages_okay.append(False)
            missing_brick_voltages = [i+1 for i, x in enumerate(brick_voltages_okay) if not x]
            all_brick_voltages_okay = all(brick_voltages_okay)
            #update set of missing brick voltages
            for brick in missing_brick_voltages:
                overall_missing_brick_voltages.add(brick)  

            #check that every entry in columns starting with temperature is a numeric value in a reasonable range
            temperatures_okay = []
            for column in battery_monitor_data.filter(like='pack_temperature['):
                if pd.to_numeric(battery_monitor_data[column], errors='coerce').between(MIN_CELL_TEMP_CRITERIA, MAX_CELL_TEMP_CRITERIA).all():
                    temperatures_okay.append(True)
                else:    
                    temperatures_okay.append(False)
            missing_temperatures = [i+1 for i, x in enumerate(temperatures_okay) if not x]
            all_temperatures_okay = all(temperatures_okay)
            #update set of missing temperatures
            for temp in missing_temperatures:
                overall_missing_temperatures.add(temp)

            #compare pack current measurements to current shunt measurements
            first_time = math.ceil(data['approx_realtime_sec'].min() / CURRENT_SAMPLING_INTERVAL) * CURRENT_SAMPLING_INTERVAL   #round up to the nearest interval
            last_time = math.floor(data['approx_realtime_sec'].max() / CURRENT_SAMPLING_INTERVAL) * CURRENT_SAMPLING_INTERVAL   #round down to the nearest interval
            t = first_time
            while t <= last_time:
                avg_pack_current = data[(data['approx_realtime_sec'] >= t) & (data['approx_realtime_sec'] < t+CURRENT_SAMPLING_INTERVAL) & (data['identifier'] == '/battery_monitor.zip_battery_telemetry')]['pack_current'].mean()
                avg_shunt_current = data[(data['approx_realtime_sec'] >= t) & (data['approx_realtime_sec'] < t+CURRENT_SAMPLING_INTERVAL) & (data['identifier'] == '/htf.shunt.status')]['current'].mean()
                new_row = {'time': [t], 'pack_current': [avg_pack_current], 'shunt_current': [avg_shunt_current]}
                current_sampling_data = pd.concat([current_sampling_data, pd.DataFrame(new_row)])
                t += CURRENT_SAMPLING_INTERVAL
            # perform moving average on 'pack_current'
            current_sampling_data['pack_current_smoothed'] = current_sampling_data['pack_current'].rolling(window=10).mean()
            current_sampling_data['shunt_current_smoothed'] = current_sampling_data['shunt_current'].rolling(window=10).mean()
            current_sampling_data['current_error'] = current_sampling_data['shunt_current_smoothed'] - current_sampling_data['pack_current_smoothed']
        
            current_error_okay = current_sampling_data[(current_sampling_data['time'] >= first_time) & (current_sampling_data['time'] <= last_time)]['current_error'].abs().max() < CURRENT_ERROR_CRITERIA

            #evaluate chunk test pass
            chunk_passed = all([max_voltage_okay,
                        min_voltage_okay,
                        brick_delta_okay,
                        max_brick_voltage_okay,
                        min_brick_voltage_okay,
                        all_brick_voltages_okay,
                        all_temperatures_okay, 
                        current_error_okay,
                            ])
            
            chunks_passed.append(chunk_passed)
            chunk_count += 1
            
            if not chunk_passed:
                fail_reason = ((not max_voltage_okay) * 'max voltage exceeded, ') + ((not min_voltage_okay) * 'min voltage exceeded, ') + ((not brick_delta_okay) * 'brick delta exceeded, ') + ((not max_brick_voltage_okay) * 'max brick voltage exceeded, ') + ((not min_brick_voltage_okay) * 'min brick voltage exceeded, ') + ((not all_brick_voltages_okay) * 'brick voltages not numeric, ') + ((not all_temperatures_okay) * 'temperatures not numeric, ') + ((not current_error_okay) * 'current error exceeded, ')
                fail_reason = fail_reason[:-2]
                logger.error(f"Test failed on chunk {iteration}: {fail_reason}")
        

    #compile overall validation test results
    num_chunks_passed = sum(chunks_passed)
    logger.debug(f"Number of chunks passed: {num_chunks_passed}/{chunk_count}")
    overall_max_voltage = max(chunks_max_voltage)
    overall_max_voltage_okay = overall_max_voltage < MAX_VOLTAGE_CRITERIA
    logger.debug(f"Max voltage okay: {overall_max_voltage_okay}")
    overall_min_voltage = min(chunks_min_voltage)
    overall_min_voltage_okay = overall_min_voltage > MIN_VOLTAGE_CRITERIA
    logger.debug(f"Min voltage okay: {overall_min_voltage_okay}")
    overall_max_brick_delta = max(chunks_max_brick_delta)
    overall_brick_delta_okay = overall_max_brick_delta < BRICK_DELTA_CRITERIA
    logger.debug(f"Brick delta okay: {overall_brick_delta_okay}")
    overall_missing_brick_voltages_str = ','.join(str(brick) for brick in overall_missing_brick_voltages) if overall_missing_brick_voltages else 'none'         #turn the set into a string to be uploaded, if empty make it 'none'
    overall_all_brick_voltages_okay = not bool(overall_missing_brick_voltages)  #true if this set is empty (no missing bricks)
    logger.debug(f"Brick voltages okay: {overall_all_brick_voltages_okay}")
    overall_max_brick_voltage = max(chunks_max_brick_voltage)
    overall_max_brick_voltage_okay = overall_max_brick_voltage < MAX_BRICK_VOLTAGE_CRITERIA
    logger.debug(f"Max brick voltage okay: {overall_max_brick_voltage_okay}")
    overall_min_brick_voltage = min(chunks_min_brick_voltage)
    overall_min_brick_voltage_okay = overall_min_brick_voltage > MIN_BRICK_VOLTAGE_CRITERIA
    logger.debug(f"Min brick voltage okay: {overall_min_brick_voltage_okay}")
    overall_missing_temperatures_str = ','.join(str(temp) for temp in overall_missing_temperatures) if overall_missing_temperatures else 'none'         #turn the set into a string to be uploaded, if empty make it 'none'
    overall_all_temperatures_okay = not bool(overall_missing_temperatures)  #true if this set is empty (no missing temperatures)
    logger.debug(f"Temperatures okay: {overall_all_temperatures_okay}")
    overall_max_current_error = current_sampling_data['current_error'].abs().max()
    overall_current_error_okay = overall_max_current_error < CURRENT_ERROR_CRITERIA
    logger.debug(f"Max current error okay: {overall_current_error_okay}")

    overall_test_passed = all([overall_max_voltage_okay,
                        overall_min_voltage_okay,
                        overall_brick_delta_okay,
                        overall_max_brick_voltage_okay,
                        overall_min_brick_voltage_okay,
                        overall_all_brick_voltages_okay,
                        overall_all_temperatures_okay,
                        overall_current_error_okay,
                            ])


    # Please use the ValidateItem dataclass to store the validation results
    validator_result_dict["csv_row_count"] = asdict(ValidateItem(value=row_count, type="int", criteria="x>0", units="rows"))
    logger.debug(f"csv_row_count: {row_count}")

    validator_result_dict["max_voltage"] = asdict(ValidateItem(value=overall_max_voltage, type="float", criteria=f"x<{MAX_VOLTAGE_CRITERIA}", units="V"))
    logger.debug(f"max_voltage: {overall_max_voltage}")

    validator_result_dict["min_voltage"] = asdict(ValidateItem(value=overall_min_voltage, type="float", criteria=f"x>{MIN_VOLTAGE_CRITERIA}", units="V"))
    logger.debug(f"min_voltage: {overall_min_voltage}")

    validator_result_dict["max_brick_delta"] = asdict(ValidateItem(value=overall_max_brick_delta, type="float", criteria=f"x<{BRICK_DELTA_CRITERIA}", units="V"))
    logger.debug(f"max_brick_delta: {overall_max_brick_delta}")

    validator_result_dict["max_brick_voltage"] = asdict(ValidateItem(value=overall_max_brick_voltage, type="float", criteria=f"x<{MAX_BRICK_VOLTAGE_CRITERIA}", units="V"))
    logger.debug(f"max_brick_voltage: {overall_max_brick_voltage}")

    validator_result_dict["min_brick_voltage"] = asdict(ValidateItem(value=overall_min_brick_voltage, type="float", criteria=f"x>{MIN_BRICK_VOLTAGE_CRITERIA}", units="V"))
    logger.debug(f"min_brick_voltage: {overall_min_brick_voltage}")

    validator_result_dict["missing_brick_voltages"] = asdict(ValidateItem(value=overall_missing_brick_voltages_str, type="str", criteria="x == 'none'", units="status"))
    logger.debug(f"missing_brick_voltages: {overall_missing_brick_voltages_str}")

    validator_result_dict["missing_temperatures"] = asdict(ValidateItem(value=overall_missing_temperatures_str, type="str", criteria="x == 'none'", units="status"))
    logger.debug(f"missing_temperatures: {overall_missing_temperatures_str}")

    validator_result_dict["max_current_error"] = asdict(ValidateItem(value=overall_max_current_error, type="float", criteria=f"-{CURRENT_ERROR_CRITERIA}<x<{CURRENT_ERROR_CRITERIA}", units="A"))
    logger.debug(f"max_current_error: {overall_max_current_error}")

    # (Do not modify) Save the validation result to a JSON file
    save_validation_result(overall_test_passed, validator_result_json)

    return overall_test_passed


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