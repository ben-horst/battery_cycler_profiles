import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, List

import click
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

pd.options.mode.copy_on_write = True

logger = logging.getLogger(__name__)


@dataclass
class ValidateItem:
    value: Any
    type: str
    criteria: Optional[str] = None
    units: Optional[str] = None

@dataclass
class AttachFile:
    key: str
    file_path: str


validator_result_dict: Dict[str, Dict] = {}
attached_files: List[AttachFile] = []

IDEAL_CAPACITY = 4.5*6    #battery capacity in Ah

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
    BRICK_DELTA_CRITERIA = 0.05

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

    #pulse train calculations
    SMALL_PULSE_CURRENT = 27.0     #current of small pulse in A
    LARGE_PULSE_CURRENT = 108.0    #current of large pulse in A
    PULSE_SAMPLING_INTERVAL = 0.5  #interval to average current and voltage measurements over to calculate DCIR
    amp_s_before_pulses = 0
    pulse_data = pd.DataFrame(columns=['time','amp-s','current','voltage'])

    #slow discharge/charge calculations
    DCHG_CHG_SAMPING_INTERVAL = 5.0  #interval to average current measurements over to calculate total charge capacity
    amp_s_before_discharge = 0
    amp_s_before_charge = 0
    discharge_data = pd.DataFrame(columns=['time','amp-s','current','voltage'])
    charge_data = pd.DataFrame(columns=['time','amp-s','current','voltage'])

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

            # Remove rows where 'step_name' is 'Unknown'
            data = data[data['step_name'] != 'Unknown']

            if not data.empty:
                
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

                #find pulse train portion and calcualte DCIR at 1C & 4C for every SOC
                #find rows where step_name is "10min_rest_before_pulse" and find the starting current_counter value
                amp_s_before_pulses = find_initial_amp_s(data, '10min_rest_before_pulse', 'max', amp_s_before_pulses)
                # find rows where step_name is "C_pulse_"... and run moving average to time sync all signals
                new_data = perform_average_and_update_data(data, 'C_pulse_', PULSE_SAMPLING_INTERVAL)
                pulse_data = pd.concat([pulse_data, new_data])
            

                #find c/5 discharge portion and calculate total charge capacity
                amp_s_before_discharge = find_initial_amp_s(data, '30min_rest_before_slow_discharge', 'max', amp_s_before_discharge)
                # find rows where step_name is c/5_discharge_to_0_percent' and run moving average to time sync all signals
                new_data = perform_average_and_update_data(data, 'c/20_discharge_to_0_percent', DCHG_CHG_SAMPING_INTERVAL)
                discharge_data = pd.concat([discharge_data, new_data])

                #find c/5 charge portion and calculate total charge capacity
                #find rows where step_name is "30min_rest_before_slow_charge" and find the starting current_counter value
                amp_s_before_charge = find_initial_amp_s(data, '30min_rest_before_slow_charge', 'min', amp_s_before_charge)
                # find rows where step_name is c/5_charge_to_100_percent' and run moving average to time sync all signals
                new_data = perform_average_and_update_data(data, 'c/20_charge_to_100_percent', DCHG_CHG_SAMPING_INTERVAL)
                charge_data = pd.concat([charge_data, new_data])
                
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
            
            else:   #if the chunk is empty, all parsing is skipped and the chunk is a pass
                print('empty chunk')
                chunk_passed = True
            
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
    

    #calculate total discharge and charge capacities
    total_discharge_capacity = abs(discharge_data['amp-s'].max() - discharge_data['amp-s'].min()) / 3600   #convert from A*s to Ah
    total_charge_capacity = abs(charge_data['amp-s'].max() - charge_data['amp-s'].min()) / 3600   #convert from A*s to Ah
    measured_capacity = (total_discharge_capacity + total_charge_capacity) / 2      #average of the two capacities

    #calculate SOC columns for each dataset, both with the measured capacity and the ideal capacity
    pulse_data['soc'], pulse_data['soc_ideal'], pulse_data['consumed_ah'] = calculate_soc(pulse_data, amp_s_before_pulses, measured_capacity, IDEAL_CAPACITY)
    discharge_data['soc'], discharge_data['soc_ideal'], discharge_data['consumed_ah'] = calculate_soc(discharge_data, amp_s_before_discharge, measured_capacity, IDEAL_CAPACITY)
    charge_data['soc'], charge_data['soc_ideal'], charge_data['consumed_ah'] = calculate_soc(charge_data, amp_s_before_discharge, measured_capacity, IDEAL_CAPACITY)

    #separate pulse data into the 1C and 4C portions
    small_pulse_data = pulse_data[(abs(pulse_data['current']) > (SMALL_PULSE_CURRENT - 1)) & (abs(pulse_data['current']) < (SMALL_PULSE_CURRENT + 1))]
    large_pulse_data = pulse_data[(abs(pulse_data['current']) > (LARGE_PULSE_CURRENT - 2)) & (abs(pulse_data['current']) < (LARGE_PULSE_CURRENT + 2))]
    #use the c/5 discharge data to interpolate an approximated OCV curve then calculate DCIR at each step
    small_pulse_data['ocv'], small_pulse_data['dcir'] = interpolate_ocv_and_calc_dcir(small_pulse_data, discharge_data)
    large_pulse_data['ocv'], large_pulse_data['dcir'] = interpolate_ocv_and_calc_dcir(large_pulse_data, discharge_data)
    #calculate DCIR curves
    dcir_4C_1s, _, _ = generate_dcir_durve(small_pulse_data, 1)
    dcir_4C_3s, _, _ = generate_dcir_durve(large_pulse_data, 3)
    
    generate_plots(discharge_data, charge_data, dcir_4C_1s, dcir_4C_3s)

    #generate nice exportable data, with a standardized SOC axis
    #export_data = pd.DataFrame(columns=['soc', 'charge_voltage', 'discharge_voltage', 'dcir_small_pulse', 'dcir_large_pulse'])

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

    validator_result_dict["total_discharge_capacity"] = asdict(ValidateItem(value=total_discharge_capacity, type="float", criteria="x>0", units="Ah"))
    logger.debug(f"total_discharge_capacity: {total_discharge_capacity}")

    validator_result_dict["total_charge_capacity"] = asdict(ValidateItem(value=total_charge_capacity, type="float", criteria="x>0", units="Ah"))
    logger.debug(f"total_charge_capacity: {total_charge_capacity}")

    attached_files.append(AttachFile(
        key='ocv curve', file_path='ocv_curve.png'))
    
    attached_files.append(AttachFile(
        key='dcir curve', file_path='dcir_curve.png'))

    # (Do not modify) Save the validation result to a JSON file
    save_validation_result(overall_test_passed, validator_result_json)

    return overall_test_passed

def perform_average_and_update_data(data, stepname, sampling_interval):
    all_rows = data[data['step_name'].str.contains(stepname)]
    averaged_data = pd.DataFrame()
    if not all_rows.empty:
        first_time_pulse = math.ceil(all_rows['approx_realtime_sec'].min() / sampling_interval) * sampling_interval   #round up to the nearest interval
        last_time_pulse = math.floor(all_rows['approx_realtime_sec'].max() / sampling_interval) * sampling_interval   #round down to the nearest interval
        t = first_time_pulse
        while t <= last_time_pulse:
            shunt_current = all_rows[(all_rows['approx_realtime_sec'] >= t) & (all_rows['approx_realtime_sec'] < t+sampling_interval) & (all_rows['identifier'] == '/htf.shunt.status')]['current'].mean()
            shunt_voltage = all_rows[(all_rows['approx_realtime_sec'] >= t) & (all_rows['approx_realtime_sec'] < t+sampling_interval) & (all_rows['identifier'] == '/htf.shunt.status')]['voltage'].mean()
            voltage = all_rows[(all_rows['approx_realtime_sec'] >= t) & (all_rows['approx_realtime_sec'] < t+sampling_interval) & (all_rows['identifier'] == '/battery_monitor.zip_battery_telemetry')]['pack_voltage'].mean()
            amp_s = all_rows[(all_rows['approx_realtime_sec'] >= t) & (all_rows['approx_realtime_sec'] < t+sampling_interval) & (all_rows['identifier'] == '/htf.shunt.status')]['current_counter'].mean()
            new_row = {'time': [t], 'amp-s': [amp_s], 'current': [shunt_current], 'voltage': [voltage], 'shunt_voltage': [shunt_voltage]}
            averaged_data = pd.concat([averaged_data, pd.DataFrame(new_row)])
            t += sampling_interval
    return averaged_data

def find_initial_amp_s(data, stepname, direction, current_initial_amp_s):
    #find rows where the identifier and search for either the max or min amp_s value, depending on direction
    amp_s = data[(data['step_name'] == stepname) & (data['identifier'] == '/htf.shunt.status')]['current_counter']
    if amp_s is not None:
        if direction == 'max':
            return max(current_initial_amp_s, amp_s.max())
        elif direction == 'min':
            return min(current_initial_amp_s, amp_s.min())
        
def calculate_soc(data, amp_s_at_100soc, capacity, nameplate_capacity):
    consumed_ah = (amp_s_at_100soc - data['amp-s']) / 3600
    soc_real = 100 * (1 - (consumed_ah / capacity))
    soc_ideal = 100 * (1 - (consumed_ah / nameplate_capacity))
    return soc_real, soc_ideal, consumed_ah


def interpolate_ocv_and_calc_dcir(pulse_data, dchg_data):
    #open circuit voltage approximated by c/5 discharge data
    ocv_curve = dchg_data[['soc', 'voltage']]   #create a curve of voltage vs SOC from the discharge data
    #sort by SOC and clean things up for interpolation
    ocv_curve = ocv_curve.sort_values(by='soc')
    ocv_curve = ocv_curve.drop_duplicates(subset='soc', keep='first')
    #interpolate the voltage data into the pulse data
    ocv = np.interp(pulse_data['soc'], ocv_curve['soc'], ocv_curve['voltage'])
    #calculate DCIR = (OCV-Vpulse) / Ipulse
    dcir = (ocv - pulse_data['voltage']) / abs(pulse_data['current'])
    return ocv, dcir

def generate_dcir_durve(pulse_data, delay):
    #calculates a DCIR curve from the data at the points delay seconds after the start of each pulse
    dcir_diffs = pulse_data['dcir'].diff()
    #find the locations where the pulses start by zero crossing of dcir derivative
    dcir_curve = pd.DataFrame()
    pulse_starts = pulse_data[(dcir_diffs.shift(-1) > 0) & (dcir_diffs < 0)]
    pulse_ends = pulse_data[(dcir_diffs.shift(-1) < 0) & (dcir_diffs > 0)]
    for t in pulse_starts['time']:
        data_after_delay = pulse_data[pulse_data['time'] >= t + delay]
        dcir_curve = pd.concat([dcir_curve, data_after_delay.iloc[0:1]])
    return dcir_curve, pulse_starts, pulse_ends

def generate_plots(discharge_data, charge_data, dcir_4C_1s, dcir_4C_3s):
    #Suppress logs from specific modules that are loud and not useful.
    try:
        suppress_logger = logging.getLogger('matplotlib')
        suppress_logger.setLevel(logging.CRITICAL)
        suppress_logger = logging.getLogger('PIL.PngImagePlugin')
        suppress_logger.setLevel(logging.CRITICAL)
    except Exception as e:
        pass

    plt.figure()
    plt.plot(discharge_data['consumed_ah'], discharge_data['voltage'], label='discharge')
    plt.plot(charge_data['consumed_ah'], charge_data['voltage'], label='charge')
    plt.gca().invert_xaxis()
    plt.xlabel('consumed Ah')
    plt.ylabel('voltage')
    plt.legend()
    plt.title('C/20 Charge/Discharge')
    plt.savefig('ocv_curve.png')

    plt.figure()
    plt.plot(dcir_4C_1s['consumed_ah'], dcir_4C_1s['dcir'], label='4C/1s')
    plt.plot(dcir_4C_3s['consumed_ah'], dcir_4C_3s['dcir'], label='4C/3s')
    plt.gca().invert_xaxis()
    plt.legend()
    plt.xlabel('consumed Ah')
    plt.ylabel('resistance')
    plt.title('DCIR')
    plt.savefig('dcir_curve.png')


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
     # Add the attached files to the dictionary
    for attached_file in attached_files:
        validator_result_dict[attached_file.key] = asdict(
            ValidateItem(value=attached_file.file_path, type="str")
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