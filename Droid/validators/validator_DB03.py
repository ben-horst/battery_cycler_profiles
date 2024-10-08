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
import datetime
import pickle

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

ATTACHED_FILE_KEY = "attached_file"
if sys.platform.startswith("linux"):
    ATTACHED_FILE_DIRECTORY = "/tmp/zipline-htf/attached_files"
else:
    ATTACHED_FILE_DIRECTORY = ""

plot_file_name = 'timeseries.png'
plot_file_path = os.path.join(ATTACHED_FILE_DIRECTORY, plot_file_name)

validation_data_file_name = 'validation_data.pkl'
validation_data_file_path = os.path.join(ATTACHED_FILE_DIRECTORY, validation_data_file_name)

battery_telemetry_data_file_name = 'battery_telemetry_data.pkl'
battery_telemetry_data_file_path = os.path.join(ATTACHED_FILE_DIRECTORY, battery_telemetry_data_file_name)

validator_result_dict: Dict[str, Dict] = {}
attached_files: List[AttachFile] = []

IDEAL_CAPACITY = 2.8    #battery capacity in Ah

#test run with:
#sudo python validators/validator_DB03.py --csv_file sample_logs/DB03_sample_log.csv --validator_result_json sample_logs/test_json.json

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
    start_time = 2000000000
    end_time = 0

    all_battery_data = pd.DataFrame()   #all useful battery data in the csv file
    
    #pack min & max voltages - must be 35 V - 59 V
    chunks_max_voltage = []     #list of max voltage at each chunk
    MAX_VOLTAGE_CRITERIA = 21.2
    chunks_min_voltage = []     #list of min voltge at each chunk
    MIN_VOLTAGE_CRITERIA = 12.4

    #brick delta - difference between bricks can not exceed 50 mV
    chunks_max_brick_delta = []     #list of max brick delta at each chunk
    BRICK_DELTA_CRITERIA = 0.10

    #cell min & max voltages - must be 2.5 V - 4.21 V
    chunks_max_brick_voltage = []     #list of max cell voltage at each chunk
    MAX_BRICK_VOLTAGE_CRITERIA = 4.21
    chunks_min_brick_voltage = []     #list of min cell voltage at each chunk
    MIN_BRICK_VOLTAGE_CRITERIA = 2.45

    #brick voltages - all brick voltages must be numeric values
    overall_missing_brick_voltages = set() #set of all missing brick voltages

    #temperatures - all temperatures must be numeric values between -20 and 100 degrees C
    overall_missing_temperatures = set() #set of missing all missing temperatures
    MAX_CELL_TEMP_CRITERIA = 100
    MIN_CELL_TEMP_CRITERIA = -20

    #current error - difference between pack current and shunt current cannot exceed
    CURRENT_SAMPLING_INTERVAL = 1.0      #interval to average current measurements over to compare pack and shunt
    CURRENT_ERROR_CRITERIA = 3.0
    current_sampling_data = pd.DataFrame(columns=['time', 'amp-s', 'shunt_current', 'pack_current', 'pack_voltage',])

    #check that the MOSFETs are open at the beginning of the test
    pretest_chunks_max_voltage = []     #list of max voltage at each chunk
    PRETEST_ON_VOLTAGE_CRITERIA = 12
    pretest_chunks_min_voltage = []     #list of min voltge at each chunk
    PRETEST_OFF_VOLTAGE_CRITERIA = 1.0

    #VSYS check - when VSYS is on, check the Vsys current
    VSYS_RESISTOR = 10      #resistance of load resistor applied to VSYS
    NOMINAL_VOLTAGE = 18
    VSYS_ERROR_CRITERIA = 0.5    #current threshold for VSYS to be considered on
    VSYS_SAMPLING_INTERVAL = 5.0    #interval to measure and average VSYS data
    vsys_data = pd.DataFrame(columns=['time', 'amp-s', 'shunt_current', 'pack_current', 'pack_voltage',])

    #heater check - when heater is on, it's resistance must be in range and temps should rise
    chunks_min_heater_resistance = []     #list of min heater resistance at each chunk
    HEATER_MIN_RESISTANCE_CRITERIA = 9
    chunks_max_heater_resistance = []     #list of max heater resistance at each chunk
    HEATER_MAX_RESISTANCE_CRITERIA = 15
    heater_temps = []
    HEATER_MIN_TEMP_RISE_CRITERIA = 5

    #dock thermistor check
    chunks_max_dock_resistance = []     #list of max dock resistance at each chunk
    DOCK_MAX_RESISTANCE_CRITERIA = 18000
    chunks_min_dock_resistance = []     #list of min dock resistance at each chunk
    DOCK_MIN_RESISTANCE_CRITERIA = 4000

        
    #read csv file into pandas dataframe, chunk by chunk
    data_iterator = pd.read_csv(csv_file, chunksize=CSV_CHUNK_SIZE_ROWS)
    
    # Process each chunk
    # for data in data_iterator:
    for iteration, data in enumerate(data_iterator, start=0):
        logger.debug(f"Processing chunk {iteration}")
        row_count += len(data)
        start_time = min(start_time, data['approx_realtime_sec'].min())
        end_time = max(end_time, data['approx_realtime_sec'].max())

        # check the unknown rows - mosfets should be open at first, but close during this phase
        pretest_rows = data[data['step_name'] == 'Unknown']
        pretest_chunks_max_voltage.append(pretest_rows[pretest_rows['identifier'] == '/htf.shunt.status']['voltage'].max())
        pretest_chunks_min_voltage.append(pretest_rows[pretest_rows['identifier'] == '/htf.shunt.status']['voltage'].min())

        # Remove rows where 'step_name' is 'Unknown'
        data = data[data['step_name'] != 'Unknown']

        if not data.empty:
            
            #find all rows where the identifier is /battery_monitor.droid_battery_telemetry and are relevant columns
            relevant_columns = ['step_name','approx_realtime_sec','pack_voltage','pack_current','brick_voltage[0]','brick_voltage[1]',
                                'brick_voltage[2]','brick_voltage[3]','brick_voltage[4]','brick_heater_temperature[thermistor_brick1]',
                                'brick_heater_temperature[thermistor_brick3_afe]','brick_heater_temperature[thermistor_brick5]','sys_current',
                                'is_fault_line_active','hw_undervoltage_vbatt','hw_overvoltage_vbatt','hw_ntc_undertemperature',
                                'hw_ntc_severe_overtemperature','hw_ntc_overtemperature','hw_persistent_short_circuit_discharge',
                                'hw_short_circuit_discharge','hw_persistent_overcurrent_discharge','hw_persistent_overcurrent_charge',
                                'hw_overcurrent_discharge','hw_overcurrent_charge','hw_vbatt_sum_check_fail','hw_faultn_ext']
            battery_monitor_data = data[(data['identifier'] == '/battery_monitor.droid_battery_telemetry')][relevant_columns]
            all_battery_data = pd.concat([all_battery_data, battery_monitor_data])

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

            #check that every entry in columns starting with brick_heater_temperature is a numeric value in a reasonable range
            temperatures_okay = []
            for column in battery_monitor_data.filter(like='brick_heater_temperature'):
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
            new_data = perform_average_and_update_data(data, '', CURRENT_SAMPLING_INTERVAL)
            if not new_data.empty:
                new_data['pack_current_smoothed'] = new_data['pack_current'].rolling(window=10).mean()
                new_data['shunt_current_smoothed'] = new_data['shunt_current'].rolling(window=10).mean()
                new_data['current_error'] = new_data['shunt_current_smoothed'] - new_data['pack_current_smoothed']
                current_error_okay = new_data['current_error'].fillna(0).abs().max() < CURRENT_ERROR_CRITERIA
            else:
                current_error_okay = True
            current_sampling_data = pd.concat([current_sampling_data, new_data])

            #check that during the VSYS test, the current is within the expected range
            new_data = perform_average_and_update_data(data, 'VSYS_ON', VSYS_SAMPLING_INTERVAL)
            if not new_data.empty:
                new_data['vsys_ideal_current'] = new_data['pack_voltage'] / VSYS_RESISTOR
                new_data['vsys_current_error'] = new_data['sys_current'] - new_data['vsys_ideal_current']
                new_data.loc[abs(new_data['sys_current']) < 0.75 * (NOMINAL_VOLTAGE / VSYS_RESISTOR), 'vsys_current_error'] = 0       #when vsys curent is less than 75% of expected value, it's probably off, so don't evaluate error  
                vsys_current_error_okay = new_data['vsys_current_error'].abs().max() < VSYS_ERROR_CRITERIA
            else:
                vsys_current_error_okay = True
            vsys_data = pd.concat([vsys_data, new_data])
            
            #check heater resistance values
            heater_rows = data[data['step_name'].str.contains('HEATER_ON')]
            if not heater_rows.empty:
                min_heater_resistance = heater_rows[heater_rows['identifier'] == '/htf.bk_power_supply.status']['resistance'].min()
                max_heater_resistance = heater_rows[heater_rows['identifier'] == '/htf.bk_power_supply.status']['resistance'].max()
                heater_resistance_okay = True
                if min_heater_resistance > 0:
                    chunks_min_heater_resistance.append(min_heater_resistance)
                    heater_resistance_okay = heater_resistance_okay and (min_heater_resistance > HEATER_MIN_RESISTANCE_CRITERIA)
                if max_heater_resistance > 0:
                    chunks_max_heater_resistance.append(max_heater_resistance)
                    heater_resistance_okay = heater_resistance_okay and (max_heater_resistance < HEATER_MAX_RESISTANCE_CRITERIA)
                heater_temps = heater_temps + (heater_rows['brick_heater_temperature[thermistor_brick1]'].dropna().to_list())
            else:
                heater_resistance_okay = True
                
            #check dock thermistor
            max_dock_resitance = data['dock_thermistor_resistance'].max()
            min_dock_resitance = data['dock_thermistor_resistance'].min()
            if max_dock_resitance > 0 and min_dock_resitance > 0:
                dock_resistance_okay = (max_dock_resitance < DOCK_MAX_RESISTANCE_CRITERIA) and (min_dock_resitance > DOCK_MIN_RESISTANCE_CRITERIA)
            else:
                dock_resistance_okay = True
            chunks_max_dock_resistance.append(max_dock_resitance)
            chunks_min_dock_resistance.append(min_dock_resitance)
            
            #evaluate chunk test pass
            chunk_passed = all([max_voltage_okay,
                        min_voltage_okay,
                        brick_delta_okay,
                        max_brick_voltage_okay,
                        min_brick_voltage_okay,
                        all_brick_voltages_okay,
                        all_temperatures_okay, 
                        current_error_okay,
                        vsys_current_error_okay,
                        heater_resistance_okay,
                        dock_resistance_okay,
                            ])
        
        else:   #if the chunk is empty, all parsing is skipped and the chunk is a pass
            print('empty chunk')
            chunk_passed = True
        
        chunks_passed.append(chunk_passed)
        chunk_count += 1
        
        if not chunk_passed:
            fail_reason = ((not max_voltage_okay) * 'max voltage exceeded, ') + ((not min_voltage_okay) * 'min voltage exceeded, ') + ((not brick_delta_okay) * 'brick delta exceeded, ') + ((not max_brick_voltage_okay) * 'max brick voltage exceeded, ') + ((not min_brick_voltage_okay) * 'min brick voltage exceeded, ') + ((not all_brick_voltages_okay) * 'brick voltages not numeric, ') + ((not all_temperatures_okay) * 'temperatures not numeric, ') + ((not current_error_okay) * 'current error exceeded, ') + ((not vsys_current_error_okay) * 'VSYS current error, ') + ((not heater_resistance_okay) * 'heater resistance failed, ') + ((not dock_resistance_okay) * 'dock resistance failed, ')
            fail_reason = fail_reason[:-2]
            logger.error(f"Test failed on chunk {iteration}: {fail_reason}")
        

    #compile overall validation test results
    all_battery_data['time_offset'] = all_battery_data['approx_realtime_sec'] - start_time
    total_seconds = int(end_time - start_time)
    elapsed_time = str(datetime.timedelta(seconds=total_seconds))

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

    #check MOSFET off/on during pretest
    overall_pretest_min_voltage = min(pretest_chunks_min_voltage)
    prestest_fets_off_okay = overall_pretest_min_voltage < PRETEST_OFF_VOLTAGE_CRITERIA
    logger.debug(f"Pretest off voltage okay: {prestest_fets_off_okay}")
    overall_pretest_max_voltage = max(pretest_chunks_max_voltage)
    prestest_fets_on_okay = overall_pretest_max_voltage > PRETEST_ON_VOLTAGE_CRITERIA
    logger.debug(f"Pretest on voltage okay: {prestest_fets_on_okay}")
    pretest_fets_okay = prestest_fets_off_okay and prestest_fets_on_okay


    #check VSYS current
    vsys_peak_current = vsys_data['sys_current'].max()
    avg_pack_voltage_during_vsys = vsys_data['pack_voltage'].mean()
    vsys_max_error = vsys_data['vsys_current_error'].abs().max()
    #if the peak vsys current is at least 50% of expected and error is within bounds, then the test passes
    if (vsys_peak_current > 0.50 * avg_pack_voltage_during_vsys / VSYS_RESISTOR):
        overall_vsys_current_error_okay = (vsys_max_error < VSYS_ERROR_CRITERIA)
    else:
        overall_vsys_current_error_okay = False
        vsys_max_error = 10   #if VSYS is not on, set the error to a high value to indicate a failure
    logger.debug(f"VSYS current okay: {overall_vsys_current_error_okay}")

    #check heater resistance
    overall_min_heater_resistance = min(chunks_min_heater_resistance)
    overall_max_heater_resistance = max(chunks_max_heater_resistance)
    overall_heater_resistance_okay = (overall_min_heater_resistance > HEATER_MIN_RESISTANCE_CRITERIA) and (overall_max_heater_resistance < HEATER_MAX_RESISTANCE_CRITERIA)
    logger.debug(f"Heater resistance okay: {overall_heater_resistance_okay}")

    #check heater temperature rise
    heater_temp_rise = max(heater_temps) - min(heater_temps)
    heater_temp_rise_okay = heater_temp_rise > HEATER_MIN_TEMP_RISE_CRITERIA
    logger.debug(f"Heater temp rise okay: {heater_temp_rise_okay}")

    #check dock thermistor
    overall_max_dock_resistance = max(chunks_max_dock_resistance)
    overall_min_dock_resistance = min(chunks_min_dock_resistance)
    overall_dock_resistance_okay = (overall_max_dock_resistance < DOCK_MAX_RESISTANCE_CRITERIA) and (overall_min_dock_resistance > DOCK_MIN_RESISTANCE_CRITERIA)
    logger.debug(f"Dock resistance okay: {overall_dock_resistance_okay}")

    overall_test_passed = all([overall_max_voltage_okay,
                        overall_min_voltage_okay,
                        overall_brick_delta_okay,
                        overall_max_brick_voltage_okay,
                        overall_min_brick_voltage_okay,
                        overall_all_brick_voltages_okay,
                        overall_all_temperatures_okay,
                        overall_current_error_okay,
                        overall_vsys_current_error_okay,
                        pretest_fets_okay,
                        overall_heater_resistance_okay,
                        heater_temp_rise_okay,
                        overall_dock_resistance_okay,
                            ])
    

    # Export data to a file
    with open(validation_data_file_path, "wb") as f:    #this is where to put any computed dataframes or variables that need to be saved
        pickle.dump({
            "current_sampling_data": current_sampling_data,
            "vsys_data": vsys_data
        }, f)

    with open(battery_telemetry_data_file_path, "wb") as f:     #this exports all the releveant battery telemetry data
        pickle.dump({
            "all_battery_data": all_battery_data
        }, f)
    
    try:
        generate_plots(all_battery_data)
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")


    # Please use the ValidateItem dataclass to store the validation results
    validator_result_dict["csv_row_count"] = asdict(ValidateItem(value=row_count, type="int", criteria="x>0", units="rows"))
    logger.debug(f"csv_row_count: {row_count}")

    validator_result_dict["elapsed_time"] = asdict(ValidateItem(value=elapsed_time, type="str", criteria=None, units="hh:mm:ss"))
    logger.debug(f"elapsed_time: {elapsed_time}")

    validator_result_dict["total_seconds"] = asdict(ValidateItem(value=total_seconds, type="int", criteria="x>0", units="seconds"))
    logger.debug(f"total_seconds: {total_seconds}")

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

    validator_result_dict["pretest_fets_off_voltage"] = asdict(ValidateItem(value=overall_pretest_min_voltage, type="float", criteria=f"x<{PRETEST_OFF_VOLTAGE_CRITERIA}", units="V"))
    logger.debug(f"pretest_fets_off_voltage: {overall_pretest_min_voltage}")

    validator_result_dict["pretest_fets_on_voltage"] = asdict(ValidateItem(value=overall_pretest_max_voltage, type="float", criteria=f"x>{PRETEST_ON_VOLTAGE_CRITERIA}", units="V"))
    logger.debug(f"pretest_fets_on_voltage: {overall_pretest_max_voltage}")

    validator_result_dict["vsys_max_error"] = asdict(ValidateItem(value=vsys_max_error, type="float", criteria=f"x<{VSYS_ERROR_CRITERIA}", units="A"))
    logger.debug(f"vsys_max_error: {vsys_max_error}")

    validator_result_dict["heater_min_resistance"] = asdict(ValidateItem(value=overall_min_heater_resistance, type="float", criteria=f"x>{HEATER_MIN_RESISTANCE_CRITERIA}", units="Ohm"))
    logger.debug(f"heater_min_resistance: {overall_min_heater_resistance}")

    validator_result_dict["heater_max_resistance"] = asdict(ValidateItem(value=overall_max_heater_resistance, type="float", criteria=f"x<{HEATER_MAX_RESISTANCE_CRITERIA}", units="Ohm"))
    logger.debug(f"heater_max_resistance: {overall_max_heater_resistance}")

    validator_result_dict["heater_temp_rise"] = asdict(ValidateItem(value=heater_temp_rise, type="float", criteria=f"x>{HEATER_MIN_TEMP_RISE_CRITERIA}", units="C"))
    logger.debug(f"heater_temp_rise: {heater_temp_rise}")

    validator_result_dict["dock_thermistor_max_resistance"] = asdict(ValidateItem(value=overall_max_dock_resistance, type="float", criteria=f"x<{DOCK_MAX_RESISTANCE_CRITERIA}", units="Ohm"))
    logger.debug(f"dock_max_resistance: {overall_max_dock_resistance}")

    validator_result_dict["dock_thermistor_min_resistance"] = asdict(ValidateItem(value=overall_min_dock_resistance, type="float", criteria=f"x>{DOCK_MIN_RESISTANCE_CRITERIA}", units="Ohm"))
    logger.debug(f"dock_min_resistance: {overall_min_dock_resistance}")
    
    attached_files.append(AttachFile(key=ATTACHED_FILE_KEY, file_path=plot_file_path))
    attached_files.append(AttachFile(key=ATTACHED_FILE_KEY, file_path=validation_data_file_path))
    attached_files.append(AttachFile(key=ATTACHED_FILE_KEY, file_path=battery_telemetry_data_file_path))

    # (Do not modify) Save the validation result to a JSON file
    save_validation_result(overall_test_passed, validator_result_json)

    return overall_test_passed

def perform_average_and_update_data(data, stepname, sampling_interval):
    #find all rows with the stepname and then transform the data onto a timeseries with the specified sampling interval, averaging the data within each interval
    all_rows = data[data['step_name'].str.contains(stepname)]
    averaged_data = pd.DataFrame()
    if not all_rows.empty:
        first_time = math.ceil(all_rows['approx_realtime_sec'].min() / sampling_interval) * sampling_interval   #round up to the nearest interval
        last_time = math.floor(all_rows['approx_realtime_sec'].max() / sampling_interval) * sampling_interval   #round down to the nearest interval
        t = first_time
        while t <= last_time:
            shunt_current = all_rows[(all_rows['approx_realtime_sec'] >= t) & (all_rows['approx_realtime_sec'] < t+sampling_interval) & (all_rows['identifier'] == '/htf.shunt.status')]['current'].mean()
            pack_current = all_rows[(all_rows['approx_realtime_sec'] >= t) & (all_rows['approx_realtime_sec'] < t+sampling_interval) & (all_rows['identifier'] == '/battery_monitor.droid_battery_telemetry')]['pack_current'].mean() * (-1)       #sign reversal eventually not necessary when droid sign convention updated
            shunt_voltage = all_rows[(all_rows['approx_realtime_sec'] >= t) & (all_rows['approx_realtime_sec'] < t+sampling_interval) & (all_rows['identifier'] == '/htf.shunt.status')]['voltage'].mean()
            pack_voltage = all_rows[(all_rows['approx_realtime_sec'] >= t) & (all_rows['approx_realtime_sec'] < t+sampling_interval) & (all_rows['identifier'] == '/battery_monitor.droid_battery_telemetry')]['pack_voltage'].mean()
            amp_s = all_rows[(all_rows['approx_realtime_sec'] >= t) & (all_rows['approx_realtime_sec'] < t+sampling_interval) & (all_rows['identifier'] == '/htf.shunt.status')]['current_counter'].mean()
            sys_current = all_rows[(all_rows['approx_realtime_sec'] >= t) & (all_rows['approx_realtime_sec'] < t+sampling_interval) & (all_rows['identifier'] == '/battery_monitor.droid_battery_telemetry')]['sys_current'].mean()
            new_row = {'time': [t], 'amp-s': [amp_s], 'shunt_current': [shunt_current], 'pack_current': [pack_current], 'pack_voltage': [pack_voltage], 'shunt_voltage': [shunt_voltage], 'sys_current': [sys_current]}
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


def generate_plots(data):
    #Suppress logs from specific modules that are loud and not useful.
    try:
        suppress_logger = logging.getLogger('matplotlib')
        suppress_logger.setLevel(logging.CRITICAL)
        suppress_logger = logging.getLogger('PIL.PngImagePlugin')
        suppress_logger.setLevel(logging.CRITICAL)
    except Exception as e:
        pass

    plt.figure()
    ax1 = plt.subplot(411)
    ax1.plot(data['time_offset'].to_numpy(), data['pack_voltage'].to_numpy(), label='Pack Voltage')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Pack Voltage')
    ax2 = plt.subplot(412, sharex=ax1)
    ax2.plot(data['time_offset'].to_numpy(), data['pack_current'].to_numpy(), label='Pack Current')
    ax2.set_ylabel('Current (A)')
    ax2.set_title('Pack Current')
    ax3 = plt.subplot(413, sharex=ax1)
    ax3.plot(data['time_offset'].to_numpy(), data['brick_heater_temperature[thermistor_brick1]'].to_numpy(), label='Cell 1')
    ax3.plot(data['time_offset'].to_numpy(), data['brick_heater_temperature[thermistor_brick3_afe]'].to_numpy(), label='Cell 3')
    ax3.plot(data['time_offset'].to_numpy(), data['brick_heater_temperature[thermistor_brick5]'].to_numpy(), label='Cell 5')
    ax3.legend(fontsize='small')
    ax3.set_ylabel('Temperature (C)')
    ax3.set_title('Cell Temperatures')
    ax4 = plt.subplot(414, sharex=ax1)
    for i in range(0, 5):
        ax4.plot(data['time_offset'], data[f'brick_voltage[{i}]'], label=f'Cell {i+1}')
    ax4.set_ylabel('Voltage (V)')
    ax4.set_title('Cell Voltages')
    ax4.legend(fontsize='x-small')
    plt.xlabel('Time (s)')
    plt.subplots_adjust(wspace=0, hspace=0.3)
    plt.gcf().set_size_inches(10, 13)
    plt.savefig(plot_file_path)


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