import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.mode.copy_on_write = True



with open('battery_telemetry_data.pkl', 'rb') as file:
    battery_data = pickle.load(file)
    battery_data = battery_data['all_battery_data']
    plt.plot(battery_data['approx_realtime_sec'], battery_data['brick_voltage[0]'], label='Brick 0')
    plt.plot(battery_data['approx_realtime_sec'], battery_data['brick_voltage[1]'], label='Brick 1')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.show()

with open('validation_data.pkl', 'rb') as file:
    validation_data = pickle.load(file)
    current_sampling_data = validation_data['current_sampling_data']
    #plt.plot(current_sampling_data['time'], current_sampling_data['current_error'], label='Current Error')
