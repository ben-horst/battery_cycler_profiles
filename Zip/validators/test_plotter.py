import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.mode.copy_on_write = True



with open('battery_telemetry_data.pkl', 'rb') as file:
    battery_data = pickle.load(file)
    battery_data = battery_data['all_battery_data']
    plt.plot(battery_data['time_offset'], battery_data['pack_voltage'], label='Battery Data')
    plt.show()

with open('validation_data.pkl', 'rb') as file:
    validation_data = pickle.load(file)
    current_sampling_data = validation_data['current_sampling_data']
    plt.plot(current_sampling_data['time'], current_sampling_data['current_error'], label='Current Error')
