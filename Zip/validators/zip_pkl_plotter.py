import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

pd.options.mode.copy_on_write = True

def plot_pack_voltage_and_current():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        filetypes=[("Pickle files", "*.pkl")],
        title="choose battery telemetry pkl file"
    )

    if not file_path:
        raise ValueError("No file selected")

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        battery_data = data['all_battery_data']
        plt.figure()
        plt.plot(battery_data['approx_realtime_sec'].to_numpy(), battery_data['pack_voltage'].to_numpy(), label='Pack Voltage')
        plt.plot(battery_data['approx_realtime_sec'].to_numpy(), battery_data['pack_current'].to_numpy(), label='Pack Current')
        plt.ylabel('Voltage (V) / Current (A)')
        plt.title('Pack Voltage & Current')
        plt.legend()
        plt.show()

def plot_all_cells():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        filetypes=[("Pickle files", "*.pkl")],
        title="choose battery telemetry pkl file"
    )

    if not file_path:
        raise ValueError("No file selected")

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        battery_data = data['all_battery_data']
        for i in range(0, 14):
            plt.plot(battery_data['approx_realtime_sec'], battery_data[f'brick_voltage[{i}]'], label=f'Brick {i}')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.show()

def plot_current_error():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        filetypes=[("Pickle files", "*.pkl")],
        title="choose validation data pkl file"
    )

    if not file_path:
        raise ValueError("No file selected")

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        current_sampling_data = data['current_sampling_data']
        plt.plot(current_sampling_data['time'], current_sampling_data['pack_current_smoothed'], label='Pack Current')
        plt.plot(current_sampling_data['time'], current_sampling_data['shunt_current_smoothed'], label='Shunt Current')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.show()


#plot_pack_voltage_and_current()
plot_all_cells()
#plot_current_error()

