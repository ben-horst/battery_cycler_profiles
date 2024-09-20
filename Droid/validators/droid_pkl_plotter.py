import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

pd.options.mode.copy_on_write = True

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
        for i in range(0, 5):
            plt.plot(battery_data['approx_realtime_sec'], battery_data[f'brick_voltage[{i}]'], label=f'Cell {i}')
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

def plot_vsys_error():
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
        vsys_data = data['vsys_data']
        plt.plot(vsys_data['time'], vsys_data['vsys_ideal_current'], label='vsys_ideal_current')
        plt.plot(vsys_data['time'], vsys_data['sys_current'], label='sys_current')
        plt.plot(vsys_data['time'], vsys_data['vsys_current_error'], label='vsys_current_error')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.legend()
        plt.show()


#plot_all_cells()
#plot_current_error()
plot_vsys_error()

