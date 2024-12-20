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

def plot_cell_delta_v():
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
        highest_brick_voltage = battery_data.filter(like='brick_voltage[').max(axis=1)
        lowest_brick_voltage = battery_data.filter(like='brick_voltage[').min(axis=1)
        delta_v = highest_brick_voltage - lowest_brick_voltage
        plt.plot(battery_data['approx_realtime_sec'], delta_v)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Max Delta Voltage (V)')
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
        plt.plot(current_sampling_data['time'], current_sampling_data['current_error'], label='Current Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.legend()
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

def plot_delta_from_validation_data():
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
        brick_delta_data = data['brick_delta_data']
        plt.plot(brick_delta_data['time'], brick_delta_data['brick_delta'], label='brick_deltas (blips removed)')
        plt.plot(brick_delta_data['time'], brick_delta_data['brick_delta_raw'], label='brick_deltas (raw)', linestyle='None', marker='o', markersize=2)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.show()


#plot_all_cells()
#plot_cell_delta_v()
#plot_current_error()
#plot_vsys_error()
plot_delta_from_validation_data()

