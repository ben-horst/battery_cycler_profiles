import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt

def open_csv_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select a CSV file"
    )
    return file_path

def plot_heater_data():
    if 'current' not in data.columns:
        raise ValueError("The CSV file does not contain a 'current' column")
    time = data['monotonic_sec'] - data['monotonic_sec'][0]
    plt.plot(time, data['current'])
    plt.title('Heater Overtemp Test')
    plt.xlabel('Time (s)')
    plt.ylabel('Heater Current (A)')
    plt.show()

if __name__ == "__main__":
    csv_file_path = open_csv_file()
    if not csv_file_path:
        raise ValueError("No file selected")
    data = pd.read_csv(csv_file_path)
    plot_heater_data()
