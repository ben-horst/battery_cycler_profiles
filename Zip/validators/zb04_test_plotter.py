import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.mode.copy_on_write = True

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

def generate_simlified_ocv_curve(discharge_data, charge_data):
    #builds an OCV curve from the discharge and charge data
    ocv_curve = pd.DataFrame()
    #TODO make discretized soc value generator

with open('battery_data.pkl', 'rb') as file:
    data = pickle.load(file)

discharge_data = data['discharge_data']
charge_data = data['charge_data']
small_pulse_data = data['small_pulse_data']
large_pulse_data = data['large_pulse_data']

large_pulse_data = pd.DataFrame(large_pulse_data)
print(large_pulse_data)

dcir_4C_1s, _, _ = generate_dcir_durve(large_pulse_data, 1)
dcir_4C_10s, _, _ = generate_dcir_durve(large_pulse_data, 10)

discharge_data = pd.DataFrame(discharge_data)
charge_data = pd.DataFrame(charge_data)


plt.figure()
plt.plot(discharge_data['soc'], discharge_data['voltage'], label='discharge')
plt.plot(charge_data['soc'], charge_data['voltage'], label='charge')
plt.xlabel('soc')
plt.ylabel('voltage')
plt.legend()
plt.title('C/5 Charge/Discharge')

plt.figure()
plt.plot(dcir_4C_1s['consumed_ah'], dcir_4C_1s['dcir'], label='4C/1s')
plt.plot(dcir_4C_10s['consumed_ah'], dcir_4C_10s['dcir'], label='4C/10s')
plt.gca().invert_xaxis()
plt.legend()
plt.xlabel('consumed Ah')
plt.ylabel('resistance')
plt.title('DCIR')


plt.show()