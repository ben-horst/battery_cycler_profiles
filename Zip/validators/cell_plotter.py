import pickle
import matplotlib.pyplot as plt

# Load the pickle file
with open('validation_data.pkl', 'rb') as file:
    data = pickle.load(file)

all_battery_data = data['all_battery_data']
# Plot every column that includes 'brick'
for column in all_battery_data.columns:
    if 'hw_ov_brick_bitfield' in column:
        plt.plot(all_battery_data[column], label=column, marker='o')

# Add labels and legend
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()