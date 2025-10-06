import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Runs sheet
runs_df = pd.read_excel('Leagues/MLB/data/MLB2025.xlsx', sheet_name='Runs')






# Select relevant columns and clean
runs_df = runs_df[['Date', 'DK Diff', 'O/U', 'Total_Actual', 'DK Line']].copy()
runs_df.columns = ['Date', 'DK_Diff', 'OU', 'Total_Actual', 'DK_Line']
runs_df.dropna(subset=['DK_Diff', 'OU', 'Total_Actual', 'DK_Line'], inplace=True)

# Determine if prediction was correct
runs_df['Correct'] = np.where(
    ((runs_df['OU'] == 'O') & (runs_df['Total_Actual'] > runs_df['DK_Line'])) |
    ((runs_df['OU'] == 'U') & (runs_df['Total_Actual'] < runs_df['DK_Line'])),
    1, 0
)

# Bin DK Diff values from 0 to 6 in 0.5 intervals
bins = np.arange(0, 6.5, 0.5)
labels = [f"{i:.1f}-{i+0.5:.1f}" for i in bins[:-1]]
runs_df['DK_Diff_Bin'] = pd.cut(runs_df['DK_Diff'], bins=bins, labels=labels, include_lowest=True)

# Convert dates and sort
runs_df['Date'] = pd.to_datetime(runs_df['Date'])
runs_df.sort_values(by='Date', inplace=True)

# Split Overs and Unders
overs_df = runs_df[runs_df['OU'] == 'O'].copy()
unders_df = runs_df[runs_df['OU'] == 'U'].copy()

# Function to calculate cumulative accuracy
def cumulative_accuracy_by_bin(df):
    bin_accuracy = []
    for bin_label in df['DK_Diff_Bin'].unique():
        bin_df = df[df['DK_Diff_Bin'] == bin_label]
        if not bin_df.empty:
            grouped = bin_df.groupby('Date')['Correct'].agg(['sum', 'count']).sort_index()
            grouped['Cumulative_Correct'] = grouped['sum'].cumsum()
            grouped['Cumulative_Count'] = grouped['count'].cumsum()
            grouped['Cumulative_Accuracy'] = grouped['Cumulative_Correct'] / grouped['Cumulative_Count']
            grouped['Bin'] = bin_label
            bin_accuracy.append(grouped[['Cumulative_Accuracy', 'Bin']])
    return pd.concat(bin_accuracy).reset_index()

# Calculate cumulative accuracy
overs_acc_df = cumulative_accuracy_by_bin(overs_df)
unders_acc_df = cumulative_accuracy_by_bin(unders_df)

# Plot function

 # Plot with subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Overs
for bin_label in overs_acc_df['Bin'].unique():
    bin_df = overs_acc_df[overs_acc_df['Bin'] == bin_label]
    axes[0].plot(bin_df['Date'], bin_df['Cumulative_Accuracy'], label=bin_label)
axes[0].set_title('Cumulative Accuracy Over Time (Overs)')
axes[0].set_ylabel('Cumulative Accuracy')
axes[0].legend(title='DK Diff Bin')
axes[0].grid(True)

# Unders
for bin_label in unders_acc_df['Bin'].unique():
    bin_df = unders_acc_df[unders_acc_df['Bin'] == bin_label]
    axes[1].plot(bin_df['Date'], bin_df['Cumulative_Accuracy'], label=bin_label)
axes[1].set_title('Cumulative Accuracy Over Time (Unders)')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Cumulative Accuracy')
axes[1].legend(title='DK Diff Bin')
axes[1].grid(True)

plt.tight_layout()
plt.show()


