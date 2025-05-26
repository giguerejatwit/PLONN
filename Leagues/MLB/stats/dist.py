"""
I want to see a distribution from Actual-Predicted Ks
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_excel('Leagues/MLB/data/MLB2025.xlsx', sheet_name='MLP')

error = data['Actual - Pred K'].dropna()
data = data.dropna(subset=['Actual K'], axis=0)

std = np.std(error)
print(f'Standard Deviation: {std}')

plt.figure(figsize=(10, 6))
sns.kdeplot(error, color='blue', fill=True)
mean_error = error.mean()
plt.axvline(mean_error, color='blue', linestyle='--', label='Mean')
plt.axvline(mean_error + std, color='red', linestyle='--', label='+1 STD')
plt.axvline(mean_error + (2*std), color='m', linestyle='--', label='+2 STD')
plt.axvline(mean_error - std, color='red', linestyle='--', label='-1 STD')
plt.axvline(mean_error - (2*std), color='m', linestyle='--', label='-2 STD')
# plt.legend()

plt.title('Distribution of Actual - Predicted Ks')
plt.xlabel('Error (Actual Ks - Predicted Ks)')
plt.ylabel('Density')

# plt.show()
print(f"Standard Deviation: {std}")

print(error)

error_conf_df = data[["Actual - Pred K", "Confidence"]].dropna()

# Bin confidence values
bins = [0, 0.5, 1, 1.5, 2, 2.5, float('inf')]
labels = ["<0.5", "0.5–1.0", "1.0–1.5", "1.5–2.0", "2.0–2.5", "2.5+"]
error_conf_df["Confidence Bin"] = pd.cut(error_conf_df["Confidence"], bins=bins, labels=labels, include_lowest=True)

# Set up plot
plt.figure(figsize=(12, 6))
sns.boxplot(x="Confidence Bin", y="Actual - Pred K", data=error_conf_df, palette="coolwarm")
plt.axhline(0, linestyle="--", color="black")
plt.title("Distribution of Prediction Error by Confidence Level")
plt.xlabel("Confidence Bin")
plt.ylabel("Error (Actual - Predicted Ks)")
plt.tight_layout()
# plt.show()


# Under fades
unders = data[data['O/U'] == 'U']
unders = unders.dropna(subset=['Actual K', 'Confidence', 'W/L'], axis=0)
print('Number of Total Unders:', len(unders))

#calc W/L
unders_win = len(unders[unders['W/L'] == 'W'])
unders_loss = len(unders[unders['W/L'] == 'L'])
print(f"Unders Win: {unders_win}")
print(f"Unders Loss: {unders_loss}")
win_loss = unders_win / (unders_win + unders_loss)
print(f"Win/Loss %: {win_loss}")

#now calc unders withing golden range 0.5 < x < 1.5
golden_range = unders[(unders['Confidence'] > 0.75) & (unders['Confidence'] < 1.5)]
print('Numbers of Unders in Golden Range:', len(golden_range))

golden_range_w = golden_range[golden_range['W/L'] == 'W']
print('Number of Unders in Golden Range that Won:', len(golden_range_w))
golden_range_l = golden_range[golden_range['W/L'] == 'L']
print('Number of Unders in Golden Range that Lost:', len(golden_range_l))
print('Win/Loss % in Golden Range:', len(golden_range_w) / (len(golden_range_w) + len(golden_range_l)))


# over fades
overs = data[data['O/U'] == 'O']
overs = overs.dropna(subset=['Actual K', 'Confidence', 'W/L'], axis=0)
print('Number of Total Overs:', len(overs))

#calc W/L
overs_win = len(overs[overs['W/L'] == 'W'])
overs_loss = len(overs[overs['W/L'] == 'L'])
print(f"Unders Win: {overs_win}")
print(f"Unders Loss: {overs_loss}")
win_loss = overs_win / (overs_win + overs_loss)
print(f"Win/Loss %: {win_loss}")

#now calc unders withing golden range 0.5 < x < 1.5
golden_range = overs[((overs['Confidence'] > 0) & (overs['Confidence'] < .25)) | ((overs['Confidence'] > 1.25) & (overs['Confidence'] < 2.25))]
print('Numbers of overs in Golden Range:', len(golden_range))

golden_range_w = golden_range[golden_range['W/L'] == 'W']
print('Number of overs in Golden Range that Won:', len(golden_range_w))
golden_range_l = golden_range[golden_range['W/L'] == 'L']
print('Number of overs in Golden Range that Lost:', len(golden_range_l))
print('Win/Loss % in Golden Range:', len(golden_range_w) / (len(golden_range_w) + len(golden_range_l)))


import matplotlib.pyplot as plt
dk_line = [3.5, 4.5, 5.5, 6.5]
# Filter for W's only and drop missing confidence
under_wins = unders[(unders['W/L'] == 'W')].dropna(subset=['Confidence'])
over_wins = overs[(overs['W/L'] == 'W')].dropna(subset=['Confidence'])
under_losses = unders[(unders['W/L'] == 'L')].dropna(subset=['Confidence'])
over_losses = overs[(overs['W/L'] == 'L')].dropna(subset=['Confidence'])

# Define confidence bins
bins = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5]

# Plot histogram

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

# Top-left: Over Wins
axes[0, 0].hist(over_wins['Confidence'], bins=bins, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Over Wins')
axes[0, 0].grid(True)

# Top-right: Over Losses
axes[0, 1].hist(over_losses['Confidence'], bins=bins, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Over Losses')
axes[0, 1].grid(True)

# Bottom-left: Under Wins
axes[1, 0].hist(under_wins['Confidence'], bins=bins, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Under Wins')
axes[1, 0].grid(True)

# Bottom-right: Under Losses
axes[1, 1].hist(under_losses['Confidence'], bins=bins, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Under Losses')
axes[1, 1].grid(True)

# Common labels
for ax in axes.flat:
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')

plt.suptitle('Confidence Distribution by Bet Outcome', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# 2. Ensure Confidence is numeric and drop any missing W/L or Confidence
data['Confidence'] = pd.to_numeric(data['Confidence'], errors='coerce')
df = data.dropna(subset=['W/L', 'Confidence'])

# 3. Define your bins and labels
bins   = [-float('inf'), 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, float('inf')]
labels = [
    '<0.25', '0.25–0.5', '0.5–0.75', '0.75–1.0', 
    '1.0–1.25', '1.25–1.5', '1.5–1.75', '1.75–2.0', 
    '2.0–2.25', '2.25–2.5', '2.5+'
]
df['Conf Bin'] = pd.cut(df['Confidence'], bins=bins, labels=labels)

# 4. Compute win percentage per bind
win_pct = (
    df
    .groupby('Conf Bin')['W/L']
    .apply(lambda x: (x == 'W').sum() / x.size * 100)
    .reset_index(name='Win %')
)
# Compute sample size
sample_sizes = df.groupby('Conf Bin').size().reset_index(name='Sample Size')

# Merge and display
summary = win_pct.merge(sample_sizes, on='Conf Bin')

print(summary)

# 3. Split Overs vs Unders
overs  = df[df['O/U'] == 'O']
unders = df[df['O/U'] == 'U']

# 4. Helper to compute Win % and Sample Size
def summarize(group):
    N    = len(group)
    win  = (group['W/L'] == 'W').sum()
    return pd.Series({'Win %': win / N * 100, 'N': N})

# 5. Build separate summaries
over_summary  = overs.groupby('Conf Bin').apply(summarize).reset_index()
under_summary = unders.groupby('Conf Bin').apply(summarize).reset_index()

# 6. Merge side by side
summary = pd.merge(
    over_summary, 
    under_summary, 
    on='Conf Bin', 
    how='outer',
    suffixes=('_Over', '_Under')
)

print(summary)
# fade the O/U for 3.5, 4.5, 5.5, 6.5
#TODO

hanger_w = data[data['DK Line -0.5'] == 'W']
hanger_l = data[data['DK Line -0.5'] == 'L']

print(f'Hanger Wins: {len(hanger_w)}')
print(f'Hanger Losses: {len(hanger_l)}')
print(f'Hanger Win % : {len(hanger_w) / (len(hanger_w) + len(hanger_l))}')

plt.show()