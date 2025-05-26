import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('Leagues/MLB/data/MLB2025.xlsx', sheet_name='MLP')

# 2. Filter out rows without a Win/Loss result and map to profit
df = df.dropna(subset=['W/L'])
df['profit'] = df['W/L'].map({'W': 1, 'L': -1})

# 3. Aggregate by date and compute cumulative sum
df['Date'] = pd.to_datetime(df['Date']).dt.date
daily = df.groupby('Date')['profit'].sum().reset_index()
print('Daily Gain: ', daily)
daily['cumulative'] = daily['profit'].cumsum()
print(daily['cumulative'])

# 4. Plot scatter + line of cumulative gain
plt.figure(figsize=(10, 6))
plt.scatter(daily['Date'], daily['cumulative'])
plt.plot(daily['Date'], daily['cumulative'])
plt.xlabel('Date')
plt.ylabel('Cumulative Gain ($)')
plt.title('Cumulative Gain Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()

df = df.dropna(subset=['DK Line -0.5'])

# 3. Map W/L to profit: +1.33 for a win, -1 for a loss
df['profit_adj'] = df['DK Line -0.5'].map({'W': 0.33, 'L': -1})

# 4. Compute overall stats
total_bets = len(df)
wins = (df['DK Line -0.5'] == 'W').sum()
win_rate = wins / total_bets * 100
ev_per_bet = win_rate/100 * 0.33 + (1 - win_rate/100) * (-1)

print(f"Total Bets: {total_bets}")
print(f"Wins: {wins}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Expected Value per Bet: {ev_per_bet:.4f}")

# 5. Aggregate by date and compute cumulative gain
df['Date'] = pd.to_datetime(df['Date']).dt.date
daily = df.groupby('Date')['profit_adj'].sum().reset_index()
daily['cumulative'] = daily['profit_adj'].cumsum()
final_gain = daily['cumulative'].iloc[-1]
print(f"Final Cumulative Gain: ${final_gain:.2f}")

# 6. Plot scatter + line of adjusted cumulative gain
plt.figure(figsize=(10, 6))
plt.scatter(daily['Date'], daily['cumulative'], label='Daily Cumulative')
plt.plot(daily['Date'], daily['cumulative'])
plt.xlabel('Date')
plt.ylabel('Cumulative Gain ($)')
plt.title('Adjusted Cumulative Gain Over Time (DK Line -0.5)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
# plt.show()

# 2. Ensure Confidence is numeric and drop rows missing essential data
df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
df = df.dropna(subset=['W/L', 'Confidence', 'O/U'])

# 3. Define strategy filters
und_filter = (
    (df['O/U'] == 'U') & (
        ((df['Confidence'] >= 0.5) & (df['Confidence'] < 1.0)) |
        ((df['Confidence'] >= 1.25) & (df['Confidence'] < 1.5)) |
        ((df['Confidence'] >= 2.0) & (df['Confidence'] <= 2.25)) |
        (df['Confidence'] > 2.25)
    )
)

#    Overs: confidence in [0,0.25), [1.0,1.75), [1.25,1.5), [2.0,2.25], (>2.25)
over_filter = (
    (df['O/U'] == 'O') & (
        (df['Confidence'] < 0.25) |
        ((df['Confidence'] >= 1.0) & (df['Confidence'] < 1.75)) |
        ((df['Confidence'] >= 1.25) & (df['Confidence'] < 1.5)) |
        ((df['Confidence'] >= 2.0) & (df['Confidence'] <= 2.25)) |
        (df['Confidence'] > 2.25)
    )
)

# 4. Select only those bets
df_strat = df[und_filter | over_filter].copy()

# 5. Map W/L to profit: +$0.80 for win, –$1 for loss
df_strat['profit'] = df_strat['W/L'].map({'W': 1, 'L': -1})

# 6. Compute and print strategy-wide stats
total_bets = len(df_strat)
wins       = (df_strat['W/L'] == 'W').sum()
win_rate   = wins / total_bets * 100
ev_per_bet = win_rate/100 * 0.80 + (1 - win_rate/100) * (-1)

print(f"Strategy Bets: {total_bets}")
print(f"Wins: {wins}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Expected Value per Bet: ${ev_per_bet:.2f}")

# 7. Aggregate by date and compute cumulative gain
df_strat['Date'] = pd.to_datetime(df_strat['Date']).dt.date
daily = df_strat.groupby('Date')['profit'].sum().reset_index(name='daily_profit')
daily['cumulative_profit'] = daily['daily_profit'].cumsum()

# 8. Print final cumulative profit
print(f"Final Cumulative Profit: ${daily['cumulative_profit'].iloc[-1]:.2f}")

# 9. Plot the result
plt.figure(figsize=(10, 6))
plt.scatter(daily['Date'], daily['cumulative_profit'], label='Strategy Cumulative')
plt.plot(daily['Date'], daily['cumulative_profit'])
plt.xlabel('Date')
plt.ylabel('Cumulative Profit ($)')
plt.title('Cumulative Profit Over Time (Filtered Strategy)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()