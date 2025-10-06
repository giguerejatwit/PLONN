import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Load your Excel file
file_path = "Leagues/NBA/data/nba.xlsx"
df = pd.read_excel(file_path, sheet_name='Full')

# Clean column names and date field
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Filter only dates within the NBA 2024–25 season
# Filter only dates within the NBA 2024–25 season
df = df[(df['date'] >= pd.Timestamp('2024-10-01')) & (df['date'] <= pd.Timestamp('2025-07-01'))]


# Only keep rows where W/L is either 'W' or 'L'
df = df[df['W/L'].isin(['W', 'L'])]

# Define payout structure (base payouts)
payouts = {
    1: 9,
    2: 29,
    3: 60,
    4: 100,
    5: 200,
    6: 400,
    7: 800,
    8: 1600,
    9: 3300,
    10: 6600,
    11: 13200,
    12: 26000,
    13: 52000,
    14: 104000,
    15: 200000
}

# Define boost structure
boosts = {
    2: 0.10,
    3: 0.20,
    4: 0.30,
    5: 0.40,
    6: 0.50,
    7: 0.60,
    8: 0.70,
    9: 0.80,
    10: 0.90
    # 11+ handled separately
}

# Group data by date
grouped = df.groupby('date')

# Calculate daily and cumulative earnings
results = []
for date, group in grouped:
    total_games = len(group)
    wins = group['W/L'].isin(['W', 'V']).sum()
    losses = (group['W/L'] == 'L').sum()

    print(f'Date: {date.strftime("%Y-%m-%d")}, Wins: {wins}, Losses: {losses}, Games: {total_games}')

    if losses == 0 and wins == total_games:
        base_payout = payouts.get(total_games, 0)
        boost = boosts.get(total_games, 1.0 if total_games >= 11 else 0.0)
        payout = base_payout * (1 + boost)
    else:
        payout = -10

    results.append({
        'date': date,
        'games': total_games,
        'daily_earnings': payout
    })

# Create DataFrame and compute cumulative earnings
earnings_df = pd.DataFrame(results)
earnings_df['cumulative_earnings'] = earnings_df['daily_earnings'].cumsum()

# Print cumulative earnings
print(earnings_df[['date', 'daily_earnings', 'cumulative_earnings']])

# Plot cumulative earnings
plt.figure(figsize=(12, 6))
plt.plot(earnings_df['date'], earnings_df['cumulative_earnings'], marker='o', linestyle='-')
plt.title('Cumulative Earnings (Perfect Days + Boosted Payouts, -$10 on Any Loss)')
plt.xlabel('Date')
plt.ylabel('Cumulative Earnings ($)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compute per-game returns: +$0.90 profit if 'W' or 'V', -$1 if 'L'
df['individual_return'] = df['W/L'].apply(lambda x: 0.90 if x in ['W', 'V'] else -1)

# Group by date to compute daily return total from $1 per game
daily_returns = df.groupby('date')['individual_return'].sum().reset_index()
daily_returns.columns = ['date', 'daily_game_return']
daily_returns['cumulative_game_return'] = daily_returns['daily_game_return'].cumsum()

# Prepare data for regression (convert dates to ordinal for numeric regression)
X = daily_returns['date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y = daily_returns['cumulative_game_return'].values

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
regression_line = model.predict(X)
average_return = daily_returns['cumulative_game_return'].mean()
# Total net profit
net_profit = daily_returns['cumulative_game_return'].iloc[-1]

# Total number of games (1 bet per game)
total_bets = df.shape[0]

# Total amount wagered ($1 per game)
total_wagered = total_bets * 1

# ROI calculation
roi_percentage = (net_profit / total_wagered) * 100
roi_percentage

print(f"Total Bets: {total_bets}")
print(f"Total Amount Wagered: ${total_wagered:.2f}")
print(f"Net Profit: ${net_profit:.2f}")
print(f"Average Return: ${average_return:.2f}")
print(f"ROI: {roi_percentage:.2f}%")
# Plot cumulative return with average and regression line
plt.figure(figsize=(12, 6))
plt.plot(daily_returns['date'], y, marker='o', linestyle='-', label='Cumulative Return')
plt.plot(daily_returns['date'], regression_line, color='green', linestyle='--', label='Trend Line (Regression)')
plt.axhline(y=average_return, color='red', linestyle='--', label=f'Average: ${average_return:.2f}')
plt.title('Cumulative Return with Average & Regression Line')
plt.xlabel('Date')
plt.ylabel('Cumulative Return ($)')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()