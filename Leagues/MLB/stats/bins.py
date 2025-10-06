import pandas as pd

df = pd.read_excel('Leagues/MLB/data/MLB2025N.xlsx')
df.dropna(subset=['DK Diff', 'O/U', 'W/L'])

# W/L calculation function
def calculate_wl(df, label):
    df = df.copy()
    df["W"] = df["W/L"] == "W"
    df["L"] = df["W/L"] == "L"
    df["V"] = df["W/L"] == "V"
    
    summary = df.groupby("Bin")[["W", "L", "V"]].sum()
    summary["Total"] = summary[["W", "L", "V"]].sum(axis=1)
    summary["Win %"] = (summary["W"] / summary["Total"] * 100).round(2)
    summary = summary.rename(columns=lambda col: f"{label}-{col}")
    return summary

def accuracy_bins() -> pd.DataFrame:
# Create bins
    bins = [x * 0.5 for x in range(0, 13)]

    labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins) - 1)]
    df["Bin"] = pd.cut(df["DK Diff"], bins=bins, labels=labels, right=False)

    # Split into Over and Under
    overs = df[df["O/U"] == "O"]
    unders = df[df["O/U"] == "U"]

    # Apply
    over_summary = calculate_wl(overs, "O")
    under_summary = calculate_wl(unders, "U")

    # Merge for easy comparison
    combined = pd.concat([over_summary, under_summary], axis=1)
    print(combined)
    return combined

if __name__ == "__main__":
    accuracy_bins()
    

