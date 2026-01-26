import os
import matplotlib

# Always use non-interactive backend to work in headless/dev-container environments.
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Plot data from a CSV file.')
parser.add_argument('arg', help='Path to the CSV file to plot.')
args = parser.parse_args()
arg = args.arg

csv_path = arg
df = pd.read_csv(csv_path)

time_col = df.columns[0]

plt.figure(figsize=(10, 6))
for col in df.columns:
    if col == time_col:
        continue
    plt.plot(df[time_col], df[col], label=col)

plt.xlabel(time_col)
plt.ylabel('Value')
plt.title('Plot of Selected Columns')
plt.legend()
plt.grid(True)
plt.tight_layout()

out = os.path.splitext(os.path.basename(csv_path))[0] + ".png"
plt.savefig(out, dpi=150)
print(f"Saved figure to {out}")