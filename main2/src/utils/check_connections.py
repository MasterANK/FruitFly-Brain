import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

connections = pd.read_csv(
    DATA_DIR / "male_cns_navigation_connections.csv"
)

print("Number of rows:", len(connections))
print("\nColumns:")
print(connections.columns.tolist())

print("\nFirst 10 rows:")
print(connections.head(10).to_string())