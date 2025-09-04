# src/inspect_sessions.py
from pathlib import Path
import pandas as pd
p = Path("data/processed")/ "sessions.csv"
df = pd.read_csv(p, low_memory=False)
print(df["label"].value_counts())

# Check session_id uniqueness and provide informative output
unique_sessions = df["session_id"].nunique()
total_sessions = len(df)
print(f"\nSession ID statistics:")
print(f"Total rows: {total_sessions}")
print(f"Unique session_ids: {unique_sessions}")
print(f"Duplicate session_ids: {total_sessions - unique_sessions}")

if not df["session_id"].is_unique:
    print("\nWARNING: session_id is not unique!")
    # Show some examples of duplicates
    duplicates = df[df.duplicated(subset=['session_id'], keep=False)].sort_values('session_id')
    print(f"Example duplicate session_ids:")
    print(duplicates[['session_id']].head(10))
else:
    print("✓ session_id is unique")
