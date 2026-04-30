# src/prepare_schools_panel.py
import os
import glob
import pandas as pd

def load_all_schools(in_dir="data/schools", out_csv="data/schools_panel.csv"):
    paths = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {in_dir}")

    frames = []
    for p in paths:
        school_id = os.path.splitext(os.path.basename(p))[0]  # e.g., "S1"
        df = pd.read_csv(p)
        if "timestamp" not in df.columns or "kWh" not in df.columns:
            raise RuntimeError(f"{p} missing required columns. Found: {list(df.columns)}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df["school_id"] = school_id
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)

    # Keep a consistent target name
    panel = panel.rename(columns={"kWh": "y"})

    # Make sure categorical columns are strings (for one-hot encoding later)
    for c in ["holidayType", "season", "term", "dayType", "schoolDay"]:
        if c in panel.columns:
            panel[c] = panel[c].astype(str)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    panel.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(panel)} rows, schools: {panel['school_id'].nunique()}")

if __name__ == "__main__":
    load_all_schools()
