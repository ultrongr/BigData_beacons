import pandas as pd
import numpy as np
from datetime import time

# -----------------------------
# Parameters
# -----------------------------
NIGHT_START = time(22, 0, 0)
NIGHT_END = time(6, 0, 0)

BEDROOM_LABEL = "Bedroom"
BATHROOM_LABEL = "Bathroom"
OUTDOOR_LABEL = "Outdoor"
OFFICE_LABEL = "Office"
KITCHEN_LABEL = "Kitchen"

MAX_GAP_SECONDS = 24 * 3600  # 24 hours

INPUT_CSV = "../Preprocessed Data/beacons_dataset_preprocessed.csv"
OUTPUT_CSV = "participant_features.csv"


# -----------------------------
# Helper
# -----------------------------
def is_night(t):
    return (t >= NIGHT_START) or (t < NIGHT_END)


# -----------------------------
# Load & prepare
# -----------------------------
df = pd.read_csv(INPUT_CSV)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["part_id", "timestamp"]).reset_index(drop=True)

df["date"] = df["timestamp"].dt.date
df["time"] = df["timestamp"].dt.time

df["next_timestamp"] = df.groupby("part_id")["timestamp"].shift(-1)
df["duration"] = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds()

# Drop last row per participant
df = df[df["duration"].notna()]

# Ignore large gaps
df = df[df["duration"] <= MAX_GAP_SECONDS]

# Room changes
df["prev_room"] = df.groupby("part_id")["room"].shift(1)
df["room_change"] = (df["room"] != df["prev_room"]).astype(int)

# Night flag
df["is_night"] = df["time"].apply(is_night)


# -----------------------------
# Feature extraction
# -----------------------------
rows = []

for pid, g in df.groupby("part_id"):
    total_time = g["duration"].sum()

    if total_time == 0:
        continue

    # --- Time-based (raw seconds) ---
    bedroom_night = g.loc[
        (g["room"] == BEDROOM_LABEL) & (g["is_night"]), "duration"
    ].sum()

    bathroom_night = g.loc[
        (g["room"] == BATHROOM_LABEL) & (g["is_night"]), "duration"
    ].sum()

    outdoor_time = g.loc[g["room"] == OUTDOOR_LABEL, "duration"].sum()
    office_time = g.loc[g["room"] == OFFICE_LABEL, "duration"].sum()
    kitchen_time = g.loc[g["room"] == KITCHEN_LABEL, "duration"].sum()

    # --- Normalize to % of total ---
    pct_bedroom_night = 100 * bedroom_night / total_time
    pct_bathroom_night = 100 * bathroom_night / total_time
    pct_outdoor = 100 * outdoor_time / total_time
    pct_office = 100 * office_time / total_time
    pct_kitchen = 100 * kitchen_time / total_time

    # --- Room changes ---
    room_changes_total = g["room_change"].sum()
    room_changes_night = g.loc[g["is_night"], "room_change"].sum()

    # --- Avg rooms per day ---
    avg_rooms_per_day = g.groupby("date")["room"].nunique().mean()

    # --- % time per room ---
    pct_time_per_room = (
        g.groupby("room")["duration"].sum() / total_time * 100
    )

    row = {
        "part_id": pid,
        "pct_time_bedroom_night": pct_bedroom_night,
        "pct_time_bathroom_night": pct_bathroom_night,
        "pct_time_outdoor": pct_outdoor,
        "pct_time_office": pct_office,
        "pct_time_kitchen": pct_kitchen,
        "room_changes_night": room_changes_night,
        "room_changes_total": room_changes_total,
        "avg_rooms_per_day": avg_rooms_per_day,
    }

    for room, pct in pct_time_per_room.items():
        row[f"pct_time_{room.lower()}"] = pct

    rows.append(row)


# -----------------------------
# Output
# -----------------------------
out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved normalized features to {OUTPUT_CSV}")
