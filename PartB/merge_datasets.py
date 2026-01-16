import pandas as pd

# Load the datasets
df_beacon_features = pd.read_csv("participant_features.csv")
df_clinical = pd.read_csv("../Preprocessed Data/clinical_dataset_preprocessed.csv")

# Columns to keep from the unfiltered dataset
cols_to_keep = [
    "part_id",
    "gait_speed_4m",
    "gait_get_up",
    "activity_regular",
    "age",
    "bmi_score",
    "raise_chair_time"
]

# Subset the unfiltered dataset
df_clinical_subset = df_clinical[cols_to_keep]

# Merge the two datasets on part_id
merged_df = df_beacon_features.merge(
    df_clinical_subset,
    on="part_id",
    how="left"
)

# Inspect result
print(merged_df.head())
