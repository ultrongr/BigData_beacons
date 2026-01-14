import pandas as pd
import time
import numpy as np

input_file = "../Raw Data/clinical_dataset_commas.csv"
output_file = "../Preprocessed Data/clinical_dataset_preprocessed.csv"

def convert_to_nominal_values(_df):
    mappings = dict()
    mappings["fried"] = {"Non frail": 0, "Pre-frail": 1, "Frail": 2}
    mappings["gender"] = {"M": 0, "F": 1}
    mappings["hospitalization_three_years"] = {999:np.nan}
    mappings["ortho_hypotension"] = {"No": 0, "Yes": 1}
    mappings["vision"] = {"Sees poorly": 0, "Sees moderately": 1, "Sees well": 2}
    mappings["audition"] = {"Hears poorly": 0, "Hears moderately": 1, "Hears well": 2}
    mappings["weight_loss"] = {"No": 0, "Yes": 1}
    mappings["exhaustion_score"] = {999: np.nan}
    mappings["raise_chair_time"] = {999: np.nan}
    mappings["balance_single"] = {"<5 sec": 0, ">5 sec": 1, "test non realizable": np.nan}
    mappings["gait_get_up"] = {999: np.nan}
    mappings["gait_speed_4m"] = {999: np.nan}
    mappings["gait_optional_binary"] = {False: 0, True: 1}
    mappings["gait_speed_slower"] = {"No": 0, "Yes": 1, "Test not adequate": np.nan}
    mappings["grip_strength_abnormal"] = {"No": 0, "Yes": 1}
    mappings["low_physical_activity"] = {"No": 0, "Yes": 1}
    mappings["falls_one_year"] = {999: np.nan}
    mappings["fractures_three_years"] = {999: np.nan}
    mappings["bmi_score"] = {999.9: np.nan}
    mappings["bmi_body_fat"] = {999.0: np.nan}
    mappings["waist"] = {999.0: np.nan}
    mappings["lean_body_mass"] = {-391.964: np.nan}
    mappings["screening_score"] = {999: np.nan}
    mappings["cognitive_total_score"] = {999.0: np.nan}
    mappings["memory_complain"] = {"No": 0, "Yes": 1}
    mappings["sleep"] = {"No sleep problem": 0, "Occasional sleep problem": 1, "Permanent sleep problem": 2}
    mappings["mmse_total_score"] = {999: np.nan}
    mappings["depression_total_score"] = {999.0: np.nan}
    mappings["anxiety_perception"] = {999.0: np.nan}
    mappings["living_alone"] = {"Yes": 1, "No": 0}
    mappings["leisure_out"] = {"Yes": 1, "No": 0}
    mappings["leisure_club"] = {"Yes": 1, "No": 0}
    mappings["social_visits"] = {999.0: np.nan}
    mappings["social_calls"] = {999.0: np.nan}
    mappings["social_phone"] = {999.0: np.nan}
    mappings["social_skype"] = {999.0: np.nan}
    mappings["social_text"] = {999.0: np.nan}
    mappings["house_suitable_participant"] = {"Yes": 1, "No": 0}
    mappings["house_suitable_professional"] = {"Yes": 1, "No": 0}
    mappings["stairs_number"] = {999.0: np.nan}
    mappings["life_quality"]= {999.0: np.nan}
    mappings["health_rate"] = {"1 - Very bad": 1,"2 - Bad": 2,"3 - Medium": 3,"4 - Good": 4,"5 - Excellent": 5}
    mappings["health_rate_comparison"] = {"1 - A lot worse": 1,"2 - A little worse": 2,"3 - About the same": 3,
                                         "4 - A little better": 4,"5 - A lot better": 5}
    mappings["pain_perception"] = {999.0: np.nan}
    mappings["activity_regular"] = {"> 2 h and < 5 h per week":2,"< 2 h per week":1,"> 5 h per week":3,"No":0}
    mappings["smoking"] = {"Never smoked":0,"Past smoker (stopped at least 6 months)":1,"Current smoker":2}
    mappings["alcohol_units"] = {999.0: np.nan}
    mappings["katz_index"] = {999.0: np.nan}
    mappings["iadl_grade"] = {999.0: np.nan}
    mappings["comorbidities_count"] = {999: np.nan}
    mappings["comorbidities_significant_count"] = {999: np.nan}
    mappings["medication_count"] = {999: np.nan}

    for column, mapping in mappings.items():
      if column in _df.columns:
          s = _df[column]
          if len(mapping) == 1 and list(mapping.values())[0] is np.nan:
              _df[column] = s.replace(mapping)
          else:
              _df[column] = s.map(mapping).where(s.isin(mapping.keys()), s)
    return _df


def convert_columns_to_numeric(df):
    """
    Convert all columns in a DataFrame to numeric.
    Prints any values that could not be converted.
    Non-convertible values are replaced with NaN.
    """
    for col in df.columns:
        original = df[col]
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Identify values that could not be converted
        failed_mask = original.notna() & df[col].isna()
        failed_values = original[failed_mask].unique()
        if len(failed_values) > 0:
            print(f"Column '{col}' - could not convert values: {failed_values}")
    
    return df

def find_data_issues(df):
    banned_values = [999, 999.0, -391.964, "test not realizable", "test not adequate"]
    banned_values = set(banned_values)
    issues = {}

    for col in df.columns:
        s = df[col]
        col_issues = {}

        # --- banned values ---
        banned_counts = {}
        for val in banned_values:
            if pd.isna(val):
                continue
            n = (s == val).sum()
            if n > 0:
                banned_counts[val] = int(n)

        if banned_counts:
            col_issues["banned_values"] = banned_counts

        # --- non-numeric values (excluding NaN) ---
        numeric = pd.to_numeric(s, errors="coerce")

        non_numeric_mask = ~s.isna() & numeric.isna()
        non_numeric_count = int(non_numeric_mask.sum())

        if non_numeric_count > 0:
            col_issues["non_numeric_count"] = non_numeric_count
            col_issues["non_numeric_examples"] = (
                s[non_numeric_mask]
                .astype(str)
                .unique()
                .tolist()[:5]
            )

        # --- negative numeric values ---
        negative_mask = numeric < 0
        negative_count = int(negative_mask.sum())

        if negative_count > 0:
            col_issues["negative_count"] = negative_count
            col_issues["negative_examples"] = (
                s[negative_mask]
                .astype(str)
                .unique()
                .tolist()[:5]
            )

        if col_issues:
            issues[col] = col_issues

    if issues:
      print("Data issues found:")
      for col, col_issues in issues.items():
          print(f"Column: {col}")
          if "banned_values" in col_issues:
              print("  Banned values:")
              for val, count in col_issues["banned_values"].items():
                  print(f"    Value: {val}, Count: {count}")
          if "non_numeric_count" in col_issues:
              print(f"  Non-numeric count (excluding NaN): {col_issues['non_numeric_count']}")
              print(f"  Examples: {col_issues['non_numeric_examples']}")

def fill_nan_with_mean(_df, columns):
    for col in columns:
        if col in _df.columns:
            mean_value = _df[col].mean()
            _df[col] = _df[col].fillna(mean_value)
    return _df

def fill_nan_with_median(_df, columns):
    for col in columns:
        if col in _df.columns:
            median_value = _df[col].median()
            _df[col] = _df[col].fillna(median_value)
    return _df

    




df = pd.read_csv(input_file)
df = convert_to_nominal_values(df)
find_data_issues(df)
all_columns = df.columns.tolist()
df = convert_columns_to_numeric(df)
df = fill_nan_with_median(df, all_columns)
df.to_csv(output_file, index=False)

