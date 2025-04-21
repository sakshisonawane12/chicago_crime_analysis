import pandas as pd
import scipy.stats as stats

def perform_hypothesis_testing(crime_file, socio_file):
    """
    Perform hypothesis testing between crime count and socioeconomic factors.

    Parameters:
    - crime_file (str): Path to the crime dataset CSV.
    - socio_file (str): Path to the socioeconomic dataset CSV.

    Returns:
    - dict: Dictionary containing correlation coefficients and p-values.
    """

    # Load datasets
    crime_df = pd.read_csv(crime_file)
    socio_df = pd.read_csv(socio_file)

    # Standardize column names for merging
    crime_df.rename(columns={"Community Area": "Community_Area"}, inplace=True)
    socio_df.rename(columns={"Community Area Number": "Community_Area"}, inplace=True)

    # Convert 'Community_Area' to numeric
    crime_df["Community_Area"] = pd.to_numeric(crime_df["Community_Area"], errors="coerce")
    socio_df["Community_Area"] = pd.to_numeric(socio_df["Community_Area"], errors="coerce")

    # Aggregate crime data at the community level
    crime_grouped = crime_df.groupby("Community_Area").agg(
        Crime_Count=("ID", "count")  # Assuming 'ID' is a unique crime identifier
    ).reset_index()

    # Merge crime data with socioeconomic data
    merged_df = crime_grouped.merge(socio_df, on="Community_Area", how="left")

    # Drop rows with missing socioeconomic data
    merged_df.dropna(inplace=True)

    # Strip column names to remove extra spaces
    merged_df.columns = merged_df.columns.str.strip()

    # Define socioeconomic columns to analyze
    socio_columns = [
        "PERCENT OF HOUSING CROWDED",
        "PERCENT HOUSEHOLDS BELOW POVERTY",
        "PERCENT AGED 16+ UNEMPLOYED",
        "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
        "PERCENT AGED UNDER 18 OR OVER 64",
        "PER CAPITA INCOME",
        "HARDSHIP INDEX"
    ]

    # Check if all socio_columns exist in the dataset
    available_columns = [col for col in socio_columns if col in merged_df.columns]

    if not available_columns:
        print("Error: None of the socioeconomic columns are present in the dataset.")
        return {}

    # Perform Pearson correlation test
    correlation_results = {}
    print("\nHypothesis Testing Results (Crime Count vs Socioeconomic Factors):")

    for col in available_columns:
        # Ensure at least two unique values exist and no NaNs in the column
        if merged_df[col].nunique() > 1 and not merged_df[col].isna().all():
            corr, p_value = stats.pearsonr(merged_df["Crime_Count"], merged_df[col])
            correlation_results[col] = (corr, p_value)

            # Interpret correlation strength
            if abs(corr) < 0.3:
                strength = "Weak"
            elif abs(corr) < 0.6:
                strength = "Moderate"
            else:
                strength = "Strong"

            print(f"{col}: Correlation = {corr:.4f} ({strength}), P-value = {p_value:.4f}")
        else:
            print(f"{col}: Skipped (Not enough unique values or all missing)")

    return correlation_results  # Returning the results as a dictionary
