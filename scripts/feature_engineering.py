import pandas as pd

# Load datasets
crime_df = pd.read_csv("data/crime.csv")  # Crime dataset (2 lakh entries)
socio_df = pd.read_csv("data/socioeconomic.csv")  # Socioeconomic dataset (78 entries)
print("Feature Engineering")
# Standardize column names to avoid space issues
crime_df.columns = crime_df.columns.str.strip().str.replace(" ", "_")
socio_df.columns = socio_df.columns.str.strip().str.replace(" ", "_")

# Ensure column names are correct
print("Crime Dataset Columns:", crime_df.columns)
print("Socioeconomic Dataset Columns:", socio_df.columns)

# Rename "Community_Area_Number" in socio_df to match crime_df's "Community_Area"
socio_df.rename(columns={"Community_Area_Number": "Community_Area"}, inplace=True)

# Aggregate crime data at the 'Community_Area' level
crime_grouped = crime_df.groupby("Community_Area").agg(
    Crime_Count=("ID", "count")  # Assuming 'ID' is a unique identifier for crimes
).reset_index()

# Merge with socioeconomic data on 'Community_Area'
merged_df = crime_grouped.merge(socio_df, on="Community_Area", how="left")

# Ensure required columns exist before calculations
if "Population" in merged_df.columns and "Median_Income" in merged_df.columns:
    # Convert to numeric to avoid errors
    merged_df["Population"] = pd.to_numeric(merged_df["Population"], errors="coerce")
    merged_df["Median_Income"] = pd.to_numeric(merged_df["Median_Income"], errors="coerce")

    # Calculate Crime Rate per 1000 people
    merged_df["Crime_Rate_per_1000"] = (merged_df["Crime_Count"] / merged_df["Population"]) * 1000

    # Compute Median Income vs. Crime Ratio
    merged_df["Crime_to_Income_Ratio"] = merged_df["Crime_Count"] / merged_df["Median_Income"]

# Load crime data again to analyze time trends
crime_df["Date"] = pd.to_datetime(crime_df["Date"], errors="coerce")  # Convert date column to datetime format

# Drop NaT values in the Date column
crime_df = crime_df.dropna(subset=["Date"])

# Extract Year-Month for trend analysis
crime_df["Year_Month"] = crime_df["Date"].dt.to_period("M")

# Aggregate crime count over time
crime_trend = crime_df.groupby("Year_Month").agg(Crime_Count=("ID", "count")).reset_index()

# Convert 'Year_Month' to string format for visualization
crime_trend["Year_Month"] = crime_trend["Year_Month"].astype(str)

# Display results
print("\nFeature Engineered Dataset (Community-Level Crime Stats):")
print(merged_df.head())

print("\nCrime Trend Over Time (Monthly Aggregation):")
print(crime_trend.head())

# Save the processed datasets
merged_df.to_csv("data/processed_crime_socio.csv", index=False)
crime_trend.to_csv("data/crime_trend.csv", index=False)

print("\nData processing completed successfully!")
