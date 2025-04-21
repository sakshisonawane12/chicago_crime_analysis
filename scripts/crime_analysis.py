import pandas as pd

def analyze_crime_trends(file_path='data/cleaned_data.csv'):
    df = pd.read_csv(file_path)

    # Most common crime locations
    print("\nTop Locations for Crimes:")
    print(df['Location_Description'].value_counts().head(10))

    # Year-wise crime trends
    print("\nYearly Crime Count:")
    print(df.groupby("Year")["ID"].count())

if __name__ == "__main__":
    analyze_crime_trends("../data/cleaned_data.csv")
