import pandas as pd


def run_eda(data):
    # Compute top crime types and missing values
    top_crime_types = data['Primary_Type'].value_counts().head(10).to_frame().to_html(classes='table table-bordered')
    missing_values = data.isnull().sum().to_frame().to_html(classes='table table-bordered')

    # Return results as a dictionary containing HTML strings
    return {
        'top_crime_types': top_crime_types,
        'missing_values': missing_values
    }

def explore_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Print basic information
    print("Dataset Overview:")
    print(df.info())

    # Print missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Print top crime types
    print("\nTop Crime Types:")
    print(df['Primary_Type'].value_counts().head(10))
    
    return df  # return the dataframe if you want to explore further

if __name__ == "__main__":
    explore_data("../data/cleaned_data.csv")  # path to your data file
