import pandas as pd

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna(subset=['Primary Type', 'Description', 'Location'])

    # Fix column names (if necessary)
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    clean_data("../data/crime.csv", "../data/cleaned_data.csv")
