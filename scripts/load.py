import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("First 5 Rows of Dataset:")
    print(df.head())

if __name__ == "__main__":
    load_data("../data/crime.csv")
