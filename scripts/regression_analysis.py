import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def perform_regression(file_path):
    # Load crime data
    df = pd.read_csv(file_path)

    # Aggregate crime counts per community area
    crime_counts = df.groupby('Community_Area')['ID'].count().reset_index()
    crime_counts.rename(columns={'ID': 'Crime_Count'}, inplace=True)

    # One-hot encode community area
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(crime_counts[['Community_Area']])
    feature_names = encoder.get_feature_names_out(['Community_Area'])
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names)

    # Define features and target
    X = encoded_df
    y = crime_counts['Crime_Count']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print results
    print("Feature Importance:", model.coef_)
    print("Intercept:", model.intercept_)

perform_regression("data/cleaned_data.csv")
