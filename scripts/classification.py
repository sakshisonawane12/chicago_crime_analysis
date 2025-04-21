import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def perform_classification(file_path='data/cleaned_data.csv'):
    # Load dataset
    df = pd.read_csv(file_path)

    print("Dataset Columns:", df.columns)


    crime_col = None
    for col in df.columns:
        if col.strip().lower() in ['primary type', 'primary_type']:
            crime_col = col
            break

    if crime_col is None:
        raise KeyError("No column found for crime type. Check column names.")

    
    df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
    
    df['Hour'] = df['Date'].dt.hour.fillna(0).astype(int)  # Handle NaN hours

    # Create target variable for severe crimes
    df['Severe_Crime'] = df[crime_col].apply(lambda x: 1 if x in ['HOMICIDE', 'ROBBERY'] else 0)

    # Selecting features
    features = ['Hour', 'Latitude', 'Longitude']
    if any(col not in df.columns for col in features):
        raise KeyError(f"Missing columns: {[col for col in features if col not in df.columns]}")

    X = df[features].copy()
    y = df['Severe_Crime']

    # Handle missing values
    X.fillna(X.mean(), inplace=True)

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:\n", report)

    return report  # Return report for further use if needed
