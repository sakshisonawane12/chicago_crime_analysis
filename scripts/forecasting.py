import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from prophet import Prophet
from io import StringIO


def run_forecasting(file_path="data/cleaned_data.csv", model_type="prophet"):
    """
    Predicts values for the next 12 months using the Prophet model.
    """
    print(f"Running forecasting with {model_type}...")
    try:
        # Load dataset
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(os.path.join(file_path))
        print("Data loaded successfully.")
        print(f"Initial DataFrame head:\n{df.head()}")
        print(f"Initial DataFrame info:\n{df.info()}")
    except FileNotFoundError:
        error_message = f"Error: File '{file_path}' not found."
        print(error_message)
        return None, None

    # Ensure 'Date' column exists
    if 'Date' not in df.columns:
        error_message = "Error: 'Date' column not found in dataset.  Prophet requires a 'Date' column."
        print(error_message)
        return None, None

    # Convert 'Date' column to datetime format, inferring the format
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        print("Date column converted to datetime.")
        print(f"DataFrame head after date conversion:\n{df.head()}")
        print(f"DataFrame info after date conversion:\n{df.info()}")
    except Exception as e:
        error_message = f"Error converting 'Date' to datetime: {e}"
        print(error_message)
        return None, None

    # Drop rows with NaT (invalid dates)
    df.dropna(subset=['Date'], inplace=True)
    print("Invalid dates dropped.")
    print(f"DataFrame head after dropping NaT:\n{df.head()}")
    print(f"Number of rows after dropping NaT: {len(df)}")

    # Resample to daily frequency, aggregating by counting the number of occurrences
    df_resampled = df.set_index('Date').resample('D').size().reset_index(name='y')

    # Prepare the data for Prophet
    df_prophet = df_resampled.rename(columns={'Date': 'ds'})
    print("Prophet dataframe head after resampling: \n", df_prophet.head())
    print("Prophet dataframe info: \n", df_prophet.info())

    # Fit the model and make predictions
    try:
        print("Fitting Prophet model...")
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        print("Prophet model fitted successfully.")
        print(f"Prophet forecast: \n{forecast.head()}")
    except Exception as e:
        error_message = f"Error fitting Prophet model: {e}"
        print(error_message)
        return None, None

    # Generate Report
    report_buffer = StringIO()
    report_buffer.write("Prophet Forecasting Report\n")
    report_buffer.write(
        "----------------------------------------------------------------------------------------\n"
    )
    report_buffer.write("Forecast Summary:\n")
    report_buffer.write(forecast[['ds', 'yhat']].to_string())  # Simplified forecast
    report_buffer.write(
        "\n----------------------------------------------------------------------------------------\n"
    )
    report_string = report_buffer.getvalue()  # Get the string

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.plot(df_prophet['ds'], df_prophet['y'], label='Observed', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='dashed', color='red')
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Prophet Forecasting")
    plt.grid(True)

    # Save the plot to a file
    static_folder = 'static'
    visualizations_folder = 'Visualizations'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.join(static_folder, visualizations_folder), exist_ok=True)
    image_filename = f'forecast_prophet.png'
    image_path = os.path.join(static_folder, visualizations_folder, image_filename)
    plt.savefig(image_path)
    plt.close()

    print(f"Forecast plot saved to: {image_path}")
    return os.path.join(visualizations_folder, image_filename), report_string
