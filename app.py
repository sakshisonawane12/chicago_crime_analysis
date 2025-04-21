import matplotlib
matplotlib.use('Agg')  # This MUST be the first line
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from flask import Flask, render_template, send_file, abort, request, jsonify
import pandas as pd
import os
from scripts import classification, clustering, eda, forecasting, visualization, regression_analysis
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import logging
import sys
import threading
import time
from datetime import datetime

# Define the data directory
DATA_DIR = 'data'
REQUIRED_MONTHS_FOR_SARIMA = 36
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def load_data(file_path):
    """Loads data from a CSV file into a Pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Loaded data successfully from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None


def train_and_evaluate_model(file_path, timeout=600):  # Added timeout
    """
    Trains a RandomForestClassifier model, evaluates its performance, and returns metrics.

    Args:
        file_path (str): Path to the CSV data file.
        timeout (int, optional): Maximum time in seconds to allow for training. Defaults to 600 seconds (10 minutes).

    Returns:
        tuple: (accuracy, classification_report_str, confusion_matrix_img_base64)
               Returns (None, None, None) in case of error or timeout.
    """
    result = [None, None, None]  # Store the result in a list so it can be modified by the thread

    def _train_and_evaluate():
        try:
            logger.info(f"Starting model training and evaluation with file: {file_path}")
            data = load_data(file_path)
            if data is None:
                return

            # 1. Data Preprocessing
            try:
                data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')  # Specify the date format and handle errors
                if data['Date'].isnull().any():  # Check if any dates failed to parse with the first format
                    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True, errors='coerce')  # infer, also with errors
                data['Year'] = data['Date'].dt.year
                data['Month'] = data['Date'].dt.month
                data['Day'] = data['Date'].dt.day
                data['Hour'] = data['Date'].dt.hour
                data['Minute'] = data['Date'].dt.minute

            except ValueError as ve:
                logger.error(f"Error processing date information: {ve}")
                result[0] = None
                result[1] = None
                result[2] = None
                return

            except KeyError as ke:
                logger.error(f"KeyError: 'Date' column not found: {ke}")
                result[0] = None
                result[1] = None
                result[2] = None
                return
            # Adjusted categorical features to reflect missing columns
            categorical_features = ['IUCR', 'Location_Description', 'Arrest', 'Domestic', 'Beat', 'District', 'Ward']
            for col in categorical_features:
                if col not in data.columns:
                    logger.warning(f"Categorical column '{col}' not found in data. Skipping this column.")
                    categorical_features.remove(col)

            # Check for missing values *before* splitting the data
            missing_values = data.isnull().sum()
            if missing_values.sum() > 0:
                logger.info("Missing values before imputation:")
                logger.info(missing_values)
                # Handle missing values using imputation (mean, median, etc.)
                data = data.fillna(data.mean())  # Example: Fill with the mean
                logger.info("Missing values after imputation:")
                logger.info(data.isnull().sum())

            # 2. Feature Selection and Data Splitting
            # Ensure only *available* columns are included in features
            features = ['Year', 'Month', 'Day', 'Hour', 'Minute'] + [f for f in categorical_features if
                                                                    f in data.columns]
            label = 'Arrest'  # changed label
            if label not in data.columns:
                logger.error(f"Target variable '{label}' not found in data.")
                result[0] = None
                result[1] = None
                result[2] = None
                return

            try:
                X = data[features]
                y = data[label]
            except KeyError as e:
                logger.error(f"Error selecting features or target variable: {e}")
                result[0] = None
                result[1] = None
                result[2] = None
                return

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 3. Model Training
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 4. Model Evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            classification_report_str = classification_report(y_test, y_pred)
            confusion_matrix_img = plot_confusion_matrix(y_test, y_pred)

            logger.info("Model training and evaluation complete.")
            result[0] = accuracy
            result[1] = classification_report_str
            result[2] = confusion_matrix_img

        except Exception as e:
            logger.error(f"Error in _train_and_evaluate: {e}")
            result[0] = None
            result[1] = None
            result[2] = None

    thread = threading.Thread(target=_train_and_evaluate)
    thread.start()
    thread.join(timeout=timeout)  # Wait for the thread to complete, with a timeout

    if thread.is_alive():
        logger.error("Model training timed out after {} seconds.".format(timeout))
        return None, None, None  # Return None values to indicate timeout

    return result[0], result[1], result[2]


def plot_confusion_matrix(y_true, y_pred):
    """
    Generates and returns the confusion matrix plot as a base64 encoded PNG.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        str: Base64 encoded PNG of the confusion matrix.
    """
    try:
        logger.info("Generating confusion matrix plot.")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = range(len(set(y_true)))
        plt.xticks(tick_marks, list(set(y_true)), rotation=45)  # Rotate x-ticks for better readability
        plt.yticks(tick_marks, list(set(y_true)))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Add the values to the cells.
        thresh = cm.max() / 2.
        for i, j in enumerate(range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()

        # Save the plot to a BytesIO object, then encode to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        confusion_matrix_img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        logger.info("Confusion matrix plot generated.")
        return confusion_matrix_img_base64

    except Exception as e:
        logger.error(f"Error in plot_confusion_matrix: {e}")
        return None


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/eda')
def eda_page():
    try:
        data = load_data(os.path.join(DATA_DIR, "cleaned_data.csv"))
        if data is None:
            return render_template('error.html', message="Failed to load data for EDA.")
        eda_output = eda.run_eda(data)
        return render_template('eda.html', eda_output=eda_output)
    except Exception as e:
        logger.error(f"Error in eda_page: {e}")
        return render_template('error.html', error_message=f"Error during EDA: {e}")


def get_number_of_months(data_file):
    """
    Placeholder function to get the number of months in your data.
    Implement your actual logic here using pandas or your data loading method.
    """
    import pandas as pd
    try:
        df = pd.read_csv(data_file)
        # Assuming you have a 'Date' column
        df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')
        num_months = df['YearMonth'].nunique()
        return num_months
    except Exception as e:
        logger.error(f"Error reading data for month count: {e}")
        return 0

@app.route('/forecasting_page', methods=['GET', 'POST'])
def forecasting_page():
    forecast_image = None
    forecast_report = None
    error_message = None
    model_type = request.form.get('model_type', 'prophet')  # Default to Prophet on form submission

    if request.method == 'POST':
        if model_type == 'sarima':
            data_file = os.path.join(DATA_DIR, "cleaned_data.csv")
            num_months = get_number_of_months(data_file)
            if num_months < REQUIRED_MONTHS_FOR_SARIMA:
                error_message = f"Cannot forecast with SARIMA: Data has {num_months} months, less than the required {REQUIRED_MONTHS_FOR_SARIMA} months."
            else:
                try:
                    logger.info(f"Attempting SARIMA forecasting with data file: {data_file}")
                    forecast_image_path, forecast_report = forecasting.run_forecasting(file_path=data_file, model_type=model_type)
                    if forecast_image_path:
                        forecast_image = forecast_image_path
                    else:
                        error_message = "Failed to generate SARIMA forecast image."
                except Exception as e:
                    error_message = f"An error occurred during SARIMA forecasting: {e}"
        elif model_type == 'prophet':
            try:
                data_file = os.path.join(DATA_DIR, "cleaned_data.csv")
                logger.info(f"Attempting Prophet forecasting with data file: {data_file}")
                forecast_image_path, forecast_report = forecasting.run_forecasting(file_path=data_file, model_type=model_type)
                if forecast_image_path:
                    forecast_image = forecast_image_path
                else:
                    error_message = "Failed to generate Prophet forecast image."
            except Exception as e:
                error_message = f"An error occurred during Prophet forecasting: {e}"
    else:
        # On initial page load (GET request), default to showing Prophet UI
        model_type = 'prophet'

    return render_template('forecasting.html', forecast_image=forecast_image,
                           forecast_report=forecast_report, error_message=error_message,
                           model_type=model_type)
@app.route('/classification')
def classification_page():
    try:
        data_file = os.path.join(DATA_DIR, "cleaned_data.csv")
        report = classification.perform_classification(data_file)
        return render_template('classification.html', classification_output=report)
    except Exception as e:
        logger.error(f"Error in classification_page: {e}")
        return render_template('classification.html', error_message=f"Error during Classification: {e}")



@app.route('/clustering')
def clustering_page():
    try:
        image_filename = clustering.perform_clustering(os.path.join(DATA_DIR, "cleaned_data.csv"))
        if image_filename:
            clustering_image_path = os.path.join('Visualizations', image_filename)
            return render_template('clustering.html', clustering_image=clustering_image_path)
        else:
            return render_template('clustering.html', error_message="Clustering analysis failed to generate a plot.")
    except Exception as e:
        logger.error(f"Error in clustering_page: {e}")
        return render_template('clustering.html', error_message=f"Error during Clustering: {e}")


@app.route('/visualizations')
def visualizations_page():
    try:
        data_file = os.path.join(DATA_DIR, "cleaned_data.csv")
        df = load_data(data_file)
        if df is None:
            return render_template('error.html', message="Failed to load data for visualizations.")
        visualization_files = visualization.run_visualizations(df)
        logger.info(f"Visualizations: {visualization_files}")
        return render_template('visualizations.html', visualizations_output=visualization_files)
    except Exception as e:
        error_message = f"An error occurred during visualization: {e}"
        logger.error(error_message)
        return render_template('visualizations.html', error_message=error_message)


@app.route('/crime_analysis')
def crime_analysis():
    """
    Route for crime analysis.
    """
    data = load_data(os.path.join(DATA_DIR, "cleaned_data.csv"))
    if data is None:
        return render_template('error.html', message="Failed to load data for crime analysis.")
    top_locations, yearly_crime_count = analyze_crime_trends(data)

    logger.info(f"Top Locations (in Flask): {top_locations}")
    logger.info(f"Yearly Crime Count (in Flask): {yearly_crime_count}")

    if top_locations and yearly_crime_count:
        return render_template('crime_analysis.html', top_locations=top_locations,
                               yearly_crime_count=yearly_crime_count)
    else:
        return render_template('error.html', message="Error occurred during crime analysis.")


def analyze_crime_trends(df):
    """
    Performs crime trend analysis.

    Args:
        df (pandas.DataFrame): The crime data.

    Returns:
        tuple: A tuple containing two dictionaries:
               - top_locations: A dictionary of top 10 crime locations and their counts.
               - yearly_crime_count: A dictionary of year-wise crime counts.
               Returns None, None in case of error.
    """
    try:
        # Most common crime locations
        top_locations_series = df['Location_Description'].value_counts().head(10)
        top_locations = top_locations_series.to_dict()

        # Year-wise crime trends
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        yearly_crime_count_series = df.groupby("Year")["ID"].count()
        yearly_crime_count = yearly_crime_count_series.to_dict()

        return top_locations, yearly_crime_count

    except Exception as e:
        logger.error(f"Error in analyze_crime_trends: {e}")
        return None, None


@app.route('/model')
def model_page():
    """
    Renders the model evaluation page.
    """
    data_file = os.path.join(DATA_DIR, "cleaned_data.csv")
    try:
        accuracy, classification_report_str, confusion_matrix_img = train_and_evaluate_model(data_file) #added
        if accuracy is None and classification_report_str is None and confusion_matrix_img is None: #added
            return render_template('error.html', message="Model training failed, possibly due to a timeout or error.")
        elif accuracy is None:
            return render_template('error.html', message="Model training failed.")

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Classification Report: {classification_report_str}")

        return render_template('model.html', accuracy=accuracy, classification_report=classification_report_str, confusion_matrix_img=confusion_matrix_img) #added
    except Exception as e:
        error_message = f"Error in model_page: {e}"
        logger.error(error_message)
        return render_template('error.html', message=error_message)

@app.route('/regression')
def regression_page():
    try:
        data_file = os.path.join(DATA_DIR, "cleaned_data.csv")
        feature_importance, intercept = regression_analysis.perform_regression(data_file)
        return render_template('regression_analysis.html', feature_importance=feature_importance.tolist(), intercept=intercept)
    except Exception as e:
        logger.error(f"Error in regression_page: {e}")
        return render_template('regression_analysis.html', error_message=f"Error during Regression Analysis: {e}")


if __name__ == '__main__':
    app.run(debug=True)