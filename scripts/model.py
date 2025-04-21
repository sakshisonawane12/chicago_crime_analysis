import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import logging
import sys
import os
import io
import base64
import matplotlib.pyplot as plt
from joblib import dump

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plots the confusion matrix and returns the plot as a base64 encoded PNG.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): List of class names. If None, unique values in y_true are used.

    Returns:
        str: Base64 encoded PNG of the confusion matrix plot, or None on error.
    """
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        if labels is None:
            tick_marks = range(len(set(y_true)))
            plt.xticks(tick_marks, set(y_true), rotation=45)
            plt.yticks(tick_marks, set(y_true))
        else:
            tick_marks = range(len(labels))
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)

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
        return confusion_matrix_img_base64

    except Exception as e:
        logging.error(f"Error in plot_confusion_matrix: {e}")
        return None


def train_model(file_path, model_output_path="model.joblib"):
    """
    Trains a Random Forest Classifier model on the data in the given CSV file,
    saves the trained model, evaluates the model, and returns the results.

    Args:
        file_path (str): Path to the CSV data file.
        model_output_path (str, optional): Path to save the trained model.
            Defaults to "model.joblib".

    Returns:
        tuple: (accuracy, classification_report_str, confusion_matrix_img_base64)
               Returns (None, None, None) if an error occurs.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None

    logger.info("Data loaded successfully.")

    # Feature selection and preprocessing
    try:
        if not all(col in df.columns for col in ['Year', 'Arrest', 'Domestic', 'Primary_Type']):
            logger.error("Error:  Not all required columns ('Year', 'Arrest', 'Domestic', 'Primary_Type') are present in the data.")
            return None, None, None

        df = df[['Year', 'Arrest', 'Domestic', 'Primary_Type']]
        df = pd.get_dummies(df, columns=['Primary_Type'], drop_first=True)  # Handles encoding
        logger.info("Features selected and one-hot encoded.")
    except KeyError as e:
        logger.error(f"Error: Column not found: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error during feature selection/preprocessing: {e}")
        return None, None, None

    # Train-Test Split
    try:
        X = df.drop(columns=['Arrest'])
        y = df['Arrest']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Data split into training and testing sets.")
    except KeyError as e:
        logger.error(f"Error: Column not found: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Error during train-test split: {e}")
        return None, None, None

    # Model training with hyperparameter tuning
    try:
        model = RandomForestClassifier(random_state=42)  # Set random_state for reproducibility
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        logger.info("Model training complete.")
        logger.info(f"Best parameters: {grid_search.best_params_}")

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None, None, None

    # Prediction and evaluation
    try:
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_report_str = classification_report(y_test, y_pred)
        confusion_matrix_img_base64 = plot_confusion_matrix(y_test, y_pred)  # Generate confusion matrix plot

        logger.info(f"Model Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report_str}")
    except Exception as e:
        logger.error(f"Error during prediction/evaluation: {e}")
        return None, None, None

    # Save the model
    try:
        dump(best_model, model_output_path)
        logger.info(f"Trained model saved to {model_output_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None, None, None
    return accuracy, classification_report_str, confusion_matrix_img_base64 # Return the confusion matrix data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest Classifier model.")
    parser.add_argument("file_path", help="Path to the CSV data file.")
    parser.add_argument("--model_output_path", default="model.joblib",
                        help="Path to save the trained model (default: model.joblib).")
    args = parser.parse_args()

    train_model(args.file_path, args.model_output_path)
