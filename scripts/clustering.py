import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np  # Import NumPy

def perform_clustering(file_path='data/simple_location_data.csv', n_clusters=3):
    """
    Performs K-Means clustering on simplified location data (Latitude, Longitude)
    and saves a simple scatter plot of clusters.

    Args:
        file_path (str, optional): Path to the simplified location data CSV file.
            Defaults to 'data/simple_location_data.csv'.
        n_clusters (int, optional): The number of clusters to form. Defaults to 3.

    Returns:
        str or None: The filename of the saved cluster plot image if successful,
                     None otherwise.
    """
    print("perform_clustering() is being called...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

    if not all(col in df.columns for col in ['Latitude', 'Longitude']):
        print("Error: Required columns ('Latitude', 'Longitude') not found.")
        return None

    location_data = df[['Latitude', 'Longitude']].dropna().copy()
    scaler = StandardScaler()
    scaled_location = scaler.fit_transform(location_data)

    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(scaled_location)
        df['Cluster'] = clusters  # Add cluster labels to the DataFrame
    except Exception as e:
        print(f"Error performing K-Means clustering: {e}")
        return None

    static_folder = 'static'
    visualizations_folder = 'Visualizations'
    os.makedirs(os.path.join(static_folder, visualizations_folder), exist_ok=True)
    image_filename = 'simplest_clusters_plot.png'
    image_path = os.path.join(static_folder, visualizations_folder, image_filename)

    plt.figure(figsize=(8, 6))
    # Use a colormap directly with scatter for simpler color mapping
    cmap = plt.cm.get_cmap('viridis', n_clusters)
    plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap=cmap, alpha=0.7, s=30)
    plt.title(f"Simplest Crime Clusters (K={n_clusters})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(ticks=np.arange(n_clusters))  # Add a colorbar with ticks
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()

    print(f"Simplest cluster plot saved at: {image_path}")
    return image_filename

if __name__ == '__main__':
    # Example of how to create a simple CSV for testing
    data = {
        'Latitude': [34.0522, 34.0560, 34.0480, 34.0600, 34.0540, 34.0700, 34.0450, 34.0650, 34.0500],
        'Longitude': [-118.2437, -118.2390, -118.2500, -118.2350, -118.2450, -118.2300, -118.2550, -118.2250, -118.2480]
    }
    simple_df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    simple_df.to_csv('data/simple_location_data.csv', index=False)
    print("Created 'data/simple_location_data.csv' for testing.")