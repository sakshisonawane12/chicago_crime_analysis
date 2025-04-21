import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_top_crimes(df, save_path):
    """Bar plot of top 10 crime types, saves to a file."""
    crime_counts = df['Primary_Type'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=crime_counts.values, y=crime_counts.index, palette="viridis", hue=crime_counts.index, legend=False)
    plt.xlabel("Count")
    plt.ylabel("Crime Type")
    plt.title("Top 10 Crime Types")
    plt.savefig(save_path)
    plt.close()

def plot_crime_trend(df, save_path):
    """Line chart of crime trends over years, saves to a file."""
    yearly_crime = df.groupby('Year').size()
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_crime.index, yearly_crime.values, marker='o', linestyle='-', color='b')
    plt.xlabel("Year")
    plt.ylabel("Number of Crimes")
    plt.title("Crime Trends Over Years")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_crime_heatmap(df, save_path):
    """Heatmap of crimes by community area and year, saves to a file."""
    pivot_table = df.pivot_table(index='Community_Area', columns='Year', values='ID', aggfunc='count', fill_value=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt='d')
    plt.xlabel("Year")
    plt.ylabel("Community Area")
    plt.title("Crime Heatmap by Community Area and Year")
    plt.savefig(save_path)
    plt.close()

def plot_crime_locations(df, save_path):
    """Scatter plot of crime locations, saves to a file."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['Longitude'], y=df['Latitude'], alpha=0.5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Crime Incidents in the City")
    plt.savefig(save_path)
    plt.close()

def plot_crime_by_hour(df, save_path):
    """Histogram of crime count by hour of the day, saves to a file."""
    df['Hour'] = pd.to_datetime(df['Date'], format="%m/%d/%Y %I:%M:%S %p", errors='coerce').dt.hour
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Hour'], bins=24, kde=True)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Crime Count")
    plt.title("Crime Distribution by Hour")
    plt.xticks(range(24))
    plt.savefig(save_path)
    plt.close()

def plot_crime_pie_chart(df, save_path):
    """Pie chart of crime type distribution, saves to a file."""
    crime_counts = df['Primary_Type'].value_counts().head(10)
    plt.figure(figsize=(8, 8))
    plt.pie(crime_counts, labels=crime_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title("Crime Type Distribution")
    plt.savefig(save_path)
    plt.close()



def run_visualizations(df, output_dir='static/Visualizations'):
    """
    Generates all visualizations and returns a dictionary of file paths.

    Args:
        df (pd.DataFrame): The crime data.
        output_dir (str): The directory to save the plots.

    Returns:
        dict: A dictionary of file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    visualization_files = {}

    plot_top_crimes(df, os.path.join(output_dir, 'top_crimes.png'))
    visualization_files['Top 10 Crime Types'] = 'Visualizations/top_crimes.png'

    plot_crime_trend(df, os.path.join(output_dir, 'crime_trend.png'))
    visualization_files['Crime Trend Over Years'] = 'Visualizations/crime_trend.png'

    plot_crime_heatmap(df, os.path.join(output_dir, 'crime_heatmap_by_community_area_and_year.png'))
    visualization_files['Crime Heatmap by Community Area and Year'] = 'Visualizations/crime_heatmap_by_community_area_and_year.png'

    plot_crime_locations(df, os.path.join(output_dir, 'crime_locations.png'))
    visualization_files['Crime Locations'] = 'Visualizations/crime_locations.png'

    plot_crime_by_hour(df, os.path.join(output_dir, 'crime_by_hour.png'))
    visualization_files['Crime by Hour'] = 'Visualizations/crime_by_hour.png'

    plot_crime_pie_chart(df, os.path.join(output_dir, 'crime_pie_chart.png'))
    visualization_files['Crime Type Distribution'] = 'Visualizations/crime_pie_chart.png'
  
    return visualization_files
