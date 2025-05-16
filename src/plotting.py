import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_distributions(dataframe, file_name=None):
    """
    Plot distributions of all numerical features in the dataframe.
    The function creates histograms for each numerical feature and saves the plot if a file name is provided.
    
    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the data to plot.
        file_name (str): The name of the file to save the plot. If None, the plot will not be saved.
    """
    sampled_df = dataframe.sample(frac=0.1, random_state=1)  # Sample 10% of the data for faster plotting
    numeric_cols = sampled_df.select_dtypes(include=[np.number]).columns
    sampled_df[numeric_cols].hist(bins=30, figsize=(15, 10))

    plt.suptitle("Distributions of Numerical Features")
    plt.tight_layout()

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", 'figs', 'distributions', file_name)
        plt.savefig(file_path, dpi=300)
        
    plt.show()

def plot_single_distribution(dataframe, column_name, title=None, xlabel=None, ylabel=None, file_name=None):
    """
    Plot the distribution of a single numerical feature in the dataframe.
    The function creates a histogram with a kernel density estimate (KDE) overlay and saves the plot if a file name is provided.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the data to plot.
        column_name (str): The name of the column to plot.
        title (str): The title of the plot. If None, no title will be set.
        xlabel (str): The label for the x-axis. If None, no label will be set.
        ylabel (str): The label for the y-axis. If None, no label will be set.
        file_name (str): The name of the file to save the plot. If None, the plot will not be saved.
    """
    plt.figure(figsize=(4, 3))
    sns.histplot(dataframe[column_name], bins=30, kde=True, alpha=0.6)
    
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.tight_layout()

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", 'figs', 'distributions', file_name)
        plt.savefig(file_path, dpi=300)

    #plt.show()
    plt.close()


def create_summary_table_visualization(dataframe, file_name=None):
    """
    Creates a summary table visualization of the dataframe and saves it as an image.
    The function uses matplotlib to create a table and saves the plot if a file name is provided.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the data to visualize.
        file_name (str): The name of the file to save the plot. If None, the plot will not be saved.
    """
    fig, ax = plt.subplots(figsize=(20, len(dataframe) * 0.2 + 2))
    ax.axis('tight')
    ax.axis('off')

    table_data = [dataframe.columns.tolist()] + dataframe.values.tolist()
    table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(dataframe.columns))))

    #plt.title("Summary Statistics for Variables", pad=20, fontsize=16)
    plt.tight_layout()

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", 'figs', 'summaries', file_name)
        plt.savefig(file_path, dpi=300)

    plt.show()