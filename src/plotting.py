import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lightgbm as lgb
import pandas as pd

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

def create_correlation_heatmap(dataframe, file_name=None):
    """
    Creates a correlation heatmap of the dataframe and saves it as an image.
    The function uses seaborn to create the heatmap and saves the plot if a file name is provided.

    Parameters:
        dataframe (pd.DataFrame): The dataframe containing the data to visualize.
        file_name (str): The name of the file to save the plot. If None, the plot will not be saved.
    """
    plt.figure(figsize=(48, 40))
    corr = dataframe.corr()
    sns.heatmap(
        corr,
        annot=True,  # Show correlation values in each square
        fmt=".2f",   # Format to 2 decimal places
        cmap='coolwarm',
        square=True,
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={"shrink": .8}
    )
    
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    if file_name is not None:
        file_path = os.path.join(os.path.dirname(__file__), "..", 'figs', 'correlation', file_name)
        plt.savefig(file_path, dpi=300)

    plt.show()

def plot_feature_distribution_by_label(df, features, label_col='relevance', bins=30):
    """
    Plots the distribution of each feature grouped by the label column.
    """
    for feat in features:
        plt.figure(figsize=(7, 4))
        for label in sorted(df[label_col].unique()):
            sns.histplot(df[df[label_col] == label][feat], 
                         label=f"{label_col}={label}", 
                         bins=bins, 
                         kde=True, 
                         stat="density", 
                         alpha=0.3)
        plt.title(f"Distribution of {feat} by {label_col}")
        plt.xlabel(feat)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"feature_distribution_{feat}.png", dpi=300)

def plot_predicted_score_vs_relevance(model, X, y, title="Predicted Score vs. Relevance"):
    """
    Plots the distribution of predicted scores for each relevance label.
    """
    df = pd.DataFrame({
        "predicted_score": model.predict(X),
        "relevance": y.values
    })
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="relevance", y="predicted_score", data=df)
    plt.title(title)
    plt.xlabel("Relevance")
    plt.ylabel("Predicted Score")
    plt.tight_layout()

def plot_ndcg_by_country(bias_df, k=5):
    sns.barplot(data=bias_df.sort_values(f'NDCG@{k}'), x=f'NDCG@{k}', y='prop_country_id')
    plt.title(f'NDCG@{k} by Property Country')
    plt.xlabel(f'Mean NDCG@{k}')
    plt.ylabel('prop_country_id')
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

def save_feature_importance_plots(best_model):
    figs_dir = os.path.join(os.path.dirname(__file__), "..", 'figs', 'model_eval')
    os.makedirs(figs_dir, exist_ok=True)
    lgb.plot_importance(best_model, max_num_features=40, importance_type='split', figsize=(12, 8))
    plt.title("Feature Importance (Split)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'feature_importance_split.png'), dpi=300)
    print("Feature importance plot saved as 'feature_importance_split.png'")
    lgb.plot_importance(best_model, max_num_features=40, importance_type='gain', figsize=(12, 8))
    plt.title("Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'feature_importance_gain.png'), dpi=300)
    print("Feature importance plot saved as 'feature_importance_gain.png'")
    lgb.plot_importance(best_model, max_num_features=40, figsize=(12, 8))
    #plt.title("Feature Importance (All)")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, 'feature_importance_all.png'), dpi=300)
    print("Feature importance plot saved as 'feature_importance_all.png'")
    #plt.close('all')