# Expedia Hotel Recommendation Model

Victoria Peterson - 15476758


## Table of Contents

1. [Overview](#overview)
2. [Usage and Installation](#usage-and-installation)
3. [Implementation](#implementation)
4. [Model](#model)
5. [Feature Engineering Highlights](#feature-engineering-highlights)
6. [Contributing](#contributing)
7. [License](#license)

## Overview



## Usage and Installation

To run the simulations and generate plots, follow these steps:

1. Clone the repository:

    ```sh
    gh repo clone VictorianHues/DataMining_Assignment2
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**: Ensure you have all the required dependencies installed. You can install them using the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

4. **Initialize Directories as Packages**: This step will ensure that all function imports from different directories are recognized by initializing the directories as packages in the virtual environment. This step may be unnecessary if your IDE automatically established the environmental variable `PYTHON_PATH` for your imports:

    ```sh
    pip install -e .
    ```

5. **Download the Expedia Data Set**: Download the dataset from the following link and place it in a ```data``` folder in the repo root:

    [Expedia Data Set (Google Drive)](https://drive.google.com/drive/folders/1KNmaIjdRvShXawcAUHKZcZH8bAGQ6fSO?usp=sharing)

6. **Complete Data Analysis and Preparation**: Run the scripts in the following order to complete data analysis and preparation:

    ```sh
    python scripts/scrpt_data_analysis.py
    python scripts/scrpt_data_cleaning.py
    python scripts/scrpt_imputation.py
    python scripts/scrpt_feature_engineering.py
    ```

7. **Run the model**: Adjust parameters and inputs, then run either of the following modelling tasks:

    ```sh
    python scripts/scrpt_gbdt.py
    python scripts/scrpt_knn.py
    ```

## Implementation

### Source Code

- csv_utils_basic.py
- data_analysis.py
- data_cleaning.py
- feature_engineering.py
- gbdt_model.py
- imputation.py
- knn_recommender.py
- plotting.py

### Scripts

- scrpt_data_analysis.py
- scrpt_data_cleaning.py
- scrpt_feature_engineering.py
- scrpt_gbdt.py
- scrpt_imputation.py
- scrpt_knn_recommender.py
- scrpt_knn.py
- scrpt_utils.py

## Model

### LightGBM LambdaRank

- Uses lambdarank objective with NDCG@5 evaluation metric.
- Includes strong feature engineering (e.g., price z-scores, country normalization).
- Incorporates fairness-aware transformations to reduce geographic bias.
- Best NDCG@5: 0.489 (val)

### K-Nearest Neighbors (KNN)

- Sampled subset of sessions used due to scalability limits.
- Predicts relevance using per-hotel features within sampled search sessions.
- Evaluated with ranking metrics (NDCG@5).
- Best NDCG@5: ~0.27 (val)

## Feature Engineering Highlights

Statistical features: log-transformed prices, z-scores.

Search-relative ranks: hotel rank within a search session by price or score.

Global aggregates: average review scores and prices by property and country.

Fairness-normalized attributes: price, star rating, and review normalized by country to reduce economic bias.

![Feature Importance Gain](https://github.com/VictorianHues/DataMining_Assignment2/blob/main/figs/model_eval/feature_importance_gain.png)

![Feature Importance Split](https://github.com/VictorianHues/DataMining_Assignment2/blob/main/figs/model_eval/feature_importance_split.png)

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
