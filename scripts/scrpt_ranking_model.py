import os
import lightgbm as lgb

from src.csv_utils_basic import *
from src.plotting import *
from scrpt_utils import *
from src.gbdt_model import *

def get_file_paths():
    base_dir = os.path.dirname(__file__)
    train_file = os.path.join(base_dir, "..", 'data', 'training_set_VU_DM_engineered.csv')
    test_file = os.path.join(base_dir, "..", 'data', 'test_set_VU_DM_engineered.csv')
    submission_file = os.path.join(base_dir, "..", 'submissions', 'submission_GBDT_5.csv')
    return train_file, test_file, submission_file

def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df

def predict_and_save(model, test_df, features, submission_path):
    test_df['predicted_score'] = model.predict(test_df[features])
    test_df['rank_within_search'] = test_df.groupby('srch_id')['predicted_score'].rank(method='first', ascending=False)
    submission_df = test_df.sort_values(['srch_id', 'rank_within_search'])[['srch_id', 'prop_id']]
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")

def GBDT_model():
    train_file, test_file, submission_file = get_file_paths()

    train_df, test_df = load_data(train_file, test_file)

    features = get_features()
    train_df = create_labels(train_df)

    X_train, y_train, group_train, X_val, y_val, group_val, val_idx = split_data(train_df, features)
    model = train_gbdt(X_train, y_train, group_train, X_val, y_val, group_val)

    evaluate_ndcg(model, train_df, X_val, val_idx)
    predict_and_save(model, test_df, features, submission_file)


if __name__ == "__main__":
    GBDT_model()
