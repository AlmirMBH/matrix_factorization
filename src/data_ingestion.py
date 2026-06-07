import pandas as pd
from export_outputs import save_csv
from settings import (
    DATASET_RATINGS_PATH,
    RATINGS_ITEM_ID_COLUMN,
    RATINGS_USER_ID_COLUMN,
    RANDOM_SEED,
    TEST_FRACTION,
    TEST_RATINGS_PATH,
    TRAIN_RATINGS_PATH,
)


"""
    Load the ratings CSV and remap original user/item IDs to sequential indices.
    Original IDs (e.g. MovieLens userId, movieId) are not always 0, 1, 2, ...
    user_index = 0 .. user_count - 1
    item_index = 0 .. item_count - 1
    user_count = number of unique users
    item_count = number of unique items
    rating_count = number of observed ratings
    rated_user_item_pairs_ratio = rating_count / (user_count * item_count)
    Returns the remapped dataframe plus these counts and the sparsity ratio.
"""
def load_and_remap_ratings() -> tuple[pd.DataFrame, int, int, int, float]:
    ratings_dataframe = pd.read_csv(DATASET_RATINGS_PATH)
    unique_user_ids = ratings_dataframe[RATINGS_USER_ID_COLUMN].unique()
    unique_item_ids = ratings_dataframe[RATINGS_ITEM_ID_COLUMN].unique()

    # Map user and item IDs to indices, as they are not sequential.
    user_id_to_index = {
        user_id: user_index
        for user_index, user_id in enumerate(unique_user_ids)
    }
    item_id_to_index = {
        item_id: item_index
        for item_index, item_id in enumerate(unique_item_ids)
    }

    # Map the user and item IDs to indices, as they are not sequential (0, 1, 2, ...).
    ratings_dataframe["user_index"] = ratings_dataframe[RATINGS_USER_ID_COLUMN].map(user_id_to_index)
    ratings_dataframe["item_index"] = ratings_dataframe[RATINGS_ITEM_ID_COLUMN].map(item_id_to_index)

    user_count = len(unique_user_ids)
    item_count = len(unique_item_ids)
    rating_count = len(ratings_dataframe)

    # Calculate the ratio of rated user-item pairs (users=rows, items=columns, how many cells are filled)
    rated_user_item_pairs_ratio = rating_count / (user_count * item_count)

    return (
        ratings_dataframe,
        user_count,
        item_count,
        rating_count,
        rated_user_item_pairs_ratio,
    )


"""
    Split remapped ratings into train and test sets per user.
    For each user_index, shuffle that user's rows with RANDOM_SEED (see settings).
    test_row_count = max(1, floor(user_rating_count * TEST_FRACTION))
    Users with one rating stay entirely in train.
    Returns train_dataframe and test_dataframe.
"""
def split_train_and_test(ratings_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    
    for _, user_ratings in ratings_dataframe.groupby("user_index", sort=False):
        shuffled_user_ratings = user_ratings.sample(frac=1.0, random_state=RANDOM_SEED)
        if len(shuffled_user_ratings) == 1:
            train_parts.append(shuffled_user_ratings)
            continue
        test_row_count = max(1, int(len(shuffled_user_ratings) * TEST_FRACTION))
        test_parts.append(shuffled_user_ratings.iloc[:test_row_count])
        train_parts.append(shuffled_user_ratings.iloc[test_row_count:])
    train_dataframe = pd.concat(train_parts, ignore_index=True)
    test_dataframe = pd.concat(test_parts, ignore_index=True)
    return train_dataframe, test_dataframe


"""
    Load cached train/test CSVs from output/ or create them on first run.
    If TRAIN_RATINGS_PATH and TEST_RATINGS_PATH exist, read them and recompute counts.
    Otherwise call load_and_remap_ratings(), split_train_and_test(), and save both CSVs.
    user_count = number of unique users (max user_index + 1)
    item_count = number of unique items (max item_index + 1)
    rating_count = len(train) + len(test)
    rated_user_item_pairs_ratio = rating_count / (user_count * item_count)
    Returns train_dataframe, test_dataframe, and these counts.
"""
def load_or_create_train_test_split() -> tuple[pd.DataFrame, pd.DataFrame, int, int, int, float]:
    train_path = TRAIN_RATINGS_PATH
    test_path = TEST_RATINGS_PATH
    output_directory = train_path.parent
    output_directory.mkdir(exist_ok=True)

    # If we have already run the script, we can load the data from the csv files generated in the previous run
    if train_path.exists() and test_path.exists():
        train_dataframe = pd.read_csv(train_path)
        test_dataframe = pd.read_csv(test_path)
        # Take the largest Id from the datsets (rows shuffled to avoid bias)
        user_count = int(max(train_dataframe["user_index"].max(), test_dataframe["user_index"].max())+ 1)
        item_count = int(max(train_dataframe["item_index"].max(), test_dataframe["item_index"].max())+ 1)
        rating_count = len(train_dataframe) + len(test_dataframe)
        # Calculate the ratio of rated user-item pairs (users=rows, items=columns, how many cells are filled)
        rated_user_item_pairs_ratio = rating_count / (user_count * item_count)
    else:
        (ratings_dataframe, user_count, item_count, rating_count, rated_user_item_pairs_ratio
        )= load_and_remap_ratings()
        train_dataframe, test_dataframe = split_train_and_test(ratings_dataframe)
        save_csv(output_directory, train_path.name, train_dataframe)
        save_csv(output_directory, test_path.name, test_dataframe)

    return (
        train_dataframe,
        test_dataframe,
        user_count,
        item_count,
        rating_count,
        rated_user_item_pairs_ratio,
    )
