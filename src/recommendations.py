import pandas as pd
from settings import (
    DATASET_MOVIES_PATH,
    RATINGS_ITEM_ID_COLUMN,
    TRAIN_RATINGS_PATH,
)
from train_model import load_model


"""
    Recommend top_count unseen items for a user, ranked by predicted rating.
    user_index = user to recommend for
    top_count = number of items to return
    Excludes items the user already rated in train (TRAIN_RATINGS_PATH).
    prediction = mu + b_u + b_i + p(u)^T q(i) via model.predict()
    Joins movies.csv for movie titles.
    Returns a dataframe with columns item_index, movie_title, and predicted_rating.
"""
def top_item_recommendations(user_index: int, top_count: int) -> pd.DataFrame:
    recommendation_rows = []
    model = load_model()
    train_dataframe = pd.read_csv(TRAIN_RATINGS_PATH)
    item_count = model.item_factors.shape[0]
    rated_item_indices = set(train_dataframe.loc[
            train_dataframe["user_index"] == user_index, "item_index"
        ])
    
    for item_index in range(item_count):
        if item_index in rated_item_indices:
            continue
        predicted_rating = model.predict(user_index, item_index)
        recommendation_rows.append((item_index, predicted_rating))
    recommendation_rows.sort(key=lambda row: row[1], reverse=True)
    top_recommendation_rows = recommendation_rows[:top_count]
    recommendations_dataframe = pd.DataFrame(
        top_recommendation_rows, columns=["item_index", "predicted_rating"]
    )

    # Map each internal item_index to the original MovieLens movieId from train.
    item_index_to_movie_id = train_dataframe[
        [RATINGS_ITEM_ID_COLUMN, "item_index"]
    ].drop_duplicates(subset="item_index")

    movies_dataframe = pd.read_csv(DATASET_MOVIES_PATH)

    # Add movieId to each recommended row via item_index.
    recommendations_dataframe = recommendations_dataframe.merge(
        item_index_to_movie_id,
        on="item_index",
        how="left",
        validate="one_to_one",
    )

    # Add movie title from movies.csv via movieId.
    recommendations_dataframe = recommendations_dataframe.merge(
        movies_dataframe[[RATINGS_ITEM_ID_COLUMN, "title"]],
        on=RATINGS_ITEM_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    
    # Return the recommendations dataframe with columns
    # item_index, movie_title, and predicted_rating.    
    return recommendations_dataframe[
        ["item_index", "title", "predicted_rating"]
    ].rename(columns={"title": "movie_title"})
