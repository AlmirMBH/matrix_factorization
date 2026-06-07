import pandas as pd
from settings import TRAIN_RATINGS_PATH
from train_model import load_model


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
    return pd.DataFrame(top_recommendation_rows, columns=["item_index", "predicted_rating"])
