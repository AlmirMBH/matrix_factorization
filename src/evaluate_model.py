import numpy as np
import pandas as pd
from typing import TypedDict
from numpy.typing import NDArray
from biased_mf import BiasedMatrixFactorization
from metrics import mean_absolute_error, root_mean_squared_error
from settings import (
    DATASET_MOVIES_PATH,
    PLOT_EXAMPLE_COUNT,
    RATINGS_ITEM_ID_COLUMN,
    RATINGS_USER_ID_COLUMN,
    RATINGS_VALUE_COLUMN,
    RANDOM_SEED,
)


class EvaluationResults(TypedDict):
    test_user_indices: NDArray[int]
    test_item_indices: NDArray[int]
    test_ratings: NDArray[float]
    test_predicted_ratings: NDArray[float]
    test_rmse: float
    test_mae: float
    baseline_rmse: float
    baseline_mae: float


"""
Baseline prediction is the train global mean rating (1x2067 array).
Baseline RMSE is the RMSE of the baseline predicted ratings, see
formula in the function implementation.
"""
def evaluate_model(
    model: BiasedMatrixFactorization, test_dataframe: pd.DataFrame
) -> EvaluationResults:
    test_user_indices = test_dataframe["user_index"].to_numpy()
    test_item_indices = test_dataframe["item_index"].to_numpy()
    test_ratings = test_dataframe[RATINGS_VALUE_COLUMN].to_numpy()
    test_predicted_ratings = model.predict_ratings_for_pairs(test_user_indices, test_item_indices)
    test_rmse = root_mean_squared_error(test_ratings, test_predicted_ratings)
    test_mae = mean_absolute_error(test_ratings, test_predicted_ratings)
    baseline_predicted_ratings = np.full(len(test_ratings), model.global_mean_rating)
    baseline_rmse = root_mean_squared_error(test_ratings, baseline_predicted_ratings)
    baseline_mae = mean_absolute_error(test_ratings, baseline_predicted_ratings)

    return {
        "test_user_indices": test_user_indices,
        "test_item_indices": test_item_indices,
        "test_ratings": test_ratings,
        "test_predicted_ratings": test_predicted_ratings,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
    }


"""Format the test predictions into a dataframe."""
def test_predictions_dataframe(
    evaluation_results: EvaluationResults,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_index": evaluation_results["test_user_indices"],
            "item_index": evaluation_results["test_item_indices"],
            "true_rating": evaluation_results["test_ratings"],
            "predicted_rating": evaluation_results["test_predicted_ratings"],
        }
    )


"""
Sample test rows with known ratings for the metrics plot.
This is evaluation display only. It is not the same as top_item_recommendations,
which ranks unrated items for one user.
"""
def plot_example_predictions_dataframe(
    test_dataframe: pd.DataFrame,
    evaluation_results: EvaluationResults,
    example_count: int = PLOT_EXAMPLE_COUNT,
) -> pd.DataFrame:
    movies_dataframe = pd.read_csv(DATASET_MOVIES_PATH)
    predictions_dataframe = test_predictions_dataframe(evaluation_results)
    merged_dataframe = test_dataframe.merge(
        predictions_dataframe,
        on=["user_index", "item_index"],
    )
    merged_dataframe = merged_dataframe.merge(
        movies_dataframe[[RATINGS_ITEM_ID_COLUMN, "title"]],
        on=RATINGS_ITEM_ID_COLUMN,
    )
    sample_count = min(example_count, len(merged_dataframe))
    example_rows = merged_dataframe.sample(n=sample_count, random_state=RANDOM_SEED)
    return pd.DataFrame(
        {
            "user_id": example_rows[RATINGS_USER_ID_COLUMN].astype(int),
            "movie_title": example_rows["title"],
            "true_rating": example_rows[RATINGS_VALUE_COLUMN].round(2),
            "predicted_rating": example_rows["predicted_rating"].round(2),
        }
    ).reset_index(drop=True)
