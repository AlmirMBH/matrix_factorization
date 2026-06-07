import numpy as np
from numpy.typing import NDArray


"""
    Compute root mean squared error on test predictions.
    RMSE = sqrt(mean((true_rating - predicted_rating)^2))
    true_ratings = observed ratings
    predicted_ratings = model predictions for the same pairs
"""
def root_mean_squared_error(
    true_ratings: NDArray[float], predicted_ratings: NDArray[float]
) -> float:
    squared_errors = (true_ratings - predicted_ratings) ** 2
    return float(np.sqrt(np.mean(squared_errors)))


"""
    Compute mean absolute error on test predictions.
    MAE = mean(|true_rating - predicted_rating|)
    true_ratings = observed ratings
    predicted_ratings = model predictions for the same pairs
"""
def mean_absolute_error(
    true_ratings: NDArray[float], predicted_ratings: NDArray[float]
) -> float:
    absolute_errors = np.abs(true_ratings - predicted_ratings)
    return float(np.mean(absolute_errors))


"""
    Compare model metrics to the global-mean baseline.
    test_rmse, test_mae = model error on the test split
    baseline_rmse, baseline_mae = errors when every prediction
    is the train global mean. Returns a message stating whether
    the model is better, worse, or mixed vs the baseline.
"""
def baseline_comparison_message(
    test_rmse: float,
    test_mae: float,
    baseline_rmse: float,
    baseline_mae: float,
) -> str:
    rmse_is_lower = test_rmse < baseline_rmse
    mae_is_lower = test_mae < baseline_mae
    if rmse_is_lower and mae_is_lower:
        return (
            "The model performs better than the global-mean baseline "
            "because both RMSE and MAE are lower."
        )
    if not rmse_is_lower and not mae_is_lower:
        return (
            "The model performs worse than the global-mean baseline "
            "because both RMSE and MAE are higher."
        )
    return (
        "The model and the global-mean baseline are mixed: "
        "one metric is lower and the other is higher."
    )
