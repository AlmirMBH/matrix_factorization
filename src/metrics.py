import numpy as np
from numpy.typing import NDArray


def root_mean_squared_error(
    true_ratings: NDArray[float], predicted_ratings: NDArray[float]
) -> float:
    squared_errors = (true_ratings - predicted_ratings) ** 2
    return float(np.sqrt(np.mean(squared_errors)))


def mean_absolute_error(
    true_ratings: NDArray[float], predicted_ratings: NDArray[float]
) -> float:
    absolute_errors = np.abs(true_ratings - predicted_ratings)
    return float(np.mean(absolute_errors))


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
