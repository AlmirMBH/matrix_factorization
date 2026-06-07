from data_ingestion import load_or_create_train_test_split
from export_outputs import save_csv, save_json
from recommendations import top_item_recommendations
from evaluate_model import evaluate_model, plot_example_predictions_dataframe, test_predictions_dataframe
from train_model import load_or_train_model, training_loss_dataframe
from plot_metrics import RunMetrics, plot_run_metrics
from settings import (
    ALS_ITERATION_COUNT,
    DEMO_USER_INDEX,
    LATENT_RANK,
    OUTPUT_DIRECTORY,
    REGULARIZATION_LAMBDA,
    TOP_RECOMMENDATION_COUNT,
)


"""
    Run the full recommendation pipeline from data load through export and plot.
    1. Load or create train/test CSVs (load_or_create_train_test_split)
    2. Load or train the model (load_or_train_model)
    3. Evaluate on the test split (evaluate_model)
    4. Build demo top-N recommendations for DEMO_USER_INDEX
    5. Save metrics, predictions, loss, and recommendation CSVs to output/
    6. Print key metrics and show the summary plot (plot_run_metrics)
"""
def run() -> None:
    output_directory = OUTPUT_DIRECTORY
    output_directory.mkdir(exist_ok=True)

    (train_dataframe, test_dataframe, user_count, item_count, rating_count, rated_user_item_pairs_ratio,
    ) = load_or_create_train_test_split()

    model = load_or_train_model()
    evaluation_results = evaluate_model(model, test_dataframe)
    top_recommendations_dataframe = top_item_recommendations(DEMO_USER_INDEX, TOP_RECOMMENDATION_COUNT)

    metrics: RunMetrics = {
        "user_count": user_count,
        "item_count": item_count,
        "rating_count": rating_count,
        "rated_user_item_pairs_ratio_percent": round(rated_user_item_pairs_ratio * 100, 2),
        "train_rating_count": len(train_dataframe),
        "test_rating_count": len(test_dataframe),
        "latent_rank": LATENT_RANK,
        "regularization_lambda": REGULARIZATION_LAMBDA,
        "als_iteration_count": ALS_ITERATION_COUNT,
        "rmse_test": round(evaluation_results["test_rmse"], 4),
        "mae_test": round(evaluation_results["test_mae"], 4),
        "rmse_baseline_global_mean": round(evaluation_results["baseline_rmse"], 4),
        "mae_baseline_global_mean": round(evaluation_results["baseline_mae"], 4),
    }

    test_predictions_output = test_predictions_dataframe(evaluation_results)
    example_predictions_output = plot_example_predictions_dataframe(
        test_dataframe, evaluation_results
    )
    loss_per_iteration_output = training_loss_dataframe(model)

    save_csv(output_directory, "test_predictions.csv", test_predictions_output)
    save_csv(output_directory, "loss_per_iter.csv", loss_per_iteration_output)
    save_json(output_directory, "metrics.json", metrics)
    save_csv(
        output_directory,
        f"top{TOP_RECOMMENDATION_COUNT}_user_{DEMO_USER_INDEX}.csv",
        top_recommendations_dataframe,
    )

    print("user_count:", metrics["user_count"])
    print("item_count:", metrics["item_count"])
    print("rating_count:", metrics["rating_count"])
    print("rated_user_item_pairs_ratio_percent:", metrics["rated_user_item_pairs_ratio_percent"])
    print("train_rating_count:", metrics["train_rating_count"])
    print("test_rating_count:", metrics["test_rating_count"])
    print("rmse_test:", metrics["rmse_test"])
    print("mae_test:", metrics["mae_test"])
    print("rmse_baseline_global_mean:", metrics["rmse_baseline_global_mean"])
    print("mae_baseline_global_mean:", metrics["mae_baseline_global_mean"])
    plot_run_metrics(metrics, loss_per_iteration_output, example_predictions_output)