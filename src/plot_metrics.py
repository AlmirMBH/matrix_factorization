import matplotlib.pyplot as plt
import pandas as pd
from metrics import baseline_comparison_message
from settings import OUTPUT_DIRECTORY


def _truncate_title(title: str, max_length: int = 40) -> str:
    if len(title) <= max_length:
        return title
    return title[: max_length - 3] + "..."


def plot_run_metrics(
    metrics: dict[str, float | int],
    loss_per_iteration_dataframe: pd.DataFrame,
    example_predictions_dataframe: pd.DataFrame,
) -> None:
    plt.close("all")
    figure = plt.figure(figsize=(12, 11), num="run_metrics")
    grid = figure.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])
    axes = [
        [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])],
        [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
    ]
    table_axis = figure.add_subplot(grid[2, :])
    figure.suptitle("Run metrics")

    evaluation_metric_names = [
        "rmse_test",
        "mae_test",
        "rmse_baseline_global_mean",
        "mae_baseline_global_mean",
    ]
    evaluation_metric_values = [metrics[name] for name in evaluation_metric_names]
    axes[0][0].bar(evaluation_metric_names, evaluation_metric_values)
    axes[0][0].set_title("Test evaluation")
    axes[0][0].set_ylabel("Error")
    axes[0][0].tick_params(axis="x", rotation=45)

    dataset_metric_names = [
        "user_count",
        "item_count",
        "rating_count",
        "train_rating_count",
        "test_rating_count",
    ]
    dataset_metric_values = [metrics[name] for name in dataset_metric_names]
    axes[0][1].bar(dataset_metric_names, dataset_metric_values)
    axes[0][1].set_title("Dataset")
    axes[0][1].tick_params(axis="x", rotation=45)

    axes[1][0].plot(
        loss_per_iteration_dataframe["iteration"],
        loss_per_iteration_dataframe["training_loss"],
    )
    axes[1][0].set_title("Training loss per ALS iteration")
    axes[1][0].set_xlabel("iteration")
    axes[1][0].set_ylabel("training loss")

    hyperparameter_lines = [
        f"latent_rank: {metrics['latent_rank']}",
        f"regularization_lambda: {metrics['regularization_lambda']}",
        f"als_iteration_count: {metrics['als_iteration_count']}",
        f"rated_user_item_pairs_ratio_percent: {metrics['rated_user_item_pairs_ratio_percent']}",
    ]
    axes[1][1].axis("off")
    axes[1][1].set_title("Hyperparameters and sparsity")
    axes[1][1].text(0, 1, "\n".join(hyperparameter_lines), va="top", fontsize=11)

    display_dataframe = example_predictions_dataframe.copy()
    display_dataframe["movie_title"] = display_dataframe["movie_title"].map(_truncate_title)
    table_rows = [
        [
            str(row["user_id"]),
            row["movie_title"],
            f"{row['true_rating']:.2f}",
            f"{row['predicted_rating']:.2f}",
        ]
        for _, row in display_dataframe.iterrows()
    ]
    table_axis.axis("off")
    table_axis.set_title("Sample test predictions")
    prediction_table = table_axis.table(
        cellText=table_rows,
        colLabels=list(display_dataframe.columns),
        loc="upper center",
        cellLoc="center",
    )
    prediction_table.auto_set_font_size(False)
    prediction_table.set_fontsize(9)
    prediction_table.scale(1, 1.3)
    table_axis.text(
        0.5,
        0.02,
        baseline_comparison_message(
            float(metrics["rmse_test"]),
            float(metrics["mae_test"]),
            float(metrics["rmse_baseline_global_mean"]),
            float(metrics["mae_baseline_global_mean"]),
        ),
        transform=table_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        wrap=True,
    )

    for axis in (axes[0][0], axes[0][1]):
        for tick_label in axis.get_xticklabels():
            tick_label.set_ha("right")

    figure.tight_layout()
    figure.subplots_adjust(bottom=0.08, hspace=0.9, top=0.94)
    output_directory = OUTPUT_DIRECTORY
    metrics_plot_path = output_directory / "metrics_plots.png"
    figure.savefig(metrics_plot_path)
    plt.show()
