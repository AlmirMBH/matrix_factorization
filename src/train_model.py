import pickle
import pandas as pd
from biased_mf import BiasedMatrixFactorization
from settings import (
    ALS_ITERATION_COUNT,
    LATENT_RANK,
    RATINGS_VALUE_COLUMN,
    REGULARIZATION_LAMBDA,
    TRAINED_MODEL_PATH,
    TRAIN_RATINGS_PATH,
)


"""
    Initialize the model with the following parameters.
    rank=10 is latent dims per user/item (10-50 is common).
    lambda=10 is stronger regularization. Low lambda overfit. Test RMSE beat baseline only after we raised it.
    iterations=40 is ALS rounds with lambda=10. More iterations did not help test RMSE much.
"""
def train_model() -> BiasedMatrixFactorization:
    train_dataframe = pd.read_csv(TRAIN_RATINGS_PATH)
    train_user_indices = train_dataframe["user_index"].to_numpy()
    train_item_indices = train_dataframe["item_index"].to_numpy()
    train_ratings = train_dataframe[RATINGS_VALUE_COLUMN].to_numpy()
    model = BiasedMatrixFactorization(LATENT_RANK, REGULARIZATION_LAMBDA, ALS_ITERATION_COUNT)
    model.fit(train_user_indices, train_item_indices, train_ratings)
    return model


def save_model(model: BiasedMatrixFactorization) -> None:
    TRAINED_MODEL_PATH.parent.mkdir(exist_ok=True)
    with TRAINED_MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)


def load_model() -> BiasedMatrixFactorization:
    with TRAINED_MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


"""Load the model if it exists, otherwise train a new model and save it."""
def load_or_train_model() -> BiasedMatrixFactorization:
    if TRAINED_MODEL_PATH.exists():
        return load_model()
    model = train_model()
    save_model(model)
    return model


"""Format the training loss into a dataframe e.g. iteration 1: 0.1, iteration 2: 0.09, etc."""
def training_loss_dataframe(model: BiasedMatrixFactorization) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iteration": range(1, len(model.training_loss_per_iteration) + 1),
            "training_loss": model.training_loss_per_iteration,
        }
    )
