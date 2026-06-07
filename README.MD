# Movie Recommendation System

This project is a simulation of a movie recommendation system based on biased matrix factorization on explicit ratings. The code is the practical part of the course project and demonstrates how biased matrix factorization works on real rating data. Similar datasets with the same structure can be used when the column names are configured in `settings.py`. The seminar paper *Recommender Systems Using Matrix Factorization* presents the theory, experiments, and results.

---

## 1) Project Background

This is an undergraduate project for the course **Machine Learning: Unsupervised Techniques** at the Faculty of Electrical Engineering, University of Sarajevo. The supervisor is **Professor Nermin Covic**.

The pipeline trains a recommender with biased matrix factorization (BMF) and Alternating Least Squares (ALS) on sparse user–item ratings.

The primary focus is evaluation of the trained model on held-out ratings. Top-item recommendations for a demo user are also exported to support the reported results.

### Accompanying paper

The seminar paper **Recommender Systems Using Matrix Factorization** presents biased matrix factorization theory, experiments, and result discussion. This repository is the practical implementation that shows how the method works on the MovieLens dataset. The LaTeX source is in `paper/seminar_paper.tex` at the project root next to the `datasets/` and `src/` directories.

---

## 2) Technologies

| Library | Role |
|---------|------|
| Python 3 | The implementation language |
| NumPy | Matrix operations and ALS updates |
| pandas | CSV loading, train/test splits, and exports |
| matplotlib | Metric plots |
| pickle | Saving and loading the trained model (standard library) |

All Python dependencies are listed in `requirements.txt`.

---

## 3) Project Setup

### First run

On the first run the app loads the data, remaps ids, splits train and test, trains the model, runs evaluation, writes output files, and opens a plot. This usually takes some time. However, the following runs will be much faster.

### Later runs

On later runs the app reuses `output/train_ratings.csv`, `output/test_ratings.csv`, and `output/trained_model.pkl` when those files already exist.

After a change to `settings.py` or the dataset, the files in `output/` must be deleted before the next run.

### Virtual environment

The virtual environment is created and activated from the `src/` directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
source .venv/bin/activate
python main.py
```

The app waits on `plt.show()` until the plot window is closed.

---

## 4) Project Structure

```
matrix_factorization/
├── paper/
│   └── seminar_paper.tex
├── datasets/
│   ├── ratings.csv
│   ├── movies.csv
│   └── ...
├── output/
└── src/
    ├── main.py
    ├── run.py
    ├── settings.py
    ├── data_ingestion.py
    ├── biased_mf.py
    ├── train_model.py
    ├── evaluate_model.py
    ├── metrics.py
    ├── recommendations.py
    ├── plot_metrics.py
    └── export_outputs.py
```

### Source code readability

The Python modules in `src/` use type annotations on function arguments and return values. Examples include `NDArray[float]` for NumPy arrays, `pd.DataFrame` for tabular data, and named types such as `EvaluationResults` where a function returns a structured result. These annotations clarify what each function expects and produces without changing runtime behavior.

Each function also has a short comment above it. In the core algorithm (`biased_mf.py`) and the data pipeline (`data_ingestion.py`), these comments state the formulas and symbol definitions used in the implementation. Additional inline comments appear inside longer functions where a step is easier to follow with a brief note (for example during ALS factor updates or id remapping).

This documentation is intended to make the implementation easier to read as well as to help locate specific parts of the code while reading the accompanying seminar paper.


---

## 5) Settings File (`settings.py`)

The file `settings.py` holds all paths and hyperparameters. Other modules read values from this file instead of taking them as function arguments.

| Setting | Description |
|---------|-------------|
| `SRC_DIRECTORY` | The path to the `src/` folder |
| `PROJECT_ROOT` | The project root directory |
| `DATASET_RATINGS_PATH` | The input ratings CSV file |
| `DATASET_MOVIES_PATH` | The MovieLens movies metadata CSV (`movies.csv`) |
| `RATINGS_USER_ID_COLUMN` | The user id column name (`userId` on MovieLens) |
| `RATINGS_ITEM_ID_COLUMN` | The item id column name (`movieId` on MovieLens) |
| `RATINGS_VALUE_COLUMN` | The rating column name (`rating` on MovieLens) |
| `OUTPUT_DIRECTORY` | The export folder at project root (`matrix_factorization/output/`) |
| `TRAINED_MODEL_PATH` | The pickled model file |
| `TRAIN_RATINGS_PATH` | The cached train split CSV |
| `TEST_RATINGS_PATH` | The cached test split CSV |
| `LATENT_RANK` | The number of latent dimensions per user and item (default 10) |
| `REGULARIZATION_LAMBDA` | The $L_2$ penalty on factor matrices (default 10.0) |
| `ALS_ITERATION_COUNT` | The number of ALS training rounds (default 40). During development, values from 10 to 70 were tried in steps of 10. |
| `TEST_FRACTION` | The fraction of ratings used for test (default 0.2) |
| `RANDOM_SEED` | The random seed for shuffling before the split and for latent-factor initialization in `fit()` (default 42) |
| `TOP_RECOMMENDATION_COUNT` | The number of items in the demo recommendation file (default 10) |
| `DEMO_USER_INDEX` | The user index used for the demo recommendation file (default 0) |
| `PLOT_EXAMPLE_COUNT` | The number of sample test predictions shown on the plot (default 8) |

---

## 6) Dataset and Data Preparation

### Source

The default dataset is **MovieLens Latest Small** from [GroupLens](http://grouplens.org/datasets/). The file `ratings.csv` is placed in the `datasets/` folder.

The default column names are `userId`, `movieId`, and `rating`.

### Steps

1. The app reads `ratings.csv`.
2. It maps each original user id and item id to dense indices `user_index` and `item_index` that start at 0.
3. For each user, it shuffles that user's ratings with `RANDOM_SEED` and holds out `TEST_FRACTION` for test ($20\%$ per user by default). Users with one rating stay in train only.
4. It writes `train_ratings.csv` and `test_ratings.csv` into `output/`.
5. It trains the model on train rows only.
6. It evaluates the model on test rows only.

On MovieLens small about $1.7\%$ of all user–item pairs have a rating.

---

## 7) Recommender Algorithm Selection

Two approaches were considered for this project. Plain matrix factorization (MF) predicts a rating from latent vectors only. Biased matrix factorization (BMF) also adds the global mean $\mu$, a per-user bias $b_u$, and a per-item bias $b_i$. BMF was selected because MovieLens star ratings follow clear patterns. Some users give higher or lower scores than others. Some movies get higher or lower scores than others. MF would have to learn those patterns inside the latent vectors. BMF models them directly and is commonly used for star ratings. This project trains BMF with alternating least squares (ALS).

### Matrix Factorization (MF)

Matrix factorization predicts a rating as the dot product of two latent vectors.

$$
\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i
$$

Here $\mathbf{p}_u$ is the latent vector for user $u$ and $\mathbf{q}_i$ is the latent vector for item $i$.

### Biased Matrix Factorization (BMF)

Biased matrix factorization adds a global mean and per-user and per-item biases.

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i
$$

| Term | Meaning |
|------|---------|
| $\mu$ | The global mean rating computed on train data |
| $b_u$ | How far user $u$ tends to rate above or below $\mu$ |
| $b_i$ | How far item $i$ tends to be rated above or below $\mu$ |
| $\mathbf{p}_u^\top \mathbf{q}_i$ | The latent dot product for user $u$ and item $i$ |

The user bias $b_u$ is how much user $u$ usually rates above or below $\mu$. The item bias $b_i$ is how much item $i$ is usually rated above or below $\mu$. The latent dot product $\mathbf{p}_u^\top \mathbf{q}_i$ is the remaining fit between that user and that item.

Matrix factorization uses only latent vectors. Biased matrix factorization also uses $\mu$, $b_u$, and $b_i$. This project implements BMF.

### Latent vectors

Each user gets one latent vector $\mathbf{p}_u$ and each item gets one latent vector $\mathbf{q}_i$. The length of each vector is set by `LATENT_RANK` (default $10$).

At the start of training, every vector is filled with small random values from a normal distribution. All-zero vectors would keep every dot product at zero, so random starts are required.

During ALS, each vector is updated from the train ratings. User vectors are refitted while item vectors stay fixed, then item vectors are refitted while user vectors stay fixed. After `ALS_ITERATION_COUNT` rounds, the vectors in `user_factors` and `item_factors` are the learned values stored in `trained_model.pkl`.

### Training (ALS)

Each ALS iteration performs three steps. First it updates user factors while item factors stay fixed. Then it updates item factors while user factors stay fixed. Then it updates the bias terms. After these three passes, the code records the training objective on the train split.

Only the latent factor matrices are regularized. Biases are not penalized.

The training objective is

$$
\sum_{(u,i) \in \text{train}} (r_{ui} - \hat{r}_{ui})^2 + \lambda \left( \|\mathbf{P}\|_F^2 + \|\mathbf{Q}\|_F^2 \right)
$$

The training loss plot shows this objective after each iteration. It is the sum of squared errors plus the regularization penalty.

---

## 8) Model

The model class is `BiasedMatrixFactorization` in `biased_mf.py`. The class is implemented in this project and is not imported from an external library.

Biased matrix factorization is used for explicit star ratings. ALS updates each user and item vector with a regularized least-squares step. The bias terms help the model outperform a predictor that always outputs the global mean.

### Saved model artifact

The main output of training is `output/trained_model.pkl`. It stores the full trained `BiasedMatrixFactorization` object through Python pickle. Evaluation, plotting, and recommendations load this file to predict ratings without retraining.

After training, the pickle contains the learned parameters below.

| Field | Description |
|-------|-------------|
| `global_mean_rating` | The average rating on the train split ($\mu$) |
| `user_bias` | How much each user usually rates above or below $\mu$ |
| `item_bias` | How much each item is usually rated above or below $\mu$ |
| `user_factors` | Latent vector matrix for all users ($610 \times 10$ by default) |
| `item_factors` | Latent vector matrix for all items ($9{,}724 \times 10$ by default) |
| `training_loss_per_iteration` | Training objective value after each ALS round |
| `rank`, `regularization_lambda`, `iteration_count` | Hyperparameters used during training |

A prediction for user index $u$ and item index $i$ is computed as $\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$ using the values stored in the pickle. The file does not contain raw MovieLens ids or individual rating rows.

---

## 9) Evaluation

All evaluation metrics are computed on the held-out test split.

### RMSE

Root mean squared error is defined as

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (r_i - \hat{r}_i)^2}
$$

### MAE

Mean absolute error is defined as

$$
\mathrm{MAE} = \frac{1}{N} \sum_{i=1}^{N} |r_i - \hat{r}_i|
$$

$\mathrm{RMSE}$ penalizes large errors more because errors are squared. $\mathrm{MAE}$ reports the average absolute error in rating units.

### Baseline

The baseline predictor sets every test rating to the train global mean $\mu$. The model $\mathrm{RMSE}$ and $\mathrm{MAE}$ are compared to the baseline $\mathrm{RMSE}$ and $\mathrm{MAE}$.

### Output files

Each run writes files to `output/`.

| File | Role |
|------|------|
| `trained_model.pkl` | The fitted `BiasedMatrixFactorization` object |
| `train_ratings.csv` | Cached train split |
| `test_ratings.csv` | Cached test split |
| `metrics.json` | Dataset counts, hyperparameters, test and baseline metrics |
| `test_predictions.csv` | True and predicted ratings on the test split |
| `loss_per_iter.csv` | Training objective per ALS iteration |
| `metrics_plots.png` | Summary figure with metrics, training loss, settings, and sample test predictions with movie titles from `movies.csv` |
| `top{N}_user_{id}.csv` | Top recommended items for the demo user |

Only `trained_model.pkl` holds the fitted recommender. The CSV and JSON files are splits or reports.

---

## 10) Results

The table below shows one of the runs on MovieLens small with default settings and random seed 42 (user 0).

| Metric | Value |
|--------|-------|
| Users | 610 |
| Items | 9,724 |
| Ratings | 100,836 |
| Matrix density | $1.7\%$ |
| Train / test ratings | $80{,}896$ / $19{,}940$ |
| Test $\mathrm{RMSE}$ | $0.89$ |
| Test $\mathrm{MAE}$ | $0.67$ |
| Baseline $\mathrm{RMSE}$ | $1.05$ |
| Baseline $\mathrm{MAE}$ | $0.83$ |

The test $\mathrm{RMSE}$ and $\mathrm{MAE}$ are lower than the baseline values. The training loss decreases from about $51{,}200$ to about $42{,}200$ over $40$ ALS iterations.

---

## 11) Known Limitations

The train-test split is done per user. Every user in the test split also has at least one rating in train.

An item can still have no train rows if all of its ratings land in test splits. ALS never updates that item's latent vector, so predictions involving that item are unreliable. This edge case is called a cold start, and requires other approaches to be solved, e.g. content-based clustering, which is out of the scope of this project.

Hyperparameters are fixed in `settings.py` and are not searched automatically. During development, ALS iteration counts from 10 to 70 were tried in steps of 10. Test RMSE changed only minimally beyond 40.

Recommendation scores are the model's raw prediction outputs and are not clipped to the 1--5 MovieLens rating range. For top-$N$ ranking, only item order matters.

This is a batch script written for coursework. It is not set up as a deployed recommendation service.

---

## 12) Recommendations

Recommendations use the saved model in `trained_model.pkl`.

The function `top_item_recommendations(user_index, top_count)` in `recommendations.py` performs the following steps.

1. The trained model is loaded from `trained_model.pkl`.
2. The train split is loaded from `train_ratings.csv`.
3. All item indices that the chosen user already rated in train are collected and excluded.
4. For every remaining item index, the model predicts a rating with $\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i$.
5. Items are sorted by predicted rating from highest to lowest.
6. Each `item_index` is mapped to `movieId` from the train split and joined with `movies.csv` to attach movie titles.
7. The first `top_count` items are written to `output/top{top_count}_user_{user_index}.csv` with columns `item_index`, `movie_title`, and `predicted_rating`.

The default demo uses `user_index` 0 and `top_count` 10 from `settings.py`. Each row in the output file is a movie the user has not rated in train, ranked by the model's raw prediction scores. Values above 5.0 can appear because the model is not constrained to the MovieLens star range.

---

## 13) Application Flow

```
main.py
  └── run()
        ├── load_or_create_train_test_split()
        ├── load_or_train_model()
        ├── evaluate_model(model, test_df)
        ├── top_item_recommendations(user, n)
        ├── save metrics and CSVs
        ├── print metrics
        └── plot_run_metrics()
```

The function `main()` calls `run()`. The function `run()` loads or creates the train and test CSV files, loads or trains the model, evaluates on test data, builds a demo recommendation list, saves outputs, prints metrics, and draws the summary plot. The recommendation steps are described in section 12.

---

## Contact

Contacts for questions and collaboration are [**Almir Mustafic**](https://github.com/AlmirMBH) and [**Merim Jusufbegovic**](https://github.com/MerimJusufbegovic).

---

## License

This project is released under the MIT License.

---

## Acknowledgments

The MovieLens dataset comes from [GroupLens Research](http://grouplens.org/datasets/) and is licensed under CC BY 4.0. The project uses pandas (BSD), NumPy (BSD), and matplotlib (BSD-compatible license).

**Professor Nermin Covic** supervised the course project.

---

## Disclaimer

The MovieLens dataset is used for coursework and simulations only. This project was completed for **Machine Learning: Unsupervised Techniques**. It is not intended as a production recommendation system.
