import numpy as np
from numpy.typing import NDArray

RatingPair = tuple[int, float]
TrainRowsByUser = list[list[RatingPair]]
TrainRowsByItem = list[list[RatingPair]]


class BiasedMatrixFactorization:
    def __init__(
        self, rank: int, regularization_lambda: float, iteration_count: int
    ) -> None:
        self.rank = rank
        self.regularization_lambda = regularization_lambda
        self.iteration_count = iteration_count
        self.global_mean_rating = 0.0
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None
        self.training_loss_per_iteration = []


    """Multiply the user and item latent vectors (dot product) and add the biases and the global mean rating."""
    def predict(self, user_index: int, item_index: int) -> float:
        user_vector = self.user_factors[user_index]
        item_vector = self.item_factors[item_index]
        latent_score = np.dot(user_vector, item_vector)
        return (
            self.global_mean_rating
            + self.user_bias[user_index]
            + self.item_bias[item_index]
            + latent_score
        )


    """Loop through the user-item ID pairs and predict the rating for each pair."""
    def predict_ratings_for_pairs(
        self,
        user_indices: NDArray[int],
        item_indices: NDArray[int],
    ) -> NDArray[float]:

        # 1x2067 in our train-test split
        predictions = np.zeros(len(user_indices))

        for pair_index in range(len(user_indices)):
            predictions[pair_index] = self.predict(
                user_indices[pair_index], item_indices[pair_index]
            )
        return predictions


    def fit(
        self,
        train_user_indices: NDArray[int],
        train_item_indices: NDArray[int],
        train_ratings: NDArray[float],
    ) -> None:
        # we start from index 0
        user_count = train_user_indices.max() + 1
        item_count = train_item_indices.max() + 1

        # Initialize the global mean rating as we need it for the predictions and later calculations.
        # Also, initialize the biases as we need it for the predictions.
        self.global_mean_rating = float(train_ratings.mean())
        self.user_bias = np.zeros(user_count)
        self.item_bias = np.zeros(item_count)

        # Create latent user vectors (user=row) with columns=rank and assign random normal values to them.
        # All zeros would make every dot product zero, so users and items would not separate.
        # Small random starts let ALS (later) update each vector in a different direction.
        self.user_factors = np.random.normal(scale=0.01, size=(user_count, self.rank))

        # Create latent item vectors (item=row) with columns=rank and assign random normal values to them.
        # All zeros would make every dot product zero, so users and items would not separate.
        # Small random starts let ALS (later) update each vector in a different direction.
        self.item_factors = np.random.normal(scale=0.01, size=(item_count, self.rank))

        # Create empty lists for each user and item to store the ratings.
        train_rows_by_user = [[] for _ in range(user_count)]
        train_rows_by_item = [[] for _ in range(item_count)]

        # Group the ratings by user and item.
        for row_index in range(len(train_ratings)):
            user_index = train_user_indices[row_index]
            item_index = train_item_indices[row_index]
            rating_value = train_ratings[row_index]

            # Create users list: userIndex: [(itemIndex, ratingValue), ...]
            # Create items list: itemIndex: [(userIndex, ratingValue), ...]
            train_rows_by_user[user_index].append((item_index, rating_value))
            train_rows_by_item[item_index].append((user_index, rating_value))

        # Create a 10x10 identity matrix for the regularization, 1s at the diagonal, 0s remaining.
        # This is used later to penalize the factors to not be too large, and to avoid overfitting.
        identity_matrix = np.eye(self.rank)

        for _ in range(self.iteration_count):
            self._update_user_factors(train_rows_by_user, identity_matrix)
            self._update_item_factors(train_rows_by_item, identity_matrix)
            self._update_bias_terms(train_rows_by_user, train_rows_by_item)
            training_loss = self._compute_training_loss(
                train_user_indices, train_item_indices, train_ratings
            )
            self.training_loss_per_iteration.append(training_loss)


    """
        Here we update latent user vecors via least squares.
        (sum(i) q(i) q(i)^T + lambda*I) p(u) = sum(i) residual(i) * q(i)
        i = one item this user rated
        q(i) = latent vector for item i
        p(u) = latent user vector
        lambda = self.regularization_lambda
        I = identity_matrix (rank x rank)
        residual(i) = rating - global rating mean - user bias - item bias
        matrix_a = left side of the above equation
        vector_b = right side of the above equation
    """
    def _update_user_factors(
        self, train_rows_by_user: TrainRowsByUser, identity_matrix: NDArray[float]
    ) -> None:
        for user_index, user_rows in enumerate(train_rows_by_user):
            if len(user_rows) == 0:
                continue

            # lambda*I, see formula above.
            matrix_a = self.regularization_lambda * identity_matrix

            # See below [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.].
            vector_b = np.zeros(self.rank)

            for item_index, rating_value in user_rows:
                # Get the latent vector for the item this user rated.
                item_vector = self.item_factors[item_index]

                # Calculate the residual rating: rating - global rating mean - user bias - item bias
                residual_rating = (
                    rating_value - self.global_mean_rating - self.user_bias[user_index] - self.item_bias[item_index]
                )

                # Native function for outer product i.e. q q^T matrix, see formula above.
                matrix_a += np.outer(item_vector, item_vector)

                # residual(i) * q(i) for each item this user rated, see formula above.
                vector_b += residual_rating * item_vector

            # Native function that solves A x = b for x
            # A = sum(i) q(i) q(i)^T + lambda*I
            # b = sum(i) residual(i) * q(i)
            # x = p(u)
            # p(u) = (sum(i) q(i) q(i)^T + lambda*I)^{−1} sum(i) residual(i) * q(i)
            self.user_factors[user_index] = np.linalg.solve(matrix_a, vector_b)


    """This function works the same way as the _update_user_factors function, but for the item factors."""
    def _update_item_factors(
        self, train_rows_by_item: TrainRowsByItem, identity_matrix: NDArray[float]
    ) -> None:
        for item_index, item_rows in enumerate(train_rows_by_item):
            if len(item_rows) == 0:
                continue
            matrix_a = self.regularization_lambda * identity_matrix
            vector_b = np.zeros(self.rank)

            for user_index, rating_value in item_rows:
                user_vector = self.user_factors[user_index]
                residual_rating = (
                    rating_value
                    - self.global_mean_rating
                    - self.user_bias[user_index]
                    - self.item_bias[item_index]
                )
                matrix_a += np.outer(user_vector, user_vector)
                vector_b += residual_rating * user_vector
            self.item_factors[item_index] = np.linalg.solve(matrix_a, vector_b)


    def _update_bias_terms(
        self, train_rows_by_user: TrainRowsByUser, train_rows_by_item: TrainRowsByItem
    ) -> None:
        for user_index, user_rows in enumerate(train_rows_by_user):
            if len(user_rows) == 0:
                continue

            bias_residuals = []
            for item_index, rating_value in user_rows:
                bias_residuals.append(
                    rating_value
                    - self.global_mean_rating
                    - self.item_bias[item_index]
                    - np.dot(self.user_factors[user_index], self.item_factors[item_index])
                )
            self.user_bias[user_index] = float(np.mean(bias_residuals))

        for item_index, item_rows in enumerate(train_rows_by_item):
            if len(item_rows) == 0:
                continue

            bias_residuals = []
            for user_index, rating_value in item_rows:
                bias_residuals.append(
                    rating_value
                    - self.global_mean_rating
                    - self.user_bias[user_index]
                    - np.dot(self.user_factors[user_index], self.item_factors[item_index])
                )
            self.item_bias[item_index] = float(np.mean(bias_residuals))

        global_residuals = []

        for user_index, user_rows in enumerate(train_rows_by_user):
            for item_index, rating_value in user_rows:
                global_residuals.append(
                    rating_value
                    - self.user_bias[user_index]
                    - self.item_bias[item_index]
                    - np.dot(
                        self.user_factors[user_index],
                        self.item_factors[item_index],
                    )
                )
        if len(global_residuals) > 0:
            self.global_mean_rating = float(np.mean(global_residuals))


    def _compute_training_loss(
        self,
        train_user_indices: NDArray[int],
        train_item_indices: NDArray[int],
        train_ratings: NDArray[float],
    ) -> float:
        predicted_ratings = self.predict_ratings_for_pairs(
            train_user_indices, train_item_indices
        )
        squared_errors = (train_ratings - predicted_ratings) ** 2
        factor_penalty = self.regularization_lambda * (
            np.sum(self.user_factors**2) + np.sum(self.item_factors**2)
        )
        return float(np.sum(squared_errors) + factor_penalty)
