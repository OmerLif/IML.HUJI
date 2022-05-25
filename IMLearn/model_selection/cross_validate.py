from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator



def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # split the x and y data into cv folds
    cv_x_folds = np.array_split(X, cv)
    cv_y_folds = np.array_split(y, cv)
    train_scores = []
    validation_scores = []
    for i in range(cv):
        # create a copy of the estimator
        estimator_copy = deepcopy(estimator)
        # get the training data for this fold
        train_x = np.concatenate(cv_x_folds[:i] + cv_x_folds[i + 1:])
        train_y = np.concatenate(cv_y_folds[:i] + cv_y_folds[i + 1:])
        # fit the estimator to the training data
        estimator_copy.fit(train_x, train_y)
        # get the validation data for this fold
        validation_x = cv_x_folds[i]
        validation_y = cv_y_folds[i]
        # get the predicted values for the validation data and train data
        validation_predictions = estimator_copy.predict(validation_x)
        train_predictions = estimator_copy.predict(train_x)

        validation_score = scoring(validation_y, validation_predictions)
        train_score = scoring(train_y, train_predictions)
        # save the scores
        train_scores.append(train_score)
        validation_scores.append(validation_score)
    # return the average scores
    return np.mean(train_scores), np.mean(validation_scores)
