from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso, Ridge

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    model = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    # sample n_samples in uniform distribution between [-1.2,2]
    x = np.array(np.linspace(-1.2, 2, n_samples))
    clean_y = model(x)
    noise_data = np.random.normal(loc=0, scale=noise, size=len(clean_y))
    dirty_y = clean_y + noise_data
    # split into training and testing portions (2/3 for training, 1/3 for testing)
    test_x_clean, test_y_clean, train_x_clean, train_y_clean = Q_1_plot_data(clean_y, x, noise)
    test_x_dirty, test_y_dirty, train_x_dirty, train_y_dirty = Q_1_plot_data(dirty_y, x, noise)
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors_clean = validate_errors_clean = []
    if noise == 0:
        train_errors_clean, validate_errors_clean = Q_2_poly_over_clean(train_x_clean, train_y_clean)

    # split into training and testing portions (2/3 for training, 1/3 for testing)
    test_x_dirty, test_y_dirty, train_errors_dirty, train_x_dirty, train_y_dirty, validate_errors_dirty = Q_2_poly_over_dirty(
        dirty_y, x, noise)
    # best degree is

    test_results_over_best_fit(test_x_clean, test_x_dirty, test_y_clean, test_y_dirty, train_errors_clean,
                               train_errors_dirty, train_x_clean, train_x_dirty, train_y_clean, train_y_dirty,
                               validate_errors_clean, validate_errors_dirty, noise)


def Practical_part_1():
    select_polynomial_degree()
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)


def test_results_over_best_fit(test_x_clean, test_x_dirty, test_y_clean, test_y_dirty, train_errors_clean,
                               train_errors_dirty, train_x_clean, train_x_dirty, train_y_clean, train_y_dirty,
                               validate_errors_clean, validate_errors_dirty, noise=5):
    print(train_errors_dirty)
    print(validate_errors_dirty)
    print(train_errors_clean)
    print(validate_errors_clean)
    best_degree_dirty = np.argmin(validate_errors_dirty)
    best_degree_clean = 0
    if validate_errors_clean:
        best_degree_clean = np.argmin(validate_errors_clean)
    print(f"Best degree for dirty data is {best_degree_dirty}", f"Best degree for clean data is {best_degree_clean}")
    # fit a polynoimal model with the best degree and plot the mean square error results
    poly_dirty = PolynomialFitting(best_degree_dirty)
    poly_clean = PolynomialFitting(best_degree_clean)
    poly_dirty.fit(train_x_dirty, train_y_dirty)
    poly_clean.fit(train_x_clean, train_y_clean)
    # predict the test data
    test_y_pred_dirty = poly_dirty.predict(test_x_dirty)
    test_y_pred_clean = poly_clean.predict(test_x_clean)
    # present the error results
    print(f"Mean square error for dirty data is {mean_square_error(test_y_pred_dirty, test_y_dirty)} noise level is {noise}")
    print(f"Mean square error for clean data is {mean_square_error(test_y_pred_clean, test_y_clean)}")


def Q_2_poly_over_dirty(dirty_y, x, noise=5):
    train_x_dirty, train_y_dirty, test_x_dirty, test_y_dirty = split_train_test(x, dirty_y, 0.667)
    train_x_dirty = train_x_dirty.flatten()
    test_x_dirty = test_x_dirty.flatten()
    train_errors_dirty = []
    validate_errors_dirty = []
    for degree in range(0, 10):
        # Create a polynomial fitting object
        train_error, validate_error = cross_validate(PolynomialFitting(degree), train_x_dirty, train_y_dirty,
                                                     mean_square_error, cv=5)
        train_errors_dirty.append(train_error)
        validate_errors_dirty.append(validate_error)
    # plot the training and validation errors
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, 10), y=train_errors_dirty, mode='lines+markers', name='Training error'))
    fig.add_trace(
        go.Scatter(x=np.arange(0, 10), y=validate_errors_dirty, mode='lines+markers', name='Validation error'))
    # add axes titles and graph title
    fig.update_layout(title_text=f'Polynomial fitting over dirty data with noise level {noise}', xaxis_title_text='Degree',
                      yaxis_title_text='Error')
    fig.show()
    return test_x_dirty, test_y_dirty, train_errors_dirty, train_x_dirty, train_y_dirty, validate_errors_dirty


def Q_2_poly_over_clean(train_x_clean, train_y_clean):
    train_errors_clean = []
    validate_errors_clean = []
    for degree in range(0, 10):
        # Create a polynomial fitting object
        train_error, validate_error = cross_validate(PolynomialFitting(degree), train_x_clean, train_y_clean,
                                                     mean_square_error, cv=5)
        train_errors_clean.append(train_error)
        validate_errors_clean.append(validate_error)
    # plot the training and validation errors
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, 10), y=train_errors_clean, mode='lines+markers', name='Training error'))
    fig.add_trace(
        go.Scatter(x=np.arange(0, 10), y=validate_errors_clean, mode='lines+markers', name='Validation error'))
    # add axes titles and graph title
    fig.update_layout(title_text='Polynomial fitting over clean data', xaxis_title_text='Degree',
                      yaxis_title_text='Error')
    fig.show()
    return train_errors_clean, validate_errors_clean


def Q_1_plot_data(clean_y, x, noise=0):
    train_x_clean, train_y_clean, test_x_clean, test_y_clean = split_train_test(x, clean_y, 0.667)
    train_x_clean = train_x_clean.flatten()
    test_x_clean = test_x_clean.flatten()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=train_x_clean, y=train_y_clean, mode='markers', name='Training data', marker_color='blue'))
    fig.add_trace(go.Scatter(x=test_x_clean, y=test_y_clean, mode='markers', name='Test data', marker_color='red'))
    fig.update_layout(title=f'Training and test data with noise level of {noise}', xaxis_title='x', yaxis_title='y')
    fig.show()
    return test_x_clean, test_y_clean, train_x_clean, train_y_clean


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X_test, X_train, l1_ratios, y_test, y_train = load_data(n_evaluations, n_samples)
    # use cross_validate to evaluate the performance of Ridge and Lasso regression for each regularization parameter
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    train_errors_lasso, train_errors_ridge, validate_errors_lasso, validate_errors_ridge = hyper_parameters_eval(
        X_train, l1_ratios, y_train)

    # Question 8 - Plot the training and validation errors for Ridge and Lasso regressions for each regularization parameter
    plot_lasso_ridge_errors(l1_ratios, train_errors_lasso, train_errors_ridge, validate_errors_lasso,
                            validate_errors_ridge)
    compare_regression_with_best_lambda(X_test, X_train, l1_ratios, validate_errors_lasso, validate_errors_ridge,
                                        y_test, y_train)


def compare_regression_with_best_lambda(X_test, X_train, l1_ratios, validate_errors_lasso, validate_errors_ridge,
                                        y_test, y_train):
    print(f'Best lambda for Ridge: {l1_ratios[np.argmin(validate_errors_ridge)]}')
    print(f'Best lambda for Lasso: {l1_ratios[np.argmin(validate_errors_lasso)]}')
    # fit the model with the best regularization parameter over the training data for both Ridge and Lasso and Linear Regression
    ridge_model = Ridge(alpha=l1_ratios[np.argmin(validate_errors_ridge)])
    ridge_model.fit(X_train, y_train)
    lasso_model = Lasso(alpha=l1_ratios[np.argmin(validate_errors_lasso)])
    lasso_model.fit(X_train, y_train)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    # print the mean squared error for the Ridge, Lasso and Linear Regression models
    print(f'Mean squared error for Ridge: {mean_square_error(y_test, ridge_model.predict(X_test))}')
    print(f'Mean squared error for Lasso: {mean_square_error(y_test, lasso_model.predict(X_test))}')
    print(f'Mean squared error for Linear Regression: {mean_square_error(y_test, linear_model.predict(X_test))}')


def plot_lasso_ridge_errors(l1_ratios, train_errors_lasso, train_errors_ridge, validate_errors_lasso,
                            validate_errors_ridge):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=l1_ratios, y=train_errors_ridge, mode='lines+markers', name='Training error - Ridge'))
    fig.add_trace(
        go.Scatter(x=l1_ratios, y=validate_errors_ridge, mode='lines+markers', name='Validation error - Ridge'))
    fig.add_trace(go.Scatter(x=l1_ratios, y=train_errors_lasso, mode='lines+markers', name='Training error - Lasso'))
    fig.add_trace(
        go.Scatter(x=l1_ratios, y=validate_errors_lasso, mode='lines+markers', name='Validation error- Lasso'))
    # add axes titles and graph title
    fig.update_layout(title_text='Training and validation errors for Ridge and Lasso regressions',
                      xaxis_title_text='Lambda',
                      yaxis_title_text='Error')
    fig.show()


def hyper_parameters_eval(X_train, l1_ratios, y_train):
    train_errors_ridge, validate_errors_ridge = [], []
    train_errors_lasso, validate_errors_lasso = [], []
    for lam in l1_ratios:
        # Question 6 - Use cross_validate to evaluate the performance of Ridge and Lasso regression for each regularization parameter
        ridge_model = RidgeRegression(lam=lam, include_intercept=True)
        train_error_ridge, validate_error_ridge = cross_validate(ridge_model, X_train, y_train, mean_square_error, cv=5)
        train_errors_ridge.append(train_error_ridge)
        validate_errors_ridge.append(validate_error_ridge)
        lasso_model = Lasso(alpha=lam)
        train_error_lasso, validate_error_lasso = cross_validate(lasso_model, X_train, y_train, mean_square_error, cv=5)
        train_errors_lasso.append(train_error_lasso)
        validate_errors_lasso.append(validate_error_lasso)
    return train_errors_lasso, train_errors_ridge, validate_errors_lasso, validate_errors_ridge


def load_data(n_evaluations, n_samples):
    X, y = datasets.load_diabetes(return_X_y=True)
    # choose the first n_samples samples as training data
    X_train, y_train = X[:n_samples], y[:n_samples]
    # choose the remaining samples as testing data
    X_test, y_test = X[n_samples:], y[n_samples:]
    # Question 6 - Create a list of regularization parameter values to evaluate
    l1_ratios = np.linspace(0.005, 1, n_evaluations)
    return X_test, X_train, l1_ratios, y_test, y_train


if __name__ == '__main__':
    np.random.seed(0)
    Practical_part_1()
    select_regularization_parameter()
