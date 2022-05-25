from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import kfold_cross_validation as kfold
from sklearn.model_selection import KFold


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
    x =  np.array(np.linspace(-1.2, 2, n_samples))
    clean_y = model(x)
    noise_data = np.random.normal(loc=0, scale=noise, size=len(clean_y))
    dirty_y = clean_y + noise_data
    # split into training and testing portions (2/3 for training, 1/3 for testing)
    test_x_clean, test_y_clean, train_x_clean, train_y_clean = Q_1_plot_clean_data(clean_y, x)
    test_x_dirty, test_y_dirty, train_x_dirty, train_y_dirty = Q_1_plot_clean_data(dirty_y, x)
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors_clean = validate_errors_clean = []
    if noise == 0:
        train_errors_clean, validate_errors_clean = Q_2_poly_over_clean(train_x_clean, train_y_clean)

    # split into training and testing portions (2/3 for training, 1/3 for testing)
    test_x_dirty, test_y_dirty, train_errors_dirty, train_x_dirty, train_y_dirty, validate_errors_dirty = Q_2_poly_over_dirty(
        dirty_y, x)
    # best degree is

    test_results_over_best_fit(test_x_clean, test_x_dirty, test_y_clean, test_y_dirty, train_errors_clean,
                               train_errors_dirty, train_x_clean, train_x_dirty, train_y_clean, train_y_dirty,
                               validate_errors_clean, validate_errors_dirty)

def Practical_part_1():
    select_polynomial_degree()
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)


def test_results_over_best_fit(test_x_clean, test_x_dirty, test_y_clean, test_y_dirty, train_errors_clean,
                               train_errors_dirty, train_x_clean, train_x_dirty, train_y_clean, train_y_dirty,
                               validate_errors_clean, validate_errors_dirty):
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
    print(f"Mean square error for dirty data is {mean_square_error(test_y_pred_dirty, test_y_dirty)}")
    print(f"Mean square error for clean data is {mean_square_error(test_y_pred_clean, test_y_clean)}")

def Q_2_poly_over_dirty(dirty_y, x):
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
    fig.update_layout(title_text='Polynomial fitting over dirty data', xaxis_title_text='Degree',
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


def Q_1_plot_clean_data(clean_y, x):
    train_x_clean, train_y_clean, test_x_clean, test_y_clean = split_train_test(x, clean_y, 0.667)
    train_x_clean = train_x_clean.flatten()
    test_x_clean = test_x_clean.flatten()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=train_x_clean, y=train_y_clean, mode='markers', name='Training data', marker_color='blue'))
    fig.add_trace(go.Scatter(x=test_x_clean, y=test_y_clean, mode='markers', name='Test data', marker_color='red'))
    fig.update_layout(title='Training and test data', xaxis_title='x', yaxis_title='y')
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
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # ()
    raise NotImplementedError()
