import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y

def Q_1(train_X, train_y, test_X, test_y, n_learners=250, noise=0):
    ada = AdaBoost(DecisionStump, n_learners)
    ada.fit(train_X, train_y)
    train_misclassification_error = []
    test_misclassification_error = []
    for i in range(1, n_learners + 1):
        test_misclassification_error.append(ada.partial_loss(test_X, test_y, i))
        train_misclassification_error.append(ada.partial_loss(train_X, train_y, i))

    plot_misclassification_error(train_misclassification_error, test_misclassification_error, n_learners, noise)
    return ada, test_misclassification_error, train_misclassification_error


def plot_misclassification_error(train_error, test_error, n_learners, noise):
    # create a array of the number of learners
    x = np.arange(1, n_learners + 1)
    fig  = go.Figure([go.Scatter(x=x, y=train_error, mode='lines', name='Training error')])
    fig.add_trace(go.Scatter(x=x, y=test_error, mode='lines', name='Test error'))
    fig.update_layout(title_text='Misclassification error of AdaBoost', xaxis_title_text='Number of learners',
                       yaxis_title_text='Misclassification error')
    if noise != 0:
        fig.update_layout(title_text=f'Misclassification error of AdaBoost with noise of level {noise}')
    fig.show()
    return fig


def Q_2_decision_boundaries(ada, test_X, test_y, T, lims):
    # T = [5, 50, 100, 250]
    # lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    # using ada model from question 1 plot the decision boundaries obtained from the model
    # using semble up to iteration T use partial_predict to get the decision boundaries
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Number of Learners Decision Boundaries: {num}" for num in T],
                        horizontal_spacing=0.1, vertical_spacing=0.1)

    for i, n_learn in enumerate(T):
        partial_predict = lambda X: ada.partial_predict(X, n_learn)
        fig.add_traces([decision_surface(partial_predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(width=1)))], rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title='AdaBoost Decision Boundaries',
                      xaxis_title='x',
                      yaxis_title='y')
    fig.show()


def Q_3_best_ensemble(ada, test_error ,test_X, test_y, lims):
    best_ensemble = np.argmin(test_error) + 1
    partial_predict = lambda X: ada.partial_predict(X, best_ensemble)
    accuracy = 1 - test_error[best_ensemble - 1]
    fig = go.Figure()
    fig.add_traces([decision_surface(partial_predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                           line=dict(width=1)))])
    fig.update_layout(title=f'Best Ensemble Decision Boundaries is {best_ensemble},'
                            f'abd accuracy of {accuracy}',
                      xaxis_title='x',
                      yaxis_title='y')
    # add the accuracy of the best ensemble

    fig.show()

def Q_4_decision_boundaries_weighted(ada,train_X, train_y,lims, noise=0):
    normalized_weights = 10 * (ada.D_ / np.max(ada.D_))
    fig = go.Figure()
    fig.add_traces([decision_surface(ada.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y, colorscale=[custom[0], custom[-1]],
                                           line=dict(width=1.5, color='black'),
                                           size = normalized_weights))])

    fig.update_layout(title='Decision Boundaries with Weighted Samples', xaxis_title='x', yaxis_title='y')
    if noise != 0:
        fig.update_layout(title=f'Decision Boundaries with Weighted Samples with noise of level: {noise}')
    fig.show()

def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada, test_error, train_error = Q_1(train_X, train_y, test_X, test_y, n_learners,noise)

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    # Question 2: Plotting decision surfaces of AdaBoost

    if noise == 0:
        Q_2_decision_boundaries(ada, test_X, test_y, T,lims)

    # Question 3: Decision surface of best performing ensemble
        Q_3_best_ensemble(ada, test_error, test_X, test_y, lims)


    # Question 4: Decision surface with weighted samples
    Q_4_decision_boundaries_weighted(ada, train_X, train_y, lims, noise)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
