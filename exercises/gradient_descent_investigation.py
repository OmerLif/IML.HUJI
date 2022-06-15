import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, accuracy_score


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []
    def callback(model=GradientDescent, **kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])
    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    fig_l1 = go.Figure()
    fig_l2 = go.Figure()
    for eta in etas[2:]:
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
        gd_l1 = GradientDescent(learning_rate=FixedLR(eta), tol=1e-5, max_iter=1000, callback=callback_l1)
        weights_l1.append(gd_l1.fit(L1(np.copy(init)), init, init))
        fig_1 = plot_descent_path(L1, np.array(weights_l1), title=f"eta={eta}")
        fig_l1.add_trace(go.Scatter(x=np.arange(len(values_l1)), y=values_l1, mode="lines", name=f"L1 norm with learning rate of:eta={eta}"))
        fig_1.show()
        gd_l2 = GradientDescent(learning_rate=FixedLR(eta), tol=1e-5, max_iter=1000, callback=callback_l2)
        weights_l2.append(gd_l2.fit(L2(np.copy(init)),init, init))
        fig_2 = plot_descent_path(L2, np.array(weights_l2), title=f"L2 norm with learning rate of: eta={eta}")
        fig_2.show()
        fig_l2.add_trace(go.Scatter(x=np.arange(len(values_l2)), y=values_l2, mode="lines", name=f"eta={eta}"))
        print(f" min L_1 with decay rate of {eta} with value: {np.min(values_l1)}")
        print(f" min L_2 with decay rate of {eta} with value: {np.min(values_l2)}")
    # update the layout of the figure
    fig_l1.update_layout(title="Fixed Learning Rate of L1 norm", xaxis_title="Iteration", yaxis_title="Objective Value")
    fig_l2.update_layout(title="Fixed Learning Rate of L2 norm", xaxis_title="Iteration", yaxis_title="Objective Value")
    fig_l1.show()
    fig_l2.show()






def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    # Plot the convergence rates for all decay-rate values in a single plot
    fig_l1 = go.Figure()
    fig_l2 = go.Figure()
    for gamma in gammas:
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
        gd_l1 = GradientDescent(learning_rate=ExponentialLR(eta, gamma), tol=1e-5, max_iter=1000, callback=callback_l1)
        weights_l1.append(gd_l1.fit(L1(np.copy(init)), init, init))
        fig_l1.add_trace(go.Scatter(x=np.arange(len(values_l1)), y=values_l1 ,name=f"gamma={gamma}"))
        gd_l2 = GradientDescent(learning_rate=ExponentialLR(eta, gamma), tol=1e-5, max_iter=1000, callback=callback_l2)
        weights_l2.append(gd_l2.fit(L2(np.copy(init)), init, init))
        fig_l2.add_trace(go.Scatter(x=np.arange(len(values_l2)), y=values_l2, name=f"gamma={gamma}"))
        print(f" min L_1 with decay rate of {gamma} with value: {np.min(values_l1)}")
        print(f" min L_2 with decay rate of {gamma} with value: {np.min(values_l2)}")
    # update the layout of the figure
    fig_l1.update_layout(title="Exponential Decay Rates for L1", xaxis_title="Iteration", yaxis_title="Objective Value")
    fig_l2.update_layout(title="Exponential Decay Rates for L2", xaxis_title="Iteration", yaxis_title="Objective Value")
    fig_l1.show()
    fig_l2.show()

    # Plot descent path for gamma=0.95
    callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
    callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
    gd_l1 = GradientDescent(learning_rate=ExponentialLR(eta, .95), tol=1e-5, max_iter=1000, callback=callback_l1)
    weights_l1.append(gd_l1.fit(L1(np.copy(init)), init, init))
    fig_1 = plot_descent_path(L1, np.array(weights_l1), title=f"gamma=0.95 L1 norm")
    fig_1.show()
    gd_l2 = GradientDescent(learning_rate=ExponentialLR(eta, .95), tol=1e-5, max_iter=1000, callback=callback_l2)
    weights_l2.append(gd_l2.fit(L2(np.copy(init)), init, init))
    fig_2 = plot_descent_path(L2, np.array(weights_l2), title=f"gamma=0.95 L2 norm")
    fig_2.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Fit logistic regression model over the training set, Use the predict_proba to
    # plot an ROC curve for a in {0, 0.01, 0.002, ..., 0.99, 1}
    etas = (0.01, 0.1, 1)
    thresh = np.linspace(0, 1, 101)
    for eta in etas:
        gd = GradientDescent(learning_rate=FixedLR(eta), tol=1e-5, max_iter=1000)
        lr = LogisticRegression(solver=gd)
        lr.fit(X_train, y_train)
        y_pred = lr.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        print(f"Best threshold for eta={eta} is {best_threshold}")
        fig1 = go.Figure(
            data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                             name="Random Class Assignment"),
                  go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                             marker_color='red',
                             hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
            layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}, with eta={eta}$",
                             xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                             yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
        fig1.show()
        break
        # losses = []
        # for threshold in thresh:
        #     y_pred = lr.predict_proba(X_test)
        #     y_pred = (y_pred > threshold).astype(int)
        #     # get score of the model
        #     score = accuracy_score(y_test, y_pred)
        #     losses.append(score)

    gd = GradientDescent(learning_rate=FixedLR(0.01), tol=1e-5, max_iter=1000)



















    # Plotting convergence rate of logistic regression over SA heart disease data
    raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)

    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
