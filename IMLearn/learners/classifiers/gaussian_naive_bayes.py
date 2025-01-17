from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
        self.vars_ = np.zeros((self.classes_.size, X.shape[1]))
        self.pi_ = np.zeros(self.classes_.size)
        for i in range(self.classes_.shape[0]):
            self.mu_[i] = np.mean(X[y == self.classes_[i]], axis=0)
            self.pi_[i] = np.sum(y == self.classes_[i]) / y.shape[0]
            self.vars_[i] = np.var(X[y == self.classes_[i]], axis=0, ddof=1)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        responses = np.zeros(X.shape[0])
        like= self.likelihood(X)
        for i in range(X.shape[0]):
            responses[i] = np.argmax(like[i])
        return responses

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        log_likelihoods = np.zeros((X.shape[0], self.classes_.size))
        # calculate the log likelihood for each sample for each class and for each feature
        # use the formula for the likelihood of a gaussian distribution
        for i in range(self.classes_.size):
            log_likelihoods[:, i] = -0.5 * np.sum(np.log(self.vars_[i]) + (X - self.mu_[i]) ** 2 / self.vars_[i], axis=1)
           # add the log of the class probabilities to the log likelihoods
        log_likelihoods += np.log(self.pi_)
        # calculate the likelihoods for each sample
        likelihoods = np.exp(log_likelihoods)
        return likelihoods



    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(self.predict(X), y)



