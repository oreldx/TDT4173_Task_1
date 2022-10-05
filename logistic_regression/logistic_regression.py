import math

import numpy as np
import numpy.ma
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, max_iterations=1000, learning_rate=0.1, enrichment=False):
        self.max_iterations = max_iterations

        self.X = None
        self.n_samples = None
        self.n_features = None

        self.weights = None
        self.learning_rate = learning_rate
        self.enrichment = enrichment

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        # Set the intercept term 1 (not x0 here)
        self.intercept_term(X)

        # Add features the initial features
        if self.enrichment:
            self.features_enrichment()

        # Retrieve specification data set
        self.n_samples, self.n_features = self.X.shape

        # Initialization of the weights to 0
        self.weights = np.zeros(self.n_features)

        for it in range(self.max_iterations):
            # Stochastic gradient descent (each sample at the time | incremental)
            for i in range(self.n_samples):
                sample = self.X[i, :]
                # Apply logistic function (sigmoid) on linear model
                y_predicted = sigmoid(np.dot(sample, self.weights))
                # Update proportional to the error term * current sample
                dw = (y[i]-y_predicted)*sample
                # Update weights
                self.weights += self.learning_rate * dw

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        # Set the intercept term x0=1
        self.intercept_term(X)

        # Add features the initial features
        if self.enrichment:
            self.features_enrichment()

        # Putting all the prediction in one array
        y_predicted = [sigmoid(np.dot(self.X[i, :], self.weights)) for i in range(self.X.shape[0])]

        # Classify in positive/negative class
        return np.asarray(y_predicted) > 0.5

    def intercept_term(self, X):
        """
        Add the intercept term x0 = 1 to the data (not name x0 here)
        Args:
            X: data as a dataframe

        """
        x0 = np.ones((X.shape[0], 1))
        self.X = np.append(np.asarray(X), x0, axis=1)

    def features_enrichment(self):
        """
        Facilite the adding of features
        - Currently only adding the absolute value of (x0 + x1)
        """
        x = np.asarray([[numpy.absolute(n[0]+n[1])] for n in self.X])
        self.X = np.append(self.X, x, axis=1)

# --- Some utility functions


def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))
