import numpy as np_numpy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from math import sqrt

from sklearn.linear_model import Lasso
from sklearn.model_selection import *


class Lasso_regressor:

    def calculate_mean_absolute_percentage_error(self, y_value_true, y_value_pred):
        y_value_true, y_value_pred = np_numpy.array(
            y_value_true), np_numpy.array(y_value_pred)
        return np_numpy.mean(np_numpy.abs((y_value_true - y_value_pred) / y_value_true)) * 100

    def Lasso_regression_dataset(self, X_value_train, X_value_test, Y_value_train, Y_value_test):
        X_value_train = X_value_train
        Y_value_train = Y_value_train
        X_value_test = X_value_test
        Y_value_test = Y_value_test

        parameters = {'alpha': [0.0001, 0.001, 0.01,
                                0.1, 1, 10, 100, 1000, 10000, 100000]}
        lasso_regressor_dataset = GridSearchCV(
            Lasso(), parameters, cv=15, scoring='neg_mean_absolute_error', n_jobs=-1)
        lasso_regressor_dataset.fit(X_value_train, Y_value_train)

        y_train_predicted_value = lasso_regressor_dataset.predict(
            X_value_train)
        y_test_predicted_value = lasso_regressor_dataset.predict(X_value_test)

        # Below are the  results for Lasso Regressor Model for the given training  data:

        print("RMSE i.e Root Mean Squared Error value is : ", sqrt(
            mse(Y_value_train.values, y_train_predicted_value)))

        print("Mean Absolute % Error for the training data is : ", round(
            self.calculate_mean_absolute_percentage_error(Y_value_train.values, y_train_predicted_value)))

        print("R-Squared value for the training data is : ",
              r2_score(Y_value_train.values, y_train_predicted_value))

        print("")

        print("Below are the  results for Lasso Regressor Model for the given Test  data:")

        print("RMSE i.e Root Mean Squared Error value is : ",
              sqrt(mse(Y_value_test, y_test_predicted_value)))

        print("MMean Absolute % Error for the test data isr: ", round(
            self.calculate_mean_absolute_percentage_error(Y_value_test, y_test_predicted_value)))

        print("R-Squared value for the test data is: ",
              r2_score(Y_value_test, y_test_predicted_value))

        print("")

    def Lasso_regression_scores(self, X_value_train, X_value_test, Y_value_train, Y_value_test):
        X_value_train = X_value_train
        Y_value_train = Y_value_train
        X_value_test = X_value_test
        Y_value_test = Y_value_test

        parameters = {'alpha': [0.0001, 0.001, 0.01,
                                0.1, 1, 10, 100, 1000, 10000, 100000]}
        lasso_regressor_dataset = GridSearchCV(
            Lasso(), parameters, cv=15, scoring='neg_mean_absolute_error', n_jobs=-1)
        lasso_regressor_dataset.fit(X_value_train, Y_value_train)

        y_train_predicted_value = lasso_regressor_dataset.predict(
            X_value_train)
        y_test_predicted_value = lasso_regressor_dataset.predict(X_value_test)

        lasso_score_regression_train = round(
            lasso_regressor_dataset.score(X_value_train, Y_value_train) * 100, 2)
        lasso_score_regression_test = round(
            lasso_regressor_dataset.score(X_value_test, Y_value_test) * 100, 2)
        return [lasso_score_regression_train, lasso_score_regression_test]
