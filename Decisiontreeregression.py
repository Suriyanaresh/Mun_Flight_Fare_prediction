import numpy as np_numpy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import *


class Decision_tree_regression:

    def calculate_mean_absolute_percentage_error(self, y_value_true, y_value_pred):
        y_value_true, y_value_pred = np_numpy.array(
            y_value_true), np_numpy.array(y_value_pred)
        return np_numpy.mean(np_numpy.abs((y_value_true - y_value_pred) / y_value_true)) * 100

    def decision_regression(self, X_value_train, X_value_test, Y_value_train, Y_value_test):
        X_value_train = X_value_train
        Y_value_train = Y_value_train
        X_value_test = X_value_test
        Y_value_test = Y_value_test
        depth_of_list = list(range(3, 30))
        parameters_grid = dict(max_depth=depth_of_list)
        decision_tree = GridSearchCV(
            DecisionTreeRegressor(), parameters_grid, cv=10)
        decision_tree.fit(X_value_train, Y_value_train)
        # we are now going to predict the values for train data and test data

        y_trained_value_predicted = decision_tree.predict(X_value_train)
        y_test_value_predicted = decision_tree.predict(X_value_test)
        print("Train Results for Decision Tree Regressor Model:")
        print("Root Mean squared Error: ", sqrt(
            mse(Y_value_train.values, y_trained_value_predicted)))
        print("Mean Absolute % Error: ", round(self.calculate_mean_absolute_percentage_error(
            Y_value_train.values, y_trained_value_predicted)))
        print("R-Squared: ", r2_score(Y_value_train.values, y_trained_value_predicted))

        print('')

        print("Test Results for Decision Tree Regressor Model:")
        print("Root Mean Squared Error: ", sqrt(
            mse(Y_value_test, y_test_value_predicted)))
        print("Mean Absolute % Error: ", round(
            self.calculate_mean_absolute_percentage_error(Y_value_test, y_test_value_predicted)))
        print("R-Squared: ", r2_score(Y_value_test, y_test_value_predicted))
        print("")

    def decision_regression_scores(self, X_value_train, X_value_test, Y_value_train, Y_value_test):
        X_value_train = X_value_train
        Y_value_train = Y_value_train
        X_value_test = X_value_test
        Y_value_test = Y_value_test
        depth_of_list = list(range(3, 30))
        parameters_grid = dict(max_depth=depth_of_list)
        decision_tree = GridSearchCV(
            DecisionTreeRegressor(), parameters_grid, cv=10)
        decision_tree.fit(X_value_train, Y_value_train)

        decision_tree_regression_trainscore = round(
            decision_tree.score(X_value_train, Y_value_train) * 100, 2)
        decision_tree_regression_testscore = round(
            decision_tree.score(X_value_test, Y_value_test) * 100, 2)
        return [decision_tree_regression_trainscore, decision_tree_regression_testscore]
