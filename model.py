# ==============================================================================
# Model Creation Stage
#
# Authors: Christopher Nosowsky
#
# ==============================================================================

import xgboost
import xgboost as xgb

import postprocess
import numpy as np
import tensorflow as tf
from preprocess import PreProcess
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


class XGBoost:
    """
    XGBoost Regression Algorithm
    """

    def __init__(self, preprocess_data: PreProcess):
        self.preprocess_data = preprocess_data

    def xg_boost(self):
        """
        XGBoost model creation, training, and predictions
        :return:
        """
        xgbr = xgb.XGBRegressor()
        xgbr.fit(self.preprocess_data.x_train, self.preprocess_data.y_train)

        train_score = xgbr.score(self.preprocess_data.x_train, self.preprocess_data.y_train)

        print("XGBoost Training Score:", str(train_score))

        pred = xgbr.predict(self.preprocess_data.x_test)

        mse = mean_squared_error(self.preprocess_data.y_test, pred)
        print("XGBoost MSE: " + str(mse))

        loss = postprocess.evaluate_loss(pred, self.preprocess_data.y_test)

        print("XGBoost Loss: " + str(loss))

        return xgbr

    def tune_xg_boost_model(self, xgbr: xgboost.XGBRegressor):
        """
        Tunes XGBoost model and recalculates loss and score

        :param xgbr: XGBoost model
        :return:
        """
        params = {
            # Parameters that we are going to tune.
            'max_depth': [int(x) for x in np.linspace(start=5, stop=20, num=1)],
            'min_child_weight': [int(x) for x in np.linspace(start=1, stop=10, num=1)],
            'eta': [0.3, 0.2, 0.1, 0.05, 0.01, 0.005],
            'subsample': [x / 10 for x in np.linspace(start=1, stop=10, num=1)],
            'colsample_bytree': [x / 10 for x in np.linspace(start=1, stop=10, num=1)],
            'n_estimators': [int(x) for x in np.linspace(start=50, stop=500, num=50)]
        }

        xgb_random_search = RandomizedSearchCV(estimator=xgbr,
                                               param_distributions=params,
                                               n_iter=5,
                                               cv=3,
                                               verbose=2,
                                               random_state=47,
                                               n_jobs=1)
        xgb_random_search.fit(self.preprocess_data.x_train, self.preprocess_data.y_train)
        print(xgb_random_search.best_params_)

        best_params = xgb_random_search.best_params_
        subsample = best_params["subsample"]
        min_child_weight = best_params["min_child_weight"]
        max_depth = best_params["max_depth"]
        eta = best_params["eta"]
        colsample_by_tree = best_params["colsample_bytree"]
        n_estimators = best_params["n_estimators"]

        print("After fine tuning")
        xgbr = xgb.XGBRegressor(n_estimators=n_estimators, verbosity=0, subsample=subsample,
                                min_child_weight=min_child_weight,
                                max_depth=max_depth, eta=eta, colsample_by_tree=colsample_by_tree)
        print(xgbr)
        xgbr.fit(self.preprocess_data.x_train, self.preprocess_data.y_train)
        train_score = xgbr.score(self.preprocess_data.x_train, self.preprocess_data.y_train)
        print("Tuned XGBoost Train score: " + str(train_score))
        pred = xgbr.predict(self.preprocess_data.x_test)
        loss = postprocess.evaluate_loss(pred, self.preprocess_data.y_test)
        print("Tuned XGBoost Loss: " + str(loss))

    @staticmethod
    def get_xgboost_feature_importances(xgbr: xgboost.XGBRegressor, feature_list):
        """
        Get XGBoost feature importance's

        :param feature_list:
        :param xgbr: XGBoost model
        :return:
        """
        d = dict()
        f = xgbr.feature_importances_
        for i, feature in enumerate(feature_list):
            d[feature] = f[i]
        return d

    @staticmethod
    def get_xgboost_max_depth(xgbr: xgboost.XGBRegressor):
        """
        Get XGBoost model max tree depth

        :param xgbr: XGBoost model
        :return:
        """
        return xgbr.max_depth


class LR:
    """
    Linear, Lasso, Ridge Regression Algorithm
    """

    def __init__(self, preprocess_data):
        self.preprocess_data = preprocess_data

    def linear_regression(self):
        """
        Linear Regression model creation, training, and predictions
        :return:
        """
        model = LinearRegression()
        model.fit(self.preprocess_data.x_train, self.preprocess_data.y_train)
        preds = model.predict(self.preprocess_data.x_test)
        rounded_preds = [round(pred) for pred in preds]
        loss = postprocess.evaluate_loss(rounded_preds, self.preprocess_data.y_test)
        print("Linear Regression Loss: " + str(loss))
        # display.display_regression(self.preprocess_data.handling_y_test,
        #                            self.preprocess_data.y_test,
        #                            rounded_preds,
        #                            "linear_regression.png")

    def ridge_regression(self):
        """
        Ridge Regression model creation, training, and predictions
        :return:
        """
        model = Ridge(alpha=0.15)
        model.fit(self.preprocess_data.x_train, self.preprocess_data.y_train)
        preds = model.predict(self.preprocess_data.x_test)
        rounded_preds = [round(pred) for pred in preds]
        loss = postprocess.evaluate_loss(rounded_preds, self.preprocess_data.y_test)
        print("Ridge Regression Loss: " + str(loss))
        # display.display_regression(self.preprocess_data.handling_y_test,
        #                            self.preprocess_data.y_test,
        #                            rounded_preds,
        #                            "ridge_regression.png")

    def lasso_regression(self):
        """
        Lasso Regression model creation, training, and predictions
        :return:
        """
        model = Lasso(alpha=1.0)
        model.fit(self.preprocess_data.x_train, self.preprocess_data.y_train)
        preds = model.predict(self.preprocess_data.x_test)
        rounded_preds = [round(pred) for pred in preds]
        loss = postprocess.evaluate_loss(rounded_preds, self.preprocess_data.y_test)
        print("Lasso Regression Loss: " + str(loss))
        # display.display_regression(self.preprocess_data.handling_y_test,
        #                            self.preprocess_data.y_test,
        #                            rounded_preds,
        #                            "lasso_regression.png")


class RandomForest:

    def __init__(self, preprocess_data: PreProcess):
        self.preprocess_data = preprocess_data

    def random_forest(self):
        """
        Random Forest model creation, training, and predictions
        :return:
        """
        rf = RandomForestRegressor(n_estimators=1000, random_state=42)
        rf.fit(self.preprocess_data.x_train, self.preprocess_data.y_train)
        preds = rf.predict(self.preprocess_data.x_test)
        rounded_preds = [round(pred) for pred in preds]
        errors = abs(preds - self.preprocess_data.y_test)
        print("Random Forest MAE:", round(np.mean(errors), 2))

        loss = postprocess.evaluate_loss(rounded_preds, self.preprocess_data.y_test)
        print("Random Forest Loss: " + str(loss))
        return rf


class NeuralNet:
    """
    Neural Network Algorithm using Keras/Tensorflow
    """

    def __init__(self, preprocess_data: PreProcess):
        self.preprocess_data = preprocess_data
        self.history = None

    def neural_net(self):
        """
        Neural Network model creation, training, and predictions
        :return:
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(12, activation=tf.nn.relu, input_shape=(12,)))
        model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(8, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt)
        self.history = model.fit(self.preprocess_data.x_train, self.preprocess_data.y_train,
                                 validation_split=0.2, epochs=50, batch_size=64)
        model.summary()
        loss = self.history.history['loss'][-1]
        val_loss = self.history.history['val_loss'][-1]

        print('FCNN Training MSE Loss: ' + str(loss))
        print('FCNN Validation MSE Loss: ' + str(val_loss) + '\n\n')

        preds = model.predict(self.preprocess_data.x_test)
        preds = preds.ravel()
        rounded_preds = [round(pred) for pred in preds]
        test_loss = postprocess.evaluate_loss(rounded_preds, self.preprocess_data.y_test)
        print("FCNN Loss: " + str(test_loss))
        return model
