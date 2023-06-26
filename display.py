# ==============================================================================
# Model Displays
#
# Authors: Christopher Nosowsky
#
# ==============================================================================

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz


def xgb_plot_tree(xgbr):
    """
    Plots the XGBoost tree

    :param xgbr:
    :return:
    """
    xgb.plot_tree(xgbr)
    plt.savefig('model_displays/xgboost_tree.png', dpi=300)
    # plt.show()


def display_head_of_data(data, number_of_head_rows):
    """
    Get head of data

    :param data: Dataframe of the delivery dates
    :param number_of_head_rows: Number of rows to display
    :return:
    """
    return data.head(number_of_head_rows)


def describe_data(data):
    """
    Describe data set

    :param data: Dataframe of the delivery dates
    :return:
    """
    return data.describe()


def display_nn_results(history):
    """
    Displays prediction results in a chart

    :param history: History from the model
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)
    plt.figure(1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig_file_name = 'model_displays/neural_net_train_validation_loss.png'
    plt.savefig(fig_file_name)
    # plt.show()


def display_random_forest_full_tree(rf: RandomForestRegressor, feature_list):
    """
    Full tree view of Random Forest model

    :param rf: Random Forest model
    :param feature_list: The list of features
    :return:
    """
    tree = rf.estimators_[5]
    export_graphviz(tree, out_file='model_displays/tree.dot', feature_names=feature_list, rounded=True, precision=1)


def get_feature_data(train, i):
    return train[1:, i:i+1]


def generate_binary_graph(data):
    headers = ["B2C", "C2C"]
    plt.bar(headers, data)
    plt.title("b2c_c2c")
    plt.show()


def generate_bar_graph(headers, values, title):
    plt.bar(headers, values)
    plt.title(title)
    plt.show()


def generate_histogram(data, title, bins):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.show()


def generate_scatterplot(data, title):
    step = 1/data.shape[0]
    x = np.arange(0, 1, step)
    y = data
    plt.scatter(x, y)
    plt.title(title)
    plt.show()


def graph_feature_i_data(train, i):
    data = get_feature_data(train, i)
    if i == 0:
        generate_binary_graph(data)
    if i == 2:
        generate_histogram(data, "declared_handling_days", 10)
    if i == 4:
        headers = np.arange(0, 40)
        values = np.zeros_like(headers)
        for i in range(data.shape[0]):
            values[data[i][0]] += 1
        generate_bar_graph(headers, values, "shipment_method_id")
    if i == 5:
        generate_scatterplot(data, "shipping_fee")
    if i == 6:
        generate_histogram(data, "carrier_min_estimate", 5)
    if i == 7:
        generate_histogram(data, "carrier_max_estimate", 5)
    if i == 10:
        headers = np.arange(0, 40)
        values = np.zeros_like(headers)
        for i in range(data.shape[0]):
            values[data[i][0]] += 1
        generate_bar_graph(headers, values, "category_id")
    if i == 11:
        generate_scatterplot(data, "item_price")
    if i == 12:
        generate_histogram(data, "quantity", 5)
    if i == 15:
        generate_scatterplot(data, "weight")


def get_printable_feature_importances(rf: RandomForestRegressor, feature_list):
    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


def display_feature_importances(rf: RandomForestRegressor, feature_list):
    importances = list(rf.feature_importances_)

    # Set the style
    plt.style.use('fivethirtyeight')

    # list of x locations for plotting
    x_values = list(range(len(importances)))

    # Make a bar chart
    plt.bar(x_values, importances, orientation='vertical')

    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')

    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    fig_file_name = 'model_displays/rf_feature_importances.png'
    plt.subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
    plt.savefig(fig_file_name)
    # plt.show()

def display_xg_boost_feature_importances(xgbr, feature_list):
    plt.clf()
    importances = list(xgbr.feature_importances_)

    # Set the style
    plt.style.use('fivethirtyeight')

    # list of x locations for plotting
    x_values = list(range(len(importances)))

    # Make a bar chart
    plt.bar(x_values, importances, orientation='vertical')

    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')

    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    fig_file_name = 'model_displays/xgbr_feature_importances.png'
    plt.subplots_adjust(left=0.4, right=0.9, bottom=0.4, top=0.9)
    plt.savefig(fig_file_name)

def display_loss_by_model():
    x_values = ['Baseline', 'XGBoost', 'XGBoost Tuned', 'Linear', 'Lasso', 'Ridge', 'FCNN', 'Random Forest']
    losses = [0.675, 0.500, 0.480, 0.525, 0.597, 0.525, 0.491, 0.524]

    plt.bar(x_values, losses, orientation='vertical')
    plt.axhline(y=0.675, color='red')
    plt.xticks(x_values, x_values, rotation=45)
    plt.ylim(0, 0.8)
    plt.yticks(np.arange(0, 0.9, 0.2))

    plt.ylabel('Loss')
    plt.title('Loss by Model')

    fig_file_name = 'model_displays/loss_by_model.png'
    plt.tight_layout()
    plt.savefig(fig_file_name)

# def display_regression(handling_days, y_test, y_pred, filename):
#     print(handling_days)
#     print(y_pred)
#     plt.scatter(handling_days, y_test, color='blue', label='Actual Data')
#     plt.plot(handling_days, y_pred, color='red', linewidth=3)
#     plt.tight_layout()
#     plt.savefig("model_displays/" + filename)
