# ==============================================================================
# Main Program for CSE 841 Project
#
# Authors: Christopher Nosowsky
#
# ==============================================================================
import display
import postprocess
from preprocess import PreProcess
from model import XGBoost, LR, RandomForest, NeuralNet
from baseline import Baseline

DELIVERY_DATES = "delivery_dates.csv"
DELIVERY_DATES_DEBUG = "delivery_dates_debug.csv"
BASELINE = "baseline_results.csv"

if __name__ == "__main__":
    print("==== Running Baseline Stage ====")
    b = Baseline(BASELINE)
    baseline = b.get_baseline_target_percentage()
    print("Baseline Target Percentage: " + str(round(baseline, 2)) + "%\n")

    baseline_industrial = b.get_business_target_percentage()
    print("Baseline Business Target Percentage: " + str(round(baseline_industrial, 2)) + "%\n")

    baseline_residential = b.get_residential_target_percentage()
    print("Baseline Residential Target Percentage: " + str(round(baseline_residential, 2)) + "%\n")

    baseline_unknown = b.get_unknown_target_percentage()
    print("Baseline Unknown Target Percentage: " + str(round(baseline_unknown, 2)) + "%\n")

    baseline_loss = b.get_target_loss()
    print("Baseline Target Loss: " + str(baseline_loss))

    print("==== Running PreProcessing Stage ====")
    if postprocess.is_debug():
        print("Debug Mode Activated")
        p = PreProcess(DELIVERY_DATES_DEBUG)
    else:
        p = PreProcess(DELIVERY_DATES)
    p.pre_process()
    feature_list = p.feature_names

    print("Visualize the Data")
    if p.features is not None:
        print(display.display_head_of_data(data=p.dataset, number_of_head_rows=5))
        print(display.describe_data(data=p.dataset))

    print('Training Features Shape:', p.x_train.shape)
    print('Training Labels Shape:', p.y_train.shape)
    print('Testing Features Shape:', p.x_test.shape)
    print('Testing Labels Shape:', p.y_test.shape)

    print("==== PreProcessing Stage Complete ====")

    print("==== Running Model Stage ====")

    print("Running XGBoost")
    xg = XGBoost(p)
    xgbr = xg.xg_boost()
    print("Running Tuned XGBoost")
    xg.tune_xg_boost_model(xgbr)

    feature_importances = xg.get_xgboost_feature_importances(xgbr, feature_list)
    print("XGBoost Feature Importances:")
    print(feature_importances)
    display.display_xg_boost_feature_importances(xgbr, feature_list)

    if xgbr is not None:
        display.xgb_plot_tree(xgbr)

    print("Running Linear Regression")
    lr = LR(p)
    lr.linear_regression()

    print("Running Lasso Regression")
    lr.lasso_regression()

    print("Running Ridge Regression")
    lr.ridge_regression()

    print("Running FCNN")
    nn = NeuralNet(p)
    nn_model = nn.neural_net()

    if nn.history is not None:
        display.display_nn_results(nn.history)

    print("Running Random Forest")
    rf = RandomForest(p)
    rf_model = rf.random_forest()
    if rf_model is not None:
        display.get_printable_feature_importances(rf_model, feature_list)
        display.display_feature_importances(rf_model, feature_list)
        display.display_random_forest_full_tree(rf_model, feature_list)
    display.display_loss_by_model()
    print("==== Model Stage Complete ====")
