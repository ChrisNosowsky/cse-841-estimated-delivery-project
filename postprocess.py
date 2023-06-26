# ==============================================================================
# Postprocessing Stage
#
# Authors: Christopher Nosowsky
#
# ==============================================================================

import sys


def evaluate_loss(preds, actual):
    """
    Evaluates the loss between the predictions and actual data
    :param preds:
    :param actual:
    :return:
    """
    early_loss, late_loss = 0, 0
    for i in range(len(preds)):
        if preds[i] > actual[i]:
            # early shipment
            early_loss += preds[i] - actual[i]
        elif preds[i] < actual[i]:
            # late shipment
            late_loss += actual[i] - preds[i]
    loss = (1 / len(preds)) * (0.4 * early_loss + 0.6 * late_loss)
    return loss


def save_model(filename, model):
    """
    Saves a machine learning model

    :param filename: File name to save as
    :param model: Model to save
    :return:
    """
    print("Saving Model...")
    model.save(filename)
    print("Saved Model.")


def is_debug():
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    else:
        v = gettrace()
        if v is None:
            return False
        else:
            return True
