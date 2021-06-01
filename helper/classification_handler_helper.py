import pandas as pd
from numpy import argmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Ftrl
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import TruePositives
from tensorflow.keras.metrics import TrueNegatives
from tensorflow.keras.metrics import FalsePositives
from tensorflow.keras.metrics import FalseNegatives
from tensorflow.keras.metrics import PrecisionAtRecall
from tensorflow.keras.metrics import SensitivityAtSpecificity
from tensorflow.keras.metrics import SpecificityAtSensitivity

optimizer_list = [SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl]
optimizer_string_list = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]

metric_list = [Accuracy, BinaryAccuracy, CategoricalAccuracy, TopKCategoricalAccuracy, AUC, Precision, Recall,
               TruePositives, TrueNegatives, FalsePositives, FalseNegatives, PrecisionAtRecall,
               SensitivityAtSpecificity, SpecificityAtSensitivity]
metric_string_list = ["accuracy", "binary_accuracy", "categorical_accuracy", "top_k_categorical_accuracy",
                      "AUC", "Precision", "Recall", "TruePositives", "TrueNegatives", "FalsePositives",
                      "FalseNegatives"]

activation_functions = ["relu", "sigmoid", "tanh", "selu", "elu", "exponential"]


def contol_instance_value(object, type):
    """Contols instance value of given object.

    Args:
      object: Python object to be controlled.
      type: Data type of object like str, dict, int etc.
    """
    if isinstance(object, type):
        print("{} data type is valid".format(str(object)))
    else:
        raise Exception("Sorry, {} cannot be anything than {}".format(str(object), str(type)))


def get_label_based_on_thresold(x, thresold):
    """Returns label of probability x which will be 0 or 1.

    Args:
      x: Probability value of target category which was produced by the classifier.
      thresold: thresold which will be used for labeling.
    """
    result = None
    if x > thresold:
        result = 1
    else:
        result = 0
    return result


def get_predictions_from_dummy_prob_matrix(dummy_prob_matrix, prediction_column_names, thresold=0.5):
    """Returns single predictions column for target category dummy probability variables.

    Args:
      dummy_prob_matrix: Dummy probability variables for target categories.
      prediction_column_names: Target categories.
      thresold: thresold which will be used for labeling.
    """
    dummy_prob_matrix_df = pd.DataFrame(dummy_prob_matrix, columns=prediction_column_names)
    for column in prediction_column_names:
        dummy_prob_matrix_df[column] = dummy_prob_matrix_df[column].apply(
            lambda x: get_label_based_on_thresold(x, thresold=thresold))

    dummy_y_train_pred = dummy_prob_matrix_df.to_numpy()
    predictions = argmax(dummy_y_train_pred, axis=1)
    return predictions
