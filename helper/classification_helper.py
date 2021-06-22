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
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

optimizer_list = [SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl]
optimizer_string_list = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]

metric_list = [Accuracy, BinaryAccuracy, CategoricalAccuracy, TopKCategoricalAccuracy, AUC, Precision, Recall,
               TruePositives, TrueNegatives, FalsePositives, FalseNegatives, PrecisionAtRecall,
               SensitivityAtSpecificity, SpecificityAtSensitivity]
metric_string_list = ["accuracy", "binary_accuracy", "categorical_accuracy", "top_k_categorical_accuracy",
                      "AUC", "Precision", "Recall", "TruePositives", "TrueNegatives", "FalsePositives",
                      "FalseNegatives"]


classification_scoring_dictionary = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "accuracy": accuracy_score,
    "roc_auc": roc_auc_score
}


def check_optimizer_value(opt):
    """Checks optimizer value.

    Args:
      opt: str or `tf.keras.optimizers`. Optimizer.
    Raises:
        Exception: if specified conditions are not met.
    """
    optimizer_instance_result_list = []
    for optimizer_class in optimizer_list:
        optimizer_instance_result = isinstance(opt, optimizer_class)
        optimizer_instance_result_list.append(optimizer_instance_result)

    optimizer_other_type_condition = True in optimizer_instance_result_list
    if isinstance(opt, str) or optimizer_other_type_condition:
        print("optimizer data type is valid")
    else:
        raise Exception("Sorry, optimizer cannot be anything than str or `tf.keras.optimizers`")

    if isinstance(opt, str):
        optimizer_str_condition = opt in optimizer_string_list
        if optimizer_str_condition:
            print("optimizer value is valid")
        else:
            raise Exception("Sorry, optimizer cannot be anything than 'sgd', "
                            "'rmsprop', 'adam', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl'")


def check_classification_metric_value(m):
    """Checks classification metric value.

    Args:
      m: str. Metric.
    Raises:
        Exception: if conditions are not met.
    """
    metric_instance_result_list = []
    for metric_class in metric_list:
        metric_instance_result = isinstance(m, metric_class)
        metric_instance_result_list.append(metric_instance_result)

    metric_other_type_condition = True in metric_instance_result_list
    if isinstance(m, str) or metric_other_type_condition:
        print("metric data type is valid")
    else:
        raise Exception("Sorry, metric cannot be anything than str or `tf.keras.metrics`")

    if isinstance(m, str):
        metric_str_condition = m in metric_string_list
        if metric_str_condition:
            print("metric value is valid")
        else:
            raise Exception("Sorry, metric cannot be anything than 'accuracy', 'binary_accuracy', "
                            "'categorical_accuracy', 'top_k_categorical_accuracy', 'AUC', 'Precision', "
                            "'Recall', 'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives'")


def get_label_based_on_thresold(x, thresold):
    """Generates label of probability x which will be 0 or 1.

    Args:
      x: float. Probability value of target category which was produced by the classifier.
      thresold: float. thresold which will be used for labeling.
    Returns:
        Label of probability x which will be 0 or 1.
    """
    result = None
    if x > thresold:
        result = 1
    else:
        result = 0
    return result


def get_predictions_from_dummy_prob_matrix(dummy_prob_matrix, prediction_column_names, threshold=0.5):
    """Generates single predictions column for target category dummy probability variables.

    Args:
      dummy_prob_matrix: pandas dataframe, series or numpy array. Dummy probability variables for target categories.
      prediction_column_names: list. Target categories.
      threshold: float. thresold which will be used for labeling.
    Returns:
        Single predictions column for target category dummy probability variables.
    """
    dummy_prob_matrix_df = pd.DataFrame(dummy_prob_matrix, columns=prediction_column_names)
    for column in prediction_column_names:
        dummy_prob_matrix_df[column] = dummy_prob_matrix_df[column].apply(
            lambda x: get_label_based_on_thresold(x, thresold=threshold))

    dummy_y_train_pred = dummy_prob_matrix_df.to_numpy()
    predictions = argmax(dummy_y_train_pred, axis=1)
    return predictions


def check_target_categories(target_categories: list):
    """Checks target_categories.
    Args:
        target_categories: list. Target varaible categories.
    Raises:
        Exception: if target_categories is None.
    """
    if target_categories is None:
        raise Exception("target_categories cannot be None in multi-class classification.")