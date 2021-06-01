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
from helper.helper import contol_instance_type

optimizer_list = [SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl]
optimizer_string_list = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]

metric_list = [Accuracy, BinaryAccuracy, CategoricalAccuracy, TopKCategoricalAccuracy, AUC, Precision, Recall,
               TruePositives, TrueNegatives, FalsePositives, FalseNegatives, PrecisionAtRecall,
               SensitivityAtSpecificity, SpecificityAtSensitivity]
metric_string_list = ["accuracy", "binary_accuracy", "categorical_accuracy", "top_k_categorical_accuracy",
                      "AUC", "Precision", "Recall", "TruePositives", "TrueNegatives", "FalsePositives",
                      "FalseNegatives"]

activation_functions = ["relu", "sigmoid", "tanh", "selu", "elu", "exponential"]


def check_classification_type_value(cl_type):
    """Checks classification type value.

    Args:
      cl_type: String. Classification type.
    """
    classification_type_condition = cl_type in ["binary", "multiclass"]
    if classification_type_condition:
        print("classification_type value is valid")
    else:
        raise Exception("Sorry, classification_type should be 'binary' or 'multiclass'")


def check_neural_network_architecture_values(nn_architecture):
    """Checks neural network architecture values.

    Args:
      nn_architecture: List. Neural network architecture.
    """
    length_of_neural_network_architecture = len(nn_architecture)
    neural_network_architecture_condition_1 = length_of_neural_network_architecture < 3

    if neural_network_architecture_condition_1:
        raise Exception("Sorry, length of neural_network_architecture can't be less than 3")

    for layer in nn_architecture:
        if isinstance(layer, bool):
            raise Exception("Sorry, neural network layer can't be anything than int")
        layer_result_1 = not isinstance(layer, int)
        if layer_result_1:
            raise Exception("Sorry, neural network layer can't be anything than int")
        layer_result_2 = layer < 0
        if layer_result_2:
            raise Exception("Sorry, neural network layer can't be less than 0")

    print("neural_network_architecture value is valid")


def check_hidden_layers_activation_value(hlaf):
    """Checks hidden layers activation value.

    Args:
      hlaf: String. Hidden layer activation function
    """
    hidden_layers_activation_function_condition = hlaf in activation_functions
    if hidden_layers_activation_function_condition:
        print("hidden_layers_activation_function value is valid")
    else:
        raise Exception("Sorry, hidden_layers_activation_function value could be 'relu',"
                        " 'sigmoid', 'tanh', 'selu', 'elu' or 'exponential'.")


def check_dropout_dictionary_values(d_dict):
    """Checks dropout dictionary values.

    Args:
      d_dict: Dictionary. Dropout dictionary.
    """
    dropout_value = d_dict["dropout"]

    contol_instance_type(dropout_value, "dropout_value", bool)

    dropout_rate = d_dict["dropout_rate"]

    contol_instance_type(dropout_rate, "dropout_rate", float)

    if dropout_rate > 0.0:
        print("dropout_rate value is valid")
    else:
        raise Exception("Sorry, dropout_rate cannot be less than 0.0")


def check_optimizer_value(opt):
    """Checks optimizer value.

    Args:
      opt: String or `tf.keras.optimizers`. Optimizer.
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


def check_metric_value(m):
    """Checks metric value.

    Args:
      m: String. Metric.
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
