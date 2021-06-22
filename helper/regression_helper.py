from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.metrics import MeanSquaredLogarithmicError
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.metrics import LogCoshError
from sklearn.metrics import r2_score


def adjusted_r2_score(y_test, y_pred, p):
    """Checks regression metric value.

    Args:
      y_test: list or numpy array. Target variable for test
      y_pred: list or numpy array. Predictions of regressor.
      p: int. Number of independent variables.
    Returns:
        adjusted_r2: float. Adjusted R2
    """
    n = len(y_test)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2


def regression_report(y_test, y_pred, p):
    """Displays regression report.
    Args:
        y_test: list or numpy array. Target variable for test
        y_pred: list or numpy array. Predictions of regressor.
        p: int. Number of independent variables.
    """
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = adjusted_r2_score(y_test, y_pred, p)
    print("Regression Report\n************************")
    print("r2 score: " + str(r2))
    print("adjusted r2 score: " + str(adjusted_r2))


metric_list = [RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError,
               MeanSquaredLogarithmicError, CosineSimilarity, LogCoshError]

metric_string_list = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
                      "mean_squared_logarithmic_error", "cosine_similarity", "logcosh"]

regression_scoring_list = ["adjusted_r2", "r2"]

regression_scoring_dictionary = {
    "adjusted_r2": adjusted_r2_score,
    "r2": r2_score
}


def check_regression_metric_value(m):
    """Checks regression metric value.

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
            raise Exception("Sorry, metric cannot be anything than 'mean_squared_error', 'mean_absolute_error', "
                            "'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', "
                            "'cosine_similarity', 'logcosh'")

