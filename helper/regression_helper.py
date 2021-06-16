from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.metrics import MeanSquaredLogarithmicError
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.metrics import LogCoshError

metric_list = [RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError,
               MeanSquaredLogarithmicError, CosineSimilarity, LogCoshError]

metric_string_list = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
                      "mean_squared_logarithmic_error", "cosine_similarity", "logcosh"]

regression_scoring_list = ["adjusted_r2", "explained_variance", "max_error", "neg_mean_absolute_error",
                           "neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_squared_log_error",
                           "neg_median_absolute_error", "r2", "neg_mean_poisson_deviance", "neg_mean_gamma_deviance",
                           "neg_mean_absolute_percentage_error"]

def check_regression_metric_value(m):
    """Checks regression metric value.

    Args:
      m: String. Metric.
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

