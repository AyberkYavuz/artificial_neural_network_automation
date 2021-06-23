from helper.regression_helper import regression_scoring_list


def contol_instance_type(object, object_name, type):
    """Contols instance type of given object.

    Args:
      object: Python object to be controlled.
      object_name: str. Name of the object for displaying messages.
      type: Data type of object like str, dict, int etc.
    Raises:
        Exception: if the condition is not met.
    """
    if isinstance(object, type):
        print("{} data type is valid".format(object_name))
    else:
        raise Exception("Sorry, {} cannot be anything than {}".format(object_name, str(type)))


def check_machine_learning_value(ml_task):
    """Checks machine_learning_task value.

    Args:
      ml_task: str. Machine learning task.
    Raises:
        Exception: if classification_type_condition is not met.
    """
    ml_task_condition = ml_task in ["binary", "multiclass", "regression"]
    if ml_task_condition:
        print("machine_learning_task value is valid")
    else:
        raise Exception("Sorry, machine_learning_task should be 'binary', 'multiclass' or 'regression.'")


def check_neural_network_architecture_values(nn_architecture):
    """Checks neural network architecture values.

    Args:
      nn_architecture: list. Neural network architecture.
    Raises:
        Exception: if conditions are not met.
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


def check_output_layer_value(ml_task, output_layer_value):
    """Checks output layer of neural network architecture.
    Args:
        ml_task: str. Machine learning task.
        output_layer_value: int.
    Raises:
        Exception: if conditions are not met.
    """
    ml_task_condition = ml_task in ["binary", "regression"]
    if ml_task_condition:
        if output_layer_value > 1:
            raise Exception("Output layer value cannot be greater than 1, "
                            "when machine_learning_task is binary or regression.")
    else:
        if output_layer_value == 1:
            raise Exception("Output layer value cannot be equal to 1, "
                            "when machine_learning_task is multiclass."
                            "Output layer should be number of categories of target variable.")


activation_functions = ["relu", "sigmoid", "tanh", "selu", "elu", "exponential"]


def check_hidden_layers_activation_value(hlaf):
    """Checks hidden layers activation value.

    Args:
      hlaf: str. Hidden layer activation function
    Raises:
        Exception: if the condition is not met.
    """
    hidden_layers_activation_function_condition = hlaf in activation_functions
    if hidden_layers_activation_function_condition:
        print("hidden_layers_activation_function value is valid")
    else:
        raise Exception("Sorry, hidden_layers_activation_function value could be 'relu',"
                        " 'sigmoid', 'tanh', 'selu', 'elu' or 'exponential'.")


def check_dropout_rate_data_type(d_rate):
    """Checks dropout rate data types.

    Args:
      d_rate: float or None. Dropout rate.
    Raises:
        Exception: if conditions are not met.
    """
    condition1 = isinstance(d_rate, float)
    condition2 = d_rate is None
    if condition1 or condition2:
        print("dropout_rate data type is valid.")
    else:
        raise Exception("Sorry, dropout_rate data can only be float or None.")


def check_dropout_rate_value(d_rate):
    """Checks dropout rate value.

    Args:
      d_rate: float. Dropout rate.
    Raises:
        Exception: if conditions are not met.
    """
    condition = isinstance(d_rate, float)
    if condition:
        if d_rate > 0.0:
            print("dropout_rate value is valid")
        else:
            raise Exception("Sorry, dropout_rate cannot be less than 0.0")


def is_number_positive(number, variable_name):
    """Checks number is spositive or not.

    Args:
      number: int. Number.
      variable_name: str. Variable name for printing messages.
    Raises:
        Exception: if the condition is not met.
    """
    if number > 0:
        print("{} value is valid".format(variable_name))
    else:
        raise Exception("Sorry, {} cannot be less than 0".format(variable_name))


def is_list_empty(lst, variable_name):
    """Checks python list is empty or not.

    Args:
      lst: list.
      variable_name: str. Variable name for printing messages.
    Raises:
        Exception: if conditions are not met.
    """
    if lst:
        print("{} is not empty".format(variable_name))
    else:
        raise Exception("Sorry, {} cannot be empty".format(variable_name))


def check_n_jobs(n_jobs):
    """Checks n_jobs attribute of ANNClassificationRandomizedSearch.

    Args:
      n_jobs: int, default=None. Number of jobs to run in parallel.
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
    Raises:
        Exception: if conditions are not met.
    """
    condition1 = isinstance(n_jobs, int)
    condition2 = n_jobs is None
    if condition1 or condition2:
        print("n_jobs value is valid")
    else:
        raise Exception("n_jobs value is not valid")


def check_instance_type_of_scoring(sc):
    """Checks the instance type of scoring
    Args:
        sc: str or None. Scoring. The selection criteria for the best model.
    Raises:
        Exception: if conditions are not met.
    """
    condition1 = isinstance(sc, str)
    condition2 = sc is None
    if condition1 or condition2:
        print("scoring instance is valid")
    else:
        raise Exception("scoring can't be anything than str or None.")


def check_value_of_scoring(sc, ml_task):
    """Checks the value of scoring.
    Args:
        sc: str or None. Scoring. The selection criteria for the best model.
        ml_task: str. Machine learning task.
    Raises:
        Exception: if conditions are not met.
    """
    condition1 = isinstance(sc, str)
    if condition1:
        if ml_task == "binary":
            if sc in ["accuracy", "roc_auc", "f1", "precision", "recall"]:
                print("scoring value is valid.")
            else:
                raise Exception("scoring value cannot be anything than 'accuracy', 'roc_auc', "
                                "'f1', 'precision', 'recall'")
        elif ml_task == "multiclass":
            if sc in ["f1", "precision", "recall"]:
                print("scoring value is valid.")
            else:
                raise Exception("scoring value cannot be anything than 'f1', 'precision', 'recall'")
        else:
            if sc in regression_scoring_list:
                print("scoring value is valid.")
            else:
                raise Exception('scoring value cannot be anything than "adjusted_r2", "r2"')


def get_value_of_scoring_none_condition(sc, ml_task):
    """Updates sc value, if sc is None.
    Args:
        sc: str or None. Scoring. The selection criteria for the best model.
        ml_task: str. Machine learning task.
    Returns:
        sc: str. Scoring. The selection criteria for the best model.
    """
    if ml_task in ["binary", "multiclass"]:
        if sc is None:
            sc = "f1"
    else:
        if sc is None:
            sc = "adjusted_r2"
    return sc
