
def contol_instance_type(object, object_name, type):
    """Contols instance type of given object.

    Args:
      object: Python object to be controlled.
      object_name: str. Name of the object.
      type: Data type of object like str, dict, int etc.
    """
    if isinstance(object, type):
        print("{} data type is valid".format(object_name))
    else:
        raise Exception("Sorry, {} cannot be anything than {}".format(object_name, str(type)))


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


activation_functions = ["relu", "sigmoid", "tanh", "selu", "elu", "exponential"]


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


def is_number_positive(number, variable_name):
    """Checks number is spositive or not.

    Args:
      number: Int. Number.
      variable_name: String. Variable name for printing messages.
    """
    if number > 0:
        print("{} value is valid".format(variable_name))
    else:
        raise Exception("Sorry, {} cannot be less than 0".format(variable_name))


def is_list_empty(lst, variable_name):
    """Checks python list is empty or not.

    Args:
      lst: Python list.
      variable_name: String. Variable name for printing messages.
    """
    if lst:
        print("{} is not empty".format(variable_name))
    else:
        raise Exception("Sorry, {} cannot be empty".format(variable_name))


def check_n_jobs(n_jobs):
    """Checks n_jobs attribute of ANNClassificationRandomizedSearch

    Args:
      n_jobs: int, default=None.Number of jobs to run in parallel.
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
    """
    condition1 = isinstance(n_jobs, int)
    condition2 = n_jobs is None
    if condition1 or condition2:
        print("n_jobs value is valid")
    else:
        print("n_jobs value is not valid")

    if condition1:
        is_number_positive(n_jobs, "n_jobs")
