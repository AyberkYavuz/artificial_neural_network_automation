from helper.helper import contol_instance_type
from helper.classification_handler_helper import check_classification_type_value
from helper.helper import is_list_empty


class ANNClassificationRandomizedSearchConfig:
    """A configuration for Keras artificial neural network classifier.

    Attributes:
      classification_type: The type of classification task. It takes 2 different values
                         which are "binary", "multiclass".
      neural_network_architecture_list: List of neural network architectures that are represented by a python list. For example;
                                [[60, 70, 80, 1], [60, 80, 1], [60, 70, 70, 1]]
      hidden_layers_activation_function_list: List of hidden layers activation function types. For example;
                                          ["sigmoid", "relu", "tanh"]
      dropout_dictionary_list: List of dropout_dictionaries. For example; [{"dropout": True, "dropout_rate": 0.01},
                            {"dropout": True, "dropout_rate": 0.02}]
      optimizer_list: List of optimizers. For example; ["adam", "sgd", "nadam"] or
                  [keras.optimizers.Adam(learning_rate=0.01), keras.optimizers.Adam(learning_rate=0.02)]
      metric_list: List of metrics. For example; ["accuracy", "Recall", "Precision", "AUC"] or
                [tf.keras.metrics.AUC(), tf.keras.metrics.Precision()]
      batch_size_list: List of batch sizes. For example; [32, 64, 128]
      epochs_list: List of epochs. For example; [10, 20, 30, 40, 50, 100]
    """
    def __init__(self, neural_network_config_list_dict: dict):
        """Artificial Neural Network Configuration Lists for Randomized Search

        Args:
          neural_network_config_list_dict: Python dictionary which includes neural network configuration lists data
        """
        self.classification_type = neural_network_config_list_dict["classification_type"]
        self.neural_network_architecture_list = neural_network_config_list_dict["neural_network_architecture_list"]
        self.hidden_layers_activation_function_list = neural_network_config_list_dict["hidden_layers_activation_function_list"]
        self.dropout_dictionary_list = neural_network_config_list_dict["dropout_dictionary_list"]
        self.optimizer_list = neural_network_config_list_dict["optimizer_list"]
        self.metric_list = neural_network_config_list_dict["metric_list"]
        self.batch_size_list = neural_network_config_list_dict["batch_size_list"]
        self.epochs_list = neural_network_config_list_dict["epochs_list"]

    @property
    def classification_type(self):
        return self._classification_type

    @classification_type.setter
    def classification_type(self, cl_type):
        contol_instance_type(cl_type, "classification_type", str)
        check_classification_type_value(cl_type)
        self._classification_type = cl_type

    @property
    def neural_network_architecture_list(self):
        return self._neural_network_architecture_list

    @neural_network_architecture_list.setter
    def neural_network_architecture_list(self, nna_list):
        variable_name = "neural_network_architecture_list"
        contol_instance_type(nna_list, variable_name, list)
        is_list_empty(nna_list, variable_name)
        self._neural_network_architecture_list = nna_list

    @property
    def hidden_layers_activation_function_list(self):
        return self._hidden_layers_activation_function_list

    @hidden_layers_activation_function_list.setter
    def hidden_layers_activation_function_list(self, hlaf_list):
        variable_name = "hidden_layers_activation_function_list"
        contol_instance_type(hlaf_list, variable_name, list)
        is_list_empty(hlaf_list, variable_name)
        self._hidden_layers_activation_function_list = hlaf_list

    @property
    def dropout_dictionary_list(self):
        return self._dropout_dictionary_list

    @dropout_dictionary_list.setter
    def dropout_dictionary_list(self, dd_list):
        variable_name = "dropout_dictionary_list"
        contol_instance_type(dd_list, variable_name, list)
        is_list_empty(dd_list, variable_name)
        self._dropout_dictionary_list = dd_list

    @property
    def optimizer_list(self):
        return self._optimizer_list

    @optimizer_list.setter
    def optimizer_list(self, o_list):
        variable_name = "optimizer_list"
        contol_instance_type(o_list, variable_name, list)
        is_list_empty(o_list, variable_name)
        self._optimizer_list = o_list

    @property
    def metric_list(self):
        return self._metric_list

    @metric_list.setter
    def metric_list(self, m_list):
        variable_name = "metric_list"
        contol_instance_type(m_list, variable_name, list)
        is_list_empty(m_list, variable_name)
        self._metric_list = m_list

    @property
    def batch_size_list(self):
        return self._batch_size_list

    @batch_size_list.setter
    def batch_size_list(self, bs_list):
        variable_name = "batch_size_list"
        contol_instance_type(bs_list, variable_name, list)
        is_list_empty(bs_list, variable_name)
        self._batch_size_list = bs_list

    @property
    def epochs_list(self):
        return self._epochs_list

    @epochs_list.setter
    def epochs_list(self, e_list):
        variable_name = "epochs_list"
        contol_instance_type(e_list, variable_name, list)
        is_list_empty(e_list, variable_name)
        self._epochs_list = e_list
