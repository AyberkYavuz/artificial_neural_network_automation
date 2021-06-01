from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model
from helper.decorators import execution_time
from helper.instance_controller import contol_instance_type
from helper.classification_handler_helper import check_classification_type_value
from helper.classification_handler_helper import check_neural_network_architecture_values
from helper.classification_handler_helper import check_hidden_layers_activation_value
from helper.classification_handler_helper import check_dropout_dictionary_values
from helper.classification_handler_helper import check_optimizer_value
from helper.classification_handler_helper import metric_list
from helper.classification_handler_helper import metric_string_list


class ANNClassificationHandlerConfig:
    """A configuration for Keras artificial neural network classifier.

    Attributes:
      classification_type: The type of classification task. It takes 2 different values
                         which are "binary", "multiclass".
      neural_network_architecture: Neural network architecture represented by a python list. For example;
                                [60, 70, 80, 1] means that 60 is input layer, 70 is the first hidden layer neuron number,
                                80 is the second hidden layer neuron number and 1 is output layer.
      hidden_layers_activation_function: hidden layers activation function type. It could be "sigmoid", "relu", "tanh" etc.
                                      Please look at https://keras.io/api/layers/activations/ for detailed information.
      dropout_dictionary: It is a python dictionary that has two attributes which are "dropout" and "dropout_rate".
                       Example usage; {"dropout": True, "dropout_rate": 0.01} which means that adding dropout layer between
                       hiddien layers and dropout rate will be 0.01.
      optimizer: String (name of optimizer) or optimizer instance. See `tf.keras.optimizers`.
      metric: List of metrics to be evaluated by the model during training and testing. Each of this can be a string
           (name of a built-in function), function or a `tf.keras.metrics.Metric` instance. See
           `tf.keras.metrics`. Typically you will use `metrics=['accuracy']`.
      batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will
               default to 32. Do not specify the `batch_size` if your data is in the form of datasets, generators,
               or `keras.utils.Sequence` instances (since they generate batches).
      epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
    """
    def __init__(self, neural_network_config: dict):
        """Compiles Keras classifier based on ann_classification_handler_config object attributes.

        Args:
          neural_network_config: Python dictionary which includes neural network configuration data
        """
        self.classification_type = neural_network_config["classification_type"]
        self.neural_network_architecture = neural_network_config["neural_network_architecture"]
        self.hidden_layers_activation_function = neural_network_config["hidden_layers_activation_function"]
        self.dropout_dictionary = neural_network_config["dropout_dictionary"]
        self.optimizer = neural_network_config["optimizer"]
        self.metric = neural_network_config["metric"]
        self.batch_size = neural_network_config["batch_size"]
        self.epochs = neural_network_config["epochs"]

    @property
    def classification_type(self):
        return self._classification_type

    @classification_type.setter
    def classification_type(self, cl_type):
        contol_instance_type(cl_type, "classification_type", str)
        check_classification_type_value(cl_type)
        self._classification_type = cl_type

    @property
    def neural_network_architecture(self):
        return self._neural_network_architecture

    @neural_network_architecture.setter
    def neural_network_architecture(self, nn_architecture):
        contol_instance_type(nn_architecture, "neural_network_architecture", list)
        check_neural_network_architecture_values(nn_architecture)
        self._neural_network_architecture = nn_architecture

    @property
    def hidden_layers_activation_function(self):
        return self._hidden_layers_activation_function

    @hidden_layers_activation_function.setter
    def hidden_layers_activation_function(self, hlaf):
        contol_instance_type(hlaf, "hidden_layers_activation_function", str)
        check_hidden_layers_activation_value(hlaf)
        self._hidden_layers_activation_function = hlaf

    @property
    def dropout_dictionary(self):
        return self._dropout_dictionary

    @dropout_dictionary.setter
    def dropout_dictionary(self, d_dict):
        contol_instance_type(d_dict, "dropout_dictionary", dict)
        check_dropout_dictionary_values(d_dict)
        self._dropout_dictionary = d_dict

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        check_optimizer_value(opt)
        self._optimizer = opt

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, m):
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

        self._metric = m

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs):
        contol_instance_type(bs, "batch_size", int)

        if bs > 0:
            print("batch_size value is valid")
        else:
            raise Exception("Sorry, batch_size cannot be less than 0")

        self._batch_size = bs

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, ep):
        contol_instance_type(ep, "epochs", int)

        if ep > 0:
            print("epochs value is valid")
        else:
            raise Exception("Sorry, epochs cannot be less than 0")

        self._epochs = ep


class ANNClassificationHandler:
    def __init__(self, ann_classification_handler_config: ANNClassificationHandlerConfig):
        """Compiles Keras classifier based on ann_classification_handler_config object attributes.

        Args:
          ann_classification_handler_config: ANNClassificationHandlerConfig instance.
        """
        self.__classification_type = ann_classification_handler_config.classification_type
        self.__neural_network_architecture = ann_classification_handler_config.neural_network_architecture
        self.__hidden_layers_activation_function = ann_classification_handler_config.hidden_layers_activation_function
        self.__dropout_dictionary = ann_classification_handler_config.dropout_dictionary
        self.__optimizer = ann_classification_handler_config.optimizer
        self.__metric = ann_classification_handler_config.metric
        self.__batch_size = ann_classification_handler_config.batch_size
        self.__epochs = ann_classification_handler_config.epochs
        self.classifier = self.design_neural_network()

    def design_neural_network(self):
        """Designs keras neural network architecture based on ANNClassificationHandlerConfig instance for classification
        task and returns it.
        """
        # Initialising the ANN
        classifier = Sequential()
        # Adding input layer and first hidden layer
        classifier.add(Dense(input_dim=self.__neural_network_architecture[0], units=self.__neural_network_architecture[1],
                             activation=self.__hidden_layers_activation_function))

        # adding other hidden layers, if they exist
        other_hidden_layer_neuron_numbers = self.__neural_network_architecture[2:-1]
        if len(other_hidden_layer_neuron_numbers) > 0:
            for hidden_layer_neuron_number in other_hidden_layer_neuron_numbers:
                if self.__dropout_dictionary["dropout"]:
                    # adding dropout layer
                    classifier.add(Dropout(rate=self.__dropout_dictionary["dropout_rate"]))
                classifier.add(Dense(hidden_layer_neuron_number,
                                     activation=self.__hidden_layers_activation_function))

        # adding output layer
        if self.__classification_type == "binary":
            classifier.add(Dense(self.__neural_network_architecture[-1], activation='sigmoid'))
        else:
            classifier.add(Dense(self.__neural_network_architecture[-1], activation='softmax'))

        # compile classifier
        if self.__classification_type == "binary":
            classifier.compile(loss='binary_crossentropy', optimizer=self.__optimizer, metrics=[self.__metric])
        else:
            classifier.compile(loss='categorical_crossentropy', optimizer=self.__optimizer, metrics=[self.__metric])

        return classifier

    def save_classifier_architecture_plot(self, png_path):
        """Plots the classifier architecture and saves it as a png file.

        Args:
          png_path: string path argument. It must be contain location information of png file
        """
        plot_model(self.classifier, to_file=png_path, show_shapes=True, show_layer_names=True)

    @execution_time
    def train_neural_network(self, X, y):
        """Trains the designed Keras classifier.

        Args:
          X: Feature set. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
          y: labels. Like the input data `X`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `X` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `X` is a dataset, generator,
          or `keras.utils.Sequence` instance, `y` should
          not be specified (since targets will be obtained from `X`).
        """
        self.classifier.fit(X, y, batch_size=self.__batch_size, epochs=self.__epochs)
