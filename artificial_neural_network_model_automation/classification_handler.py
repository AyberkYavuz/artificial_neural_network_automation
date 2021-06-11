from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model
from helper.decorators import execution_time
from helper.helper import contol_instance_type
from helper.classification_handler_helper import check_classification_type_value
from helper.helper import check_neural_network_architecture_values
from helper.helper import check_hidden_layers_activation_value
from helper.helper import check_dropout_rate_data_type
from helper.helper import check_dropout_rate_value
from helper.classification_handler_helper import check_optimizer_value
from helper.classification_handler_helper import check_metric_value
from helper.helper import is_number_positive
from helper.make_keras_pickable import make_keras_picklable


class ANNClassificationHandlerConfig:
    """A configuration for Keras artificial neural network classifier.

    Attributes:
      classification_type: str. The type of classification task. It takes 2 different values
                         which are "binary", "multiclass".
      neural_network_architecture: Neural network architecture represented by a python list. For example;
                                [60, 70, 80, 1] means that 60 is input layer, 70 is the first hidden layer neuron number,
                                80 is the second hidden layer neuron number and 1 is output layer.
      hidden_layers_activation_function: hidden layers activation function type. It could be "sigmoid", "relu", "tanh" etc.
                                      Please look at https://keras.io/api/layers/activations/ for detailed information.
      dropout_rate: None or float. If it is float, dropout layers between
                hiddien layers will be added. If it is None, dropout layers between hiddien layers will not be added.
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
        """Constructs all the necessary attributes for the ann_classification_handler_config object.

        Args:
          neural_network_config: Python dictionary which includes neural network configuration data.
        """
        self.classification_type = neural_network_config["classification_type"]
        self.neural_network_architecture = neural_network_config["neural_network_architecture"]
        self.hidden_layers_activation_function = neural_network_config["hidden_layers_activation_function"]
        dropout_rate = neural_network_config.get("dropout_rate")
        self.dropout_rate = dropout_rate
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
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, d_rate):
        check_dropout_rate_data_type(d_rate)
        check_dropout_rate_value(d_rate)
        self._dropout_rate = d_rate

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
        check_metric_value(m)
        self._metric = m

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs):
        contol_instance_type(bs, "batch_size", int)
        is_number_positive(bs, "batch_size")
        self._batch_size = bs

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, ep):
        contol_instance_type(ep, "epochs", int)
        is_number_positive(ep, "epochs")
        self._epochs = ep


class ANNClassificationHandler:
    """It designs and trains Keras classifiers

    Attributes:
      __classification_type: str.
      __neural_network_architecture: list.
      __hidden_layers_activation_function: str.
      __dropout_rate: float or None.
      __optimizer: str (name of optimizer) or optimizer instance. See `tf.keras.optimizers`.
      __metric: str (name of a built-in function), function or a `tf.keras.metrics.Metric` instance.
      __batch_size: int or None.
      __epochs: int.
      classifier: Designed Keras classifier.
    """
    def __init__(self, ann_classification_handler_config: ANNClassificationHandlerConfig):
        """Constructs all the necessary attributes for the ann_classification_handler object.

        Args:
          ann_classification_handler_config: ANNClassificationHandlerConfig instance.
        """
        self.__classification_type = ann_classification_handler_config.classification_type
        self.__neural_network_architecture = ann_classification_handler_config.neural_network_architecture
        self.__hidden_layers_activation_function = ann_classification_handler_config.hidden_layers_activation_function
        self.__dropout_rate = ann_classification_handler_config.dropout_rate
        self.__optimizer = ann_classification_handler_config.optimizer
        self.__metric = ann_classification_handler_config.metric
        self.__batch_size = ann_classification_handler_config.batch_size
        self.__epochs = ann_classification_handler_config.epochs
        make_keras_picklable()
        self.classifier = self.design_neural_network()

    def design_neural_network(self):
        """Designs keras neural network architecture based on ANNClassificationHandlerConfig instance for classification
        task.
        Returns:
            classifier: Designed Keras classifier.
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
                if self.__dropout_rate is not None:
                    # adding dropout layer
                    classifier.add(Dropout(rate=self.__dropout_rate))
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
          png_path: str path argument. It must be contain location information of png file
        """
        plot_model(self.classifier, to_file=png_path, show_shapes=True, show_layer_names=True)

    @execution_time("seconds")
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

    def get_predictions(self, X_test, threshold=0.5):
        """Producing predictions of trained Keras classifier
        X_test: array-like of shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and
                n_features is the number of features.
        threshold: float, default=0.5. Threshold.
        Returns:
            y_pred: array-like of shape (n_samples, n_output) \
                    or (n_samples,).
                    Predictions of trained Keras classifier.
        """
        y_pred = None
        if self.__classification_type == "binary":
            y_pred = self.classifier.predict(X_test)
            y_pred = [1 if prob > threshold else 0 for prob in y_pred]
        else:
            print()

        return y_pred
