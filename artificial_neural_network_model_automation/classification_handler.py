from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Ftrl
from keras.utils.vis_utils import plot_model
from artificial_neural_network_model_automation.decorators import execution_time

optimizer_list = [SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl]
optimizer_string_list = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]


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
    classification_type: str
    neural_network_architecture = list
    hidden_layers_activation_function = str
    dropout_dictionary: dict
    optimizer = object
    metric: object
    batch_size: int
    epochs: int

    def check_types_of_attributes(self):
        """Checks the types of attributes, if one of them is not okay, it raises exception.
        """
        if isinstance(self.classification_type, str):
            print("classification_type data type is okay")
        else:
            raise Exception("Sorry, classification_type cannot be anything than str")

        if isinstance(self.neural_network_architecture, list):
            print("neural_network_architecture data type is okay")
        else:
            raise Exception("Sorry, neural_network_architecture cannot be anything than list")

        if isinstance(self.hidden_layers_activation_function, str):
            print("hidden_layers_activation_function data type is okay")
        else:
            raise Exception("Sorry, hidden_layers_activation_function cannot be anything than str")

        if isinstance(self.dropout_dictionary, dict):
            print("dropout_dictionary data type is okay")
        else:
            raise Exception("Sorry, dropout_dictionary cannot be anything than dict")

        optimizer_instance_result_list = []
        for optimizer_class in optimizer_list:
            optimizer_instance_result = isinstance(self.optimizer, optimizer_class)
            optimizer_instance_result_list.append(optimizer_instance_result)

        optimizer_other_type_condition = True in optimizer_instance_result_list
        if isinstance(self.optimizer, str) or optimizer_other_type_condition:
            print("optimizer data type is okay")
        else:
            raise Exception("Sorry, dropout_dictionary cannot be anything than str or `tf.keras.optimizers`")


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
