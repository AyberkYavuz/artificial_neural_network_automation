from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from artificial_neural_network_model_automation.decorators import execution_time


class ANNClassificationHandlerConfig:
    """A configuration for Keras artificial neural network classifier.

    Attributes:
      classification_type: The type of classification task. It takes 2 different values
      which are "binary", "multiclass"
    """
    classification_type: str
    number_of_inputs: int
    number_of_hidden_layers: int
    dropout: bool
    number_of_outputs: str
    metric: object
    batch_size: int
    epochs: int


class ANNClassificationHandler:
    def __init__(self, ann_classification_handler_config: ANNClassificationHandlerConfig):
        """Initializes Keras classifier based on ann_classification_handler_config object attributes

        Args:
          ann_classification_handler_config: ANNClassificationHandlerConfig instance.
        """
        self.__classification_type = ann_classification_handler_config.classification_type
        self.__number_of_inputs = ann_classification_handler_config.number_of_inputs
        self.__number_of_hidden_layers = ann_classification_handler_config.number_of_hidden_layers
        self.__dropout = ann_classification_handler_config.dropout
        self.__number_of_outputs = ann_classification_handler_config.number_of_outputs
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
        classifier.add(Dense(self.__number_of_inputs, input_dim=self.__number_of_inputs, activation='relu'))
        if self.__dropout:
            # adding the input layer and the first hidden layer with dropout
            classifier.add(Dropout(rate=0.1))
        # adding other hidden layers
        for i in range(0, self.__number_of_hidden_layers-1):
            classifier.add(Dense(self.__number_of_inputs + 5, activation='relu'))
            if self.__dropout:
                # adding dropout layer
                classifier.add(Dropout(rate=0.1))

        # adding output layer
        if self.__classification_type == "binary":
            classifier.add(Dense(self.__number_of_outputs, activation='sigmoid'))
        else:
            classifier.add(Dense(self.__number_of_outputs, activation='softmax'))

        # compile classifier
        if self.__classification_type == "binary":
            classifier.compile(loss='binary_crossentropy', optimizer="adam", metrics=[self.__metric])
        else:
            classifier.compile(loss='categorical_crossentropy', optimizer="adam", metrics=[self.__metric])

        return classifier

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
