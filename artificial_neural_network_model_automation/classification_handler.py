from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


class ANNClassificationHandler:
    def __init__(self, classification_type, number_of_inputs, number_of_hidden_layers,
                 dropout, number_of_categories, metric):
        self.classification_type = classification_type
        self.number_of_inputs = number_of_inputs
        self.number_of_hidden_layers = number_of_hidden_layers
        self.dropout = dropout
        self.number_of_categories = number_of_categories
        self.metric = metric

    def design_neural_network(self):
        # Initialising the ANN
        classifier = Sequential()
        # Adding input layer and first hidden layer
        classifier.add(Dense(self.number_of_inputs, input_dim=self.number_of_inputs, activation='relu'))
        if self.dropout:
            # adding the input layer and the first hidden layer with dropout
            classifier.add(Dropout(rate=0.1))
        # adding other hidden layers
        for i in range(0, self.number_of_hidden_layers-1):
            classifier.add(Dense(self.number_of_inputs + 5, activation='relu'))
            if self.dropout:
                # adding dropout layer
                classifier.add(Dropout(rate=0.1))

        # adding output layer
        if self.classification_type == "binary":
            classifier.add(Dense(self.number_of_categories, activation='sigmoid'))
        else:
            classifier.add(Dense(self.number_of_categories, activation='softmax'))

        # compile classifier
        if self.classification_type == "binary":
            classifier.compile(loss='binary_crossentropy', optimizer="adam", metrics=[self.metric])
        else:
            classifier.compile(loss='categorical_crossentropy', optimizer="adam", metrics=[self.metric])

        return classifier
