from sklearn.datasets import load_iris
from tensorflow.keras.optimizers import SGD
from artificial_neural_network_model_automation.classification_handler import ANNClassificationHandlerConfig
from artificial_neural_network_model_automation.classification_handler import ANNClassificationHandler
from helper.classification_handler_helper import get_predictions_from_dummy_prob_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import np_utils

# loading data
data = load_iris()
features = data.data
target = data.target

# splitting data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

# dummy target transformation
dummy_y_train = np_utils.to_categorical(y_train)

# designing neural network
optimizer = SGD(learning_rate=0.01)
neural_network_config = {"classification_type": "multiclass",
                         "neural_network_architecture": [4, 9, 12, 9, 3],
                         "hidden_layers_activation_function": "relu",
                         "optimizer": optimizer,
                         "metric": "Recall",
                         "batch_size": 10,
                         "epochs": 50}

ann_classification_handler_config = ANNClassificationHandlerConfig(neural_network_config)

ann_classification_handler = ANNClassificationHandler(ann_classification_handler_config)

# training neural network
ann_classification_handler.train_neural_network(X_train, dummy_y_train)

# getting predictions
dummy_y_test_predictions = ann_classification_handler.classifier.predict(X_test)
y_test_predictions = get_predictions_from_dummy_prob_matrix(dummy_y_test_predictions, [0, 1, 2])

# displaying classification report
print(classification_report(y_test, y_test_predictions))
