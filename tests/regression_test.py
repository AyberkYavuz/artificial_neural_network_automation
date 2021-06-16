from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from artificial_neural_network_model_automation.artificial_neural_network_handler import ArtificialNeuralNetworkHandlerConfig
from artificial_neural_network_model_automation.artificial_neural_network_handler import ArtificialNeuralNetworkHandler
from helper.regression_helper import regression_report

# loading data
data = load_boston()
features = data.data
target = data.target

# splitting data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

# designing neural network

neural_network_config = {"machine_learning_task": "regression",
                         "neural_network_architecture": [13, 20, 20, 1],
                         "hidden_layers_activation_function": "relu",
                         "optimizer": "adam",
                         "metric": "mean_squared_error",
                         "batch_size": 10,
                         "epochs": 50}

ann_classification_handler_config = ArtificialNeuralNetworkHandlerConfig(neural_network_config)

ann_classification_handler = ArtificialNeuralNetworkHandler(ann_classification_handler_config)

# training neural network
ann_classification_handler.train_neural_network(X_train, y_train)
# making predictions
y_pred = ann_classification_handler.get_predictions(X_test)

number_of_features = X_train.shape[1]
regression_report(y_test, y_pred, number_of_features)
