from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from artificial_neural_network_model_automation.classification_handler import ANNClassificationHandlerConfig
from artificial_neural_network_model_automation.classification_handler import ANNClassificationHandler
from sklearn.metrics import classification_report
from helper.data_path_handler import get_data_path

# loading data
data = load_breast_cancer()
features = data.data
target = data.target

# splitting data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

# designing neural network

neural_network_config = {"classification_type": "binary",
                         "neural_network_architecture": [30, 40, 40, 1],
                         "hidden_layers_activation_function": "relu",
                         "optimizer": "adam",
                         "metric": "Recall",
                         "batch_size": 10,
                         "epochs": 50}
ann_classification_handler_config = ANNClassificationHandlerConfig(neural_network_config)

ann_classification_handler = ANNClassificationHandler(ann_classification_handler_config)

# save the plot of classifier architecture
png_path = get_data_path("neural_network_architecture_breast_cancer_data.png",
                         "neural_network_model_architecture_plots",
                         os_type="mac")
ann_classification_handler.save_classifier_architecture_plot(png_path)
# training neural network
ann_classification_handler.train_neural_network(X_train, y_train)
# making predictions
y_pred = ann_classification_handler.get_predictions(X_test)
# classification report
print(classification_report(y_test, y_pred))

